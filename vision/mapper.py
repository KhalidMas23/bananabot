"""
Vision → solver integration: tracker + OCR → grid, hand, bunch size, recommendations.

Pure state translation and solver/dump-candidate wiring only; no game heuristics.

Dump-candidate flags (``flag_dump_candidates`` / ``clear_dump_candidates``) run only
immediately after a **fresh** ``SolverLoop.on_state_change`` call. When the grid+hand
fingerprint matches the previous frame, the solver and dump wiring are both skipped, so
existing tracker flags persist (e.g. a dump candidate that becomes occluded while the
mapped grid and hand stay the same).

Integration notes (overlay / main wiring):
- ``CameraManager`` persists the board zone as a quadrilateral; ``StateMapper`` expects
  ``calibration["board_zone_bounds"]`` (``top_left``, ``bottom_right``) plus top-level
  ``grid_rows`` / ``grid_cols``. Bridging belongs in ``main.py`` (or equivalent), not in
  the mapper.
- Renderers should depend on the live camera feed plus ``MappedState`` from the mapper;
  they should not read ``calibration.json`` for board geometry.
- ``main`` passes ``game_active=False`` until the player presses G with a full rack so the
  solver can emit ``WAIT_START`` instead of GAME_OVER on an empty hand pre-deal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from core.bunch import BunchTracker
from core.solver import SolverLoop, SolverRecommendation
from vision.tracker import TileRecord, TileState, VisionTracker, Zone

logger = logging.getLogger(__name__)

# Integration contract name (implementation is ``VisionTracker``).
TileTracker = VisionTracker

ZoneName = Literal["board", "hand"]


@dataclass
class FrozenTileState:
    """Snapshot from the last non-occluded frame for a tile id."""

    letter: str
    zone: ZoneName
    grid_pos: tuple[int, int] | None


@dataclass
class MappedState:
    grid: dict[tuple[int, int], str]
    hand: list[str]
    bunch_size: int
    last_recommendation: SolverRecommendation | None


def _normalize_letter(raw: str | None) -> str | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    if len(s) != 1 or not ("A" <= s <= "Z"):
        return None
    return s


def _is_confirmed_dumped(rec: TileRecord) -> bool:
    return rec.state == TileState.CONFIRMED_DUMPED


def _is_occluded(rec: TileRecord) -> bool:
    return bool(rec.is_occluded)


def _zone_name(rec: TileRecord) -> str:
    if _is_confirmed_dumped(rec):
        return "confirmed_dumped"
    if rec.zone == Zone.BOARD:
        return "board"
    if rec.zone == Zone.HAND:
        return "hand"
    if rec.zone == Zone.UNKNOWN:
        return "unknown"
    return "unknown"


class StateMapper:
    """
    Maps live tracker records + frame OCR into ``MappedState`` and drives ``SolverLoop``
    when (grid, hand) changes. Manages dump-candidate flags on the tracker.
    """

    def __init__(
        self,
        tracker: TileTracker,
        bunch_tracker: BunchTracker,
        solver: SolverLoop,
        calibration: dict,
    ) -> None:
        self._tracker = tracker
        self._bunch_tracker = bunch_tracker
        self._solver = solver

        bz = calibration.get("board_zone_bounds")
        if not isinstance(bz, dict):
            raise ValueError("calibration must contain board_zone_bounds as a dict")
        required_bz = ("top_left", "bottom_right")
        missing_bz = [k for k in required_bz if k not in bz]
        if missing_bz:
            raise ValueError(
                f"board_zone_bounds missing required keys: {missing_bz} "
                f"(expected {required_bz})"
            )
        tl = bz["top_left"]
        br = bz["bottom_right"]
        if (
            not isinstance(tl, (list, tuple))
            or len(tl) != 2
            or not isinstance(br, (list, tuple))
            or len(br) != 2
        ):
            raise ValueError("board_zone_bounds top_left and bottom_right must be [x, y]")
        rows, cols = calibration.get("grid_rows"), calibration.get("grid_cols")
        if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
            raise ValueError("calibration grid_rows and grid_cols must be positive integers")

        self._x0, self._y0 = float(tl[0]), float(tl[1])
        self._x1, self._y1 = float(br[0]), float(br[1])
        self._rows = rows
        self._cols = cols

        self._frozen: dict[str, FrozenTileState] = {}
        self._last_recommendation: SolverRecommendation | None = None
        self._prev_fingerprint: int | None = None
        self._prev_zone_label: dict[str, str] = {}
        self._prev_occluded: dict[str, bool] = {}
        self._prev_effective_letter: dict[str, str | None] = {}
        self._prev_recommendation_action: str | None = None

    def _pixel_to_grid(self, px: float, py: float) -> tuple[int, int]:
        w = self._x1 - self._x0
        h = self._y1 - self._y0
        tx = 0.0 if w == 0 else (px - self._x0) / w
        ty = 0.0 if h == 0 else (py - self._y0) / h
        col_f = tx * (self._cols - 1)
        row_f = ty * (self._rows - 1)
        col = int(round(col_f))
        row = int(round(row_f))
        col = max(0, min(self._cols - 1, col))
        row = max(0, min(self._rows - 1, row))
        return (row, col)

    def _centroid(self, rec: TileRecord) -> tuple[float, float]:
        x, y, bw, bh = rec.bbox
        return (x + 0.5 * bw, y + 0.5 * bh)

    def _effective_letter(
        self, tile_id: str, rec: TileRecord, ocr_output: dict[str, str]
    ) -> str | None:
        if tile_id in ocr_output:
            return _normalize_letter(ocr_output[tile_id])
        return _normalize_letter(rec.letter)

    def _prune_frozen(self, tracker_state: dict[str, TileRecord]) -> None:
        dead = [tid for tid in self._frozen if tid not in tracker_state]
        for tid in dead:
            del self._frozen[tid]
            logger.debug("frozen cache dropped missing tile_id=%s", tid)

    def _build_grid_and_hand(
        self,
        tracker_state: dict[str, TileRecord],
        ocr_output: dict[str, str],
    ) -> tuple[dict[tuple[int, int], str], list[str]]:
        grid: dict[tuple[int, int], str] = {}
        hand: list[str] = []

        # Deterministic iteration order
        for tile_id in sorted(tracker_state.keys()):
            rec = tracker_state[tile_id]

            if _is_confirmed_dumped(rec):
                continue

            occ = _is_occluded(rec)
            prev_occ = self._prev_occluded.get(tile_id, False)
            if occ and not prev_occ:
                logger.debug("occlusion freeze: tile_id=%s", tile_id)
            if not occ and prev_occ:
                logger.debug("occlusion thaw: tile_id=%s", tile_id)

            if occ:
                st = self._frozen.get(tile_id)
                if st is None or not st.letter:
                    self._tracker.set_tile_grid_pos(tile_id, None)
                    continue
                if st.zone == "board" and st.grid_pos is not None:
                    # Push last known board cell onto the live record before grid merge
                    # so overlay sees ``grid_pos`` while ``is_occluded`` is already True.
                    self._tracker.set_tile_grid_pos(tile_id, st.grid_pos)
                    grid[st.grid_pos] = st.letter
                elif st.zone == "hand":
                    hand.append(st.letter)
                    self._tracker.set_tile_grid_pos(tile_id, None)
                elif st.zone == "board" and st.grid_pos is None:
                    # Board frozen snapshot has no cell yet (e.g. placing); never clears a
                    # non-None grid_pos — that case is handled by the branch above.
                    self._tracker.set_tile_grid_pos(tile_id, None)
                continue

            letter = self._effective_letter(tile_id, rec, ocr_output)
            if tile_id in ocr_output and _normalize_letter(ocr_output[tile_id]):
                logger.debug(
                    "OCR letter assignment: tile_id=%s letter=%s",
                    tile_id,
                    _normalize_letter(ocr_output[tile_id]),
                )
            prev_l = self._prev_effective_letter.get(tile_id)
            if letter != prev_l and letter is not None:
                logger.debug(
                    "effective letter update: tile_id=%s %r -> %r",
                    tile_id,
                    prev_l,
                    letter,
                )

            if rec.zone == Zone.BOARD:
                if not letter:
                    self._frozen.pop(tile_id, None)
                    self._tracker.set_tile_grid_pos(tile_id, None)
                    continue
                px, py = self._centroid(rec)
                row, col = self._pixel_to_grid(px, py)
                grid_pos = (row, col)
                logger.debug(
                    "grid snap: tile_id=%s px=%.1f py=%.1f -> (row=%d col=%d) letter=%s",
                    tile_id,
                    px,
                    py,
                    row,
                    col,
                    letter,
                )
                self._tracker.set_tile_grid_pos(tile_id, grid_pos)
                grid[grid_pos] = letter
                self._frozen[tile_id] = FrozenTileState(
                    letter=letter, zone="board", grid_pos=grid_pos
                )
            elif rec.zone == Zone.HAND:
                if not letter:
                    self._frozen.pop(tile_id, None)
                    self._tracker.set_tile_grid_pos(tile_id, None)
                    continue
                hand.append(letter)
                self._frozen[tile_id] = FrozenTileState(
                    letter=letter, zone="hand", grid_pos=None
                )
                self._tracker.set_tile_grid_pos(tile_id, None)
            else:
                self._frozen.pop(tile_id, None)
                self._tracker.set_tile_grid_pos(tile_id, None)

        return grid, hand

    def _log_zone_transitions(self, tracker_state: dict[str, TileRecord]) -> None:
        for tile_id in sorted(tracker_state.keys()):
            z = _zone_name(tracker_state[tile_id])
            prev = self._prev_zone_label.get(tile_id)
            if prev is not None and prev != z:
                logger.debug(
                    "zone transition: tile_id=%s %s -> %s",
                    tile_id,
                    prev,
                    z,
                )

    def _apply_dump_candidate_wiring(self, recommendation: SolverRecommendation) -> None:
        action = recommendation.action
        prev = self._prev_recommendation_action

        if action == "DUMP":
            tile = recommendation.details["tile"]
            if isinstance(tile, str) and tile:
                self._tracker.flag_dump_candidates(tile)
        elif prev == "DUMP":
            self._tracker.clear_dump_candidates()

        self._prev_recommendation_action = action

    def update(
        self,
        tracker_state: dict[str, TileRecord],
        ocr_output: dict[str, str],
        *,
        game_active: bool = True,
    ) -> tuple[MappedState, SolverRecommendation]:
        self._prune_frozen(tracker_state)
        self._log_zone_transitions(tracker_state)

        grid, hand = self._build_grid_and_hand(tracker_state, ocr_output)
        bunch_size = self._bunch_tracker.bunch_size()

        fingerprint = hash((frozenset(grid.items()), tuple(sorted(hand)), game_active))

        if self._prev_fingerprint is not None and fingerprint == self._prev_fingerprint:
            # No dump wiring here: only runs after a fresh on_state_change (below or cache miss).
            if self._last_recommendation is None:
                logger.warning(
                    "solver cache miss on static frame; recomputing (hand_size=%d grid_cells=%d)",
                    len(hand),
                    len(grid),
                )
                rec_out = self._solver.on_state_change(hand, game_active=game_active)
                self._last_recommendation = rec_out
                self._apply_dump_candidate_wiring(rec_out)
            else:
                rec_out = self._last_recommendation
            mapped = MappedState(
                grid=dict(grid),
                hand=list(hand),
                bunch_size=bunch_size,
                last_recommendation=self._last_recommendation,
            )
            self._advance_prev_maps(tracker_state, ocr_output)
            return mapped, rec_out

        logger.info(
            "solver call triggered: hand_size=%d grid_cells=%d",
            len(hand),
            len(grid),
        )
        recommendation = self._solver.on_state_change(hand, game_active=game_active)
        self._last_recommendation = recommendation
        self._prev_fingerprint = fingerprint
        self._apply_dump_candidate_wiring(recommendation)

        mapped = MappedState(
            grid=dict(grid),
            hand=list(hand),
            bunch_size=bunch_size,
            last_recommendation=self._last_recommendation,
        )
        self._advance_prev_maps(tracker_state, ocr_output)
        return mapped, recommendation

    def _advance_prev_maps(
        self,
        tracker_state: dict[str, TileRecord],
        ocr_output: dict[str, str],
    ) -> None:
        active = set(tracker_state.keys())
        for d in (self._prev_occluded, self._prev_zone_label, self._prev_effective_letter):
            for tile_id in list(d.keys()):
                if tile_id not in active:
                    del d[tile_id]
        for tile_id in sorted(tracker_state.keys()):
            rec = tracker_state[tile_id]
            self._prev_zone_label[tile_id] = _zone_name(rec)
            self._prev_occluded[tile_id] = _is_occluded(rec)
            if not _is_occluded(rec):
                self._prev_effective_letter[tile_id] = self._effective_letter(
                    tile_id, rec, ocr_output
                )
