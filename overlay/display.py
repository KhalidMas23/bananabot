"""
Display loop helpers: window management, two-panel composition, and keyboard routing.

Holds UI state (reasoning visibility, game-over lock). Does not run the solver or
mutate game logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import cv2
import numpy as np
import numpy.typing as npt

import overlay.renderer as renderer
from core.solver import SolverRecommendation
from vision.opencv_util import safe_destroy_window
from vision.tracker import TileRecord, Zone

WINDOW_TITLE: str = "BananaBot"


def _action_summary_line(recommendation: SolverRecommendation | None) -> str:
    """First line of ``reasoning`` is always the action summary (PRD §3.8)."""
    if recommendation is None:
        return "Action: —"
    raw = recommendation.reasoning or ""
    lines = raw.split("\n")
    if not lines or not lines[0].strip():
        return f"Action: {recommendation.action.replace('_', ' ')}"
    return lines[0].strip()


def _reasoning_body_verbatim(recommendation: SolverRecommendation | None) -> str:
    """Lines after the first newline of ``reasoning``, shown verbatim when expanded."""
    if recommendation is None:
        return ""
    raw = recommendation.reasoning or ""
    if "\n" not in raw:
        return ""
    return raw.split("\n", 1)[1]


def _dump_letters_from_tracker(tracker_state: Mapping[str, TileRecord]) -> list[str]:
    """Collect letters for tiles flagged as dump candidates (for grid highlighting)."""
    out: list[str] = []
    for rec in tracker_state.values():
        if rec.is_dump_candidate and rec.letter:
            out.append(rec.letter.upper())
    return out


def _any_dump_candidate(tracker_state: Mapping[str, TileRecord]) -> bool:
    return any(rec.is_dump_candidate for rec in tracker_state.values())


def _occluded_board_cells(tracker_state: Mapping[str, TileRecord]) -> set[tuple[int, int]]:
    """
    Board cells whose tile is occluded on camera.

    Skips tiles with ``grid_pos is None`` (e.g. first frame occluded before any snap).
    """
    out: set[tuple[int, int]] = set()
    for rec in tracker_state.values():
        if rec.zone != Zone.BOARD or not rec.is_occluded or rec.grid_pos is None:
            continue
        out.add((int(rec.grid_pos[0]), int(rec.grid_pos[1])))
    return out


class DisplayManager:
    """
    Compose the camera panel and digital grid, draw recommendation text, and handle
    simple key commands.

    Attributes:
        reasoning_visible: When True, the full reasoning body is shown in the text strip.
    """

    def __init__(
        self,
        right_panel_width: int = 520,
        text_strip_height: int = 140,
    ) -> None:
        """
        Args:
            right_panel_width: Pixel width of the grid + recommendation column.
            text_strip_height: Vertical space reserved under the grid for recommendation text.
        """
        self._right_w = max(120, right_panel_width)
        self._text_h = max(48, text_strip_height)
        self.reasoning_visible: bool = False
        self._game_over: bool = False
        self._game_over_recommendation: SolverRecommendation | None = None
        self._frozen_grid: dict[tuple[int, int], str] = {}
        self.last_recommendation: SolverRecommendation | None = None
        self._window_created: bool = False

    def toggle_reasoning(self) -> None:
        """Toggle visibility of the full reasoning body (``r`` key)."""
        self.reasoning_visible = not self.reasoning_visible

    def set_game_over(
        self,
        recommendation: SolverRecommendation,
        grid_state: dict[tuple[int, int], str],
    ) -> None:
        """
        Lock the display: grid stops updating, recommendation text freezes to this
        object, and a GAME OVER overlay is drawn each frame.

        ``grid_state`` is snapshotted immediately so ordering relative to :meth:`update`
        does not matter. The game-over overlay always uses this recommendation's
        ``reasoning`` text, not any later value from live ``update()`` calls.
        """
        self._game_over = True
        self._game_over_recommendation = recommendation
        self._frozen_grid = deepcopy(grid_state)

    def run_key_handler(self, key: int) -> bool:
        """
        Handle keyboard input from ``cv2.waitKey``.

        Returns:
            True if the application should quit (``q``), False otherwise.
            ``r`` toggles the reasoning panel.
        """
        k = key & 0xFF
        if k in (ord("q"), ord("Q")):
            return True
        if k in (ord("r"), ord("R")):
            self.toggle_reasoning()
        return False

    def update(
        self,
        frame: npt.NDArray[np.uint8],
        grid_state: dict[tuple[int, int], str],
        recommendation: SolverRecommendation | None,
        tracker_state: Mapping[str, TileRecord],
        *,
        warp_to_source_3x3: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """
        Compose camera + grid + recommendation text and show ``WINDOW_TITLE``.

        When not in game-over mode, stores the latest recommendation on
        ``last_recommendation``.

        ``warp_to_source_3x3`` maps rectified tile bboxes onto ``frame`` when the
        camera panel shows the raw full-FOV image; omit for legacy warped-only frames.
        """
        if not self._game_over:
            self.last_recommendation = recommendation

        grid_draw: dict[tuple[int, int], str] = (
            self._frozen_grid if self._game_over else grid_state
        )
        rec_draw: SolverRecommendation | None = (
            self._game_over_recommendation if self._game_over else recommendation
        )

        cam = renderer.draw_camera_panel(
            frame, tracker_state, warp_to_source_3x3=warp_to_source_3x3
        )
        cam = renderer.draw_dump_candidates_camera(
            cam, tracker_state, warp_to_source_3x3=warp_to_source_3x3
        )

        th = int(frame.shape[0])
        grid_h = max(1, th - self._text_h)

        occ = _occluded_board_cells(tracker_state)
        gp = renderer.draw_grid_panel(
            grid_draw, rec_draw, (self._right_w, grid_h), occluded_cells=occ
        )
        gp = renderer.draw_dump_candidates_grid(
            gp, grid_draw, _dump_letters_from_tracker(tracker_state)
        )

        right_col = np.full((th, self._right_w, 3), (48, 48, 48), dtype=np.uint8)
        right_col[:grid_h, :] = gp

        if not self._game_over:
            action_line = _action_summary_line(rec_draw)
            body = _reasoning_body_verbatim(rec_draw)
            renderer.draw_reasoning_panel(
                right_col,
                0,
                grid_h,
                self._right_w,
                action_line,
                body,
                expanded=self.reasoning_visible,
                show_dump_hint=_any_dump_candidate(tracker_state),
            )

        composite = renderer.hstack_panels(cam, right_col)

        if self._game_over and self._game_over_recommendation is not None:
            renderer.draw_game_over_overlay(
                composite, self._game_over_recommendation.reasoning
            )

        if not self._window_created:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
            self._window_created = True
        cv2.imshow(WINDOW_TITLE, composite)

    def destroy_window(self) -> None:
        """Release the OpenCV window if it was created."""
        if self._window_created:
            safe_destroy_window(WINDOW_TITLE)
            self._window_created = False
