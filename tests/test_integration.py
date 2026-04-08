"""
End-to-end integration tests: mapper + real solver stack with mocked vision I/O.

Vision is simulated via TileRecord lists and OCR dicts; camera/OCR are not used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable

import numpy as np
import pytest
from unittest.mock import MagicMock

from core.bunch import BunchTracker
from core.grid import GridValidator
from core.scorer import PlacementScorer
from core.solver import SolverLoop, SolverRecommendation
from data.dictionary import Dictionary
from vision.camera import CameraFeed
from vision.mapper import MappedState, StateMapper
from vision.tracker import TileRecord, TileState, Zone

# Match main.py import contract (CameraFeed aliases CameraManager).
assert CameraFeed is not None


@pytest.fixture(scope="session")
def wordlist_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp = tmp_path_factory.mktemp("data")
    wordlist = tmp / "TWL.txt"
    words = [
        # Length-2 runs for incremental horizontal placement of RATE
        "RA",
        "AT",
        "TE",
        "ER",
        "IT",
        "IN",
        "ON",
        "AN",
        "ATE",
        "ARE",
        "ART",
        "ERA",
        "EAR",
        "EAT",
        "RAN",
        "RAT",
        "TAR",
        "TAN",
        "NET",
        "TEN",
        "RET",
        "RATE",
        "TEAR",
        "TARE",
        "EARN",
        "NEAR",
        "RENT",
        "TERN",
        "ANTE",
        "RANT",
        "LEAN",
        "LANE",
        "LATE",
        "TALE",
        "EATER",
        "RATER",
        "QI",
        "AB",
        "CD",
        "EF",
        "HI",
        "NO",
        "TO",
        "SO",
        "LO",
        "OX",
        "XI",
        "ZA",
    ]
    wordlist.write_text("\n".join(w.upper() for w in words), encoding="utf-8")
    return str(wordlist)


@pytest.fixture
def calibration_dict() -> dict:
    return {
        "board_zone_bounds": {
            "top_left": (0, 0),
            "bottom_right": (640, 480),
        },
        "grid_rows": 30,
        "grid_cols": 30,
    }


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_tile(
    tile_id: str,
    letter: str,
    zone: Zone,
    *,
    bbox: tuple[int, int, int, int] = (10, 10, 16, 16),
    is_dump_candidate: bool = False,
    is_occluded: bool = False,
    state: TileState = TileState.ACTIVE,
    grid_pos: tuple[int, int] | None = None,
) -> TileRecord:
    return TileRecord(
        tile_id=tile_id,
        zone=zone,
        bbox=bbox,
        is_occluded=is_occluded,
        state=state,
        is_dump_candidate=is_dump_candidate,
        letter=letter.upper() if letter else None,
        grid_pos=grid_pos,
    )


def make_tracker_state(tiles: list[TileRecord]) -> dict[str, TileRecord]:
    return {t.tile_id: t for t in tiles}


def board_bbox(row: int, col: int, *, rows: int = 30, cols: int = 30) -> tuple[int, int, int, int]:
    x0, y0 = 0.0, 0.0
    x1, y1 = 640.0, 480.0
    w = x1 - x0
    h = y1 - y0
    tx = 0.0 if cols <= 1 else col / (cols - 1)
    ty = 0.0 if rows <= 1 else row / (rows - 1)
    px = x0 + tx * w
    py = y0 + ty * h
    return (int(px - 8), int(py - 8), 16, 16)


def stable_update(
    mapper: StateMapper,
    tracker_state: dict[str, TileRecord],
    ocr_output: dict[str, str],
    max_iters: int = 10,
) -> tuple[MappedState | None, SolverRecommendation | None]:
    mapped: MappedState | None = None
    recommendation: SolverRecommendation | None = None
    for _ in range(max_iters):
        mapped, recommendation = mapper.update(tracker_state, ocr_output)
        if recommendation is not None:
            return mapped, recommendation
    pytest.fail("mapper.update() returned None recommendation after max_iters")


def expected_bunch(hand_count: int, board_count: int, dumped_count: int) -> int:
    return 144 - hand_count - board_count - dumped_count


def _grid_from_mapped_and_place(
    dictionary: Dictionary, mapped: MappedState, rec: SolverRecommendation | None
) -> GridValidator:
    g = GridValidator(dictionary)
    for (r, c), ch in mapped.grid.items():
        assert g.place_letter(r, c, ch)
    if rec is not None and rec.action == "PLACE":
        w = rec.details["word"]
        row, col, dr = rec.details["row"], rec.details["col"], rec.details["direction"]
        for i, ch in enumerate(w):
            rr, cc = (row, col + i) if dr == "H" else (row + i, col)
            if g.get_letter(rr, cc) is None:
                assert g.place_letter(rr, cc, ch)
            else:
                assert g.get_letter(rr, cc) == ch
    assert g.validate_board()
    return g


def assert_reasoning(rec: SolverRecommendation | None) -> None:
    assert rec is not None
    assert isinstance(rec.reasoning, str) and len(rec.reasoning.strip()) > 0


def _bootstrap_hand(bunch: BunchTracker, letters: list[str]) -> None:
    bunch.draw_to_hand([x.upper() for x in letters])


def _tracker_for_hand_and_bunch_size(hand: list[str], bunch_size: int) -> BunchTracker:
    """Real BunchTracker with exact ``bunch_size`` after ``sync_observed_hand(hand)``."""
    bt = BunchTracker()
    bunch = bt.peek_bunch()
    hand_ct = Counter(h.upper() for h in hand)
    for letter, n in hand_ct.items():
        bunch[letter] -= n
        assert bunch[letter] >= 0, f"impossible hand multiset vs tile set: {hand}"
    to_place_total = sum(bunch.values()) - bunch_size
    assert to_place_total >= 0
    placed_list: list[str] = []
    for letter in sorted(bunch.keys()):
        while bunch[letter] > 0 and len(placed_list) < to_place_total:
            placed_list.append(letter)
            bunch[letter] -= 1
    assert sum(bunch.values()) == bunch_size
    draw_all = list(hand_ct.elements()) + placed_list
    bt.draw_to_hand(draw_all)
    bt.place_tiles(placed_list)
    bt.sync_observed_hand([h.upper() for h in hand])
    assert bt.bunch_size() == bunch_size
    return bt


def _place_word_on_grid(grid: GridValidator, word: str, row0: int, col0: int, direction: str) -> None:
    for i, ch in enumerate(word):
        r, c = (row0, col0 + i) if direction == "H" else (row0 + i, col0)
        assert grid.place_letter(r, c, ch)
    assert grid.validate_board()


def _apply_place_to_solver_grid(grid: GridValidator, rec: SolverRecommendation) -> None:
    assert rec.action == "PLACE"
    w = rec.details["word"]
    r, c, dr = rec.details["row"], rec.details["col"], rec.details["direction"]
    for i, ch in enumerate(w):
        rr, cc = (r, c + i) if dr == "H" else (r + i, c)
        ex = grid.get_letter(rr, cc)
        if ex is None:
            assert grid.place_letter(rr, cc, ch)
        else:
            assert ex == ch
    assert grid.validate_board()


class IntegrationTileTracker:
    """
    VisionTracker-compatible hooks for mapper + dump-candidate scenarios.
    Mirrors flag_dump_candidates / confirm_dump / clear_dump_candidates semantics.
    """

    def __init__(self) -> None:
        self._tracks: dict[str, TileRecord] = {}
        self._hand_present: bool = False
        self._hidden_from_active: set[str] = set()

    def seed(self, tiles: Iterable[TileRecord]) -> None:
        self._tracks = {t.tile_id: t for t in tiles}
        self._hidden_from_active.clear()

    def hide_from_active(self, tile_id: str) -> None:
        self._hidden_from_active.add(tile_id)

    def show_in_active(self, tile_id: str) -> None:
        self._hidden_from_active.discard(tile_id)

    def get_active_tiles(self) -> list[TileRecord]:
        out: list[TileRecord] = []
        for tid in sorted(self._tracks):
            r = self._tracks[tid]
            if r.state not in (TileState.ACTIVE, TileState.OCCLUDED):
                continue
            if tid in self._hidden_from_active:
                continue
            out.append(r)
        return out

    def record(self, tile_id: str) -> TileRecord | None:
        return self._tracks.get(tile_id)

    def set_tile_grid_pos(self, tile_id: str, grid_pos: tuple[int, int] | None) -> bool:
        if tile_id not in self._tracks:
            return False
        r = self._tracks[tile_id]
        self._tracks[tile_id] = replace(r, grid_pos=grid_pos)
        return True

    def flag_dump_candidates(self, letter: str) -> None:
        ch = letter.strip()[:1].upper() if letter else ""
        if not ch:
            return
        for tid, r in list(self._tracks.items()):
            if r.state == TileState.CONFIRMED_DUMPED:
                continue
            if r.letter == ch:
                self._tracks[tid] = replace(r, is_dump_candidate=True)

    def clear_dump_candidates(self) -> None:
        for tid, r in list(self._tracks.items()):
            if r.is_dump_candidate:
                self._tracks[tid] = replace(r, is_dump_candidate=False)

    def confirm_dump(self, tile_id: str) -> None:
        if tile_id not in self._tracks:
            return
        r = self._tracks[tile_id]
        if r.state == TileState.CONFIRMED_DUMPED:
            return
        letter = r.letter
        del self._tracks[tile_id]
        for tid, rec in list(self._tracks.items()):
            if letter is not None and rec.letter == letter:
                self._tracks[tid] = replace(rec, is_dump_candidate=False)

    def hand_present(self) -> bool:
        return self._hand_present


@pytest.fixture
def real_solver_stack(wordlist_path: str, calibration_dict: dict):
    dictionary = Dictionary(wordlist_path)
    bunch_tracker = BunchTracker()
    grid_validator = GridValidator(dictionary)
    scorer = PlacementScorer(dictionary)
    solver = SolverLoop(bunch_tracker, grid_validator, scorer, dictionary)

    mock_tracker = MagicMock()
    mock_ocr = MagicMock()

    mapper = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)

    return bunch_tracker, grid_validator, scorer, solver, mapper, mock_tracker, mock_ocr, dictionary


def _fresh_stack(wordlist_path: str, calibration_dict: dict, tracker: MagicMock | IntegrationTileTracker):
    dictionary = Dictionary(wordlist_path)
    bunch_tracker = BunchTracker()
    grid_validator = GridValidator(dictionary)
    scorer = PlacementScorer(dictionary)
    solver = SolverLoop(bunch_tracker, grid_validator, scorer, dictionary)
    mapper = StateMapper(tracker, bunch_tracker, solver, calibration_dict)
    return bunch_tracker, grid_validator, solver, mapper, dictionary


class TestAnchorWord:
    def test_anchor_21_tile_draw(self, wordlist_path: str, calibration_dict: dict) -> None:
        mock_tracker = MagicMock()
        dictionary = Dictionary(wordlist_path)
        bunch_tracker = BunchTracker()
        grid_validator = GridValidator(dictionary)
        solver = SolverLoop(bunch_tracker, grid_validator, PlacementScorer(dictionary), dictionary)
        mapper = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)

        # Rich multiset (legal bag counts) so anchor PLACE beats DUMP heuristics
        letters21 = ["R"] * 3 + ["A"] * 3 + ["T"] * 3 + ["E"] * 3 + ["N"] * 8 + ["I"]
        assert len(letters21) == 21
        _bootstrap_hand(bunch_tracker, letters21)

        tiles: list[TileRecord] = []
        ocr: dict[str, str] = {}
        for i, ch in enumerate(letters21):
            tid = f"h{i:02d}"
            tiles.append(
                make_tile(
                    tid,
                    ch,
                    Zone.HAND,
                    bbox=(400 + (i % 7) * 22, 400 + (i // 7) * 22, 18, 18),
                )
            )
            ocr[tid] = ch

        ts = make_tracker_state(tiles)
        mapped, rec = stable_update(mapper, ts, ocr)
        assert rec is not None
        assert rec.action == "PLACE"
        assert_reasoning(rec)
        word = rec.details["word"]
        assert dictionary.is_word(word)
        hand_ct = Counter(x.upper() for x in letters21)
        for c, n in Counter(word.upper()).items():
            assert hand_ct[c] >= n

        _grid_from_mapped_and_place(dictionary, mapped, rec)
        assert mapped.bunch_size == 144 - 21


class TestIncrementalPlacement:
    def test_each_update_validates_board(
        self, wordlist_path: str, calibration_dict: dict
    ) -> None:
        mock_tracker = MagicMock()
        bunch_tracker = BunchTracker()
        dictionary = Dictionary(wordlist_path)
        grid_validator = GridValidator(dictionary)
        solver = SolverLoop(bunch_tracker, grid_validator, PlacementScorer(dictionary), dictionary)
        mapper = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)

        anchor = "RATE"
        rest = "BCDEFGHIJKLMNOPAA"  # 17 letters
        letters21 = list(anchor + rest)
        _bootstrap_hand(bunch_tracker, letters21)

        tiles: list[TileRecord] = []
        ocr: dict[str, str] = {}
        for i, ch in enumerate(anchor):
            tiles.append(
                make_tile(
                    f"b{i}",
                    ch,
                    Zone.BOARD,
                    bbox=board_bbox(15, 15 + i),
                )
            )
            ocr[f"b{i}"] = ch
        for j, ch in enumerate(rest):
            tid = f"h{j:02d}"
            tiles.append(
                make_tile(
                    tid,
                    ch,
                    Zone.HAND,
                    bbox=(420 + (j % 6) * 20, 420 + (j // 6) * 20, 18, 18),
                )
            )
            ocr[tid] = ch

        for i, ch in enumerate(anchor):
            assert grid_validator.place_letter(15, 15 + i, ch)
        assert grid_validator.validate_board()
        bunch_tracker.place_tiles(list(anchor))

        bunch_after_anchor = bunch_tracker.bunch_size()
        moves_done = 0
        pending_cells: list[tuple[int, int, str]] = []

        while moves_done < 10:
            ts = make_tracker_state(tiles)
            mapped, rec = mapper.update(ts, ocr)
            assert mapped is not None
            assert_reasoning(rec)
            assert bunch_tracker.bunch_size() == bunch_after_anchor

            g_check = GridValidator(dictionary)
            for (r, c), ch in mapped.grid.items():
                assert g_check.place_letter(r, c, ch)
            assert g_check.validate_board()

            if rec.action == "PLACE" and not pending_cells:
                w = rec.details["word"]
                r0, c0, dr = rec.details["row"], rec.details["col"], rec.details["direction"]
                for i, ch in enumerate(w):
                    rr, cc = (r0, c0 + i) if dr == "H" else (r0 + i, c0)
                    if (rr, cc) not in mapped.grid:
                        pending_cells.append((rr, cc, ch))

            hand_tiles = [t for t in tiles if t.zone == Zone.HAND and t.letter]
            if not pending_cells:
                break

            tr, tc, want = pending_cells[0]
            match = next((t for t in hand_tiles if t.letter == want), None)
            if match is None:
                break
            idx = tiles.index(match)
            tiles[idx] = replace(
                match,
                zone=Zone.BOARD,
                bbox=board_bbox(tr, tc),
            )
            assert grid_validator.place_letter(tr, tc, want)
            assert grid_validator.validate_board()
            bunch_tracker.place_tiles([want])
            pending_cells.pop(0)
            moves_done += 1


class TestDumpScenario:
    def test_dump_then_recompute(
        self, wordlist_path: str, calibration_dict: dict
    ) -> None:
        mock_tracker = MagicMock()
        bunch_tracker = BunchTracker()
        dictionary = Dictionary(wordlist_path)
        grid_validator = GridValidator(dictionary)
        solver = SolverLoop(bunch_tracker, grid_validator, PlacementScorer(dictionary), dictionary)
        mapper = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)

        # Only Q and Z are Scrabble 10; four tiles → tie-break picks Q over Z.
        hand_letters = ["Q", "Q", "Z", "Z"]
        _bootstrap_hand(bunch_tracker, hand_letters)

        tiles = [
            make_tile("t0", "Q", Zone.HAND, bbox=(400, 400, 18, 18)),
            make_tile("t1", "Q", Zone.HAND, bbox=(430, 400, 18, 18)),
            make_tile("t2", "Z", Zone.HAND, bbox=(460, 400, 18, 18)),
            make_tile("t3", "Z", Zone.HAND, bbox=(490, 400, 18, 18)),
        ]
        ocr = {t.tile_id: t.letter or "" for t in tiles}

        ts = make_tracker_state(tiles)
        mapped, rec = stable_update(mapper, ts, ocr)
        assert rec.action == "DUMP"
        dump_letter = rec.details["tile"]
        assert dump_letter == "Q"
        assert_reasoning(rec)

        pre_bunch = bunch_tracker.bunch_size()
        drawn = bunch_tracker.peek_next(3)
        assert len(drawn) == 3

        bunch_tracker.dump(dump_letter)
        bunch_tracker.draw_from_bunch(drawn)

        # Remove one dumped Q (t0); add three hand tiles matching bunch draw order
        new_tiles = [t for t in tiles if t.tile_id != "t0"]
        new_tiles.extend(
            [
                make_tile("n1", drawn[0], Zone.HAND, bbox=(400, 420, 18, 18)),
                make_tile("n2", drawn[1], Zone.HAND, bbox=(430, 420, 18, 18)),
                make_tile("n3", drawn[2], Zone.HAND, bbox=(460, 420, 18, 18)),
            ]
        )
        ocr2 = {t.tile_id: (t.letter or "") for t in new_tiles}

        mapper2 = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)
        mapped2, rec2 = mapper2.update(make_tracker_state(new_tiles), ocr2)

        assert mapped2.bunch_size == pre_bunch - 3
        assert rec2.action != "DUMP" or rec2.details.get("tile") != dump_letter
        assert_reasoning(rec2)
        _grid_from_mapped_and_place(dictionary, mapped2, rec2 if rec2.action == "PLACE" else None)


class TestDumpCandidateMultiTile:
    @pytest.fixture
    def dump_stack(self, wordlist_path: str, calibration_dict: dict):
        tracker = IntegrationTileTracker()
        bunch_tracker, grid_validator, solver, mapper, dictionary = _fresh_stack(
            wordlist_path, calibration_dict, tracker
        )
        # Six tiles, all Scrabble ≥ 3 so DUMP tile is B (A would steal _pick_dump_tile)
        hand_letters = ["B", "B", "B", "X", "Z", "K"]
        _bootstrap_hand(bunch_tracker, hand_letters)
        tiles = [
            make_tile("tile_B_a", "B", Zone.HAND, bbox=(400, 400, 18, 18)),
            make_tile("tile_B_b", "B", Zone.HAND, bbox=(430, 400, 18, 18)),
            make_tile("tile_B_c", "B", Zone.HAND, bbox=(460, 400, 18, 18)),
            make_tile("hx", "X", Zone.HAND, bbox=(490, 400, 18, 18)),
            make_tile("hz", "Z", Zone.HAND, bbox=(520, 400, 18, 18)),
            make_tile("hk", "K", Zone.HAND, bbox=(535, 400, 18, 18)),
        ]
        tracker.seed(tiles)
        ocr = {t.tile_id: t.letter or "" for t in tiles}
        return tracker, bunch_tracker, mapper, dictionary, tiles, ocr, solver, grid_validator

    def test_all_n_tiles_flagged_same_frame(self, dump_stack) -> None:
        tracker, _, mapper, _, tiles, ocr, _, _ = dump_stack
        ts = make_tracker_state(tracker.get_active_tiles())
        mapped, rec = stable_update(mapper, ts, ocr)
        assert rec.action == "DUMP"
        assert rec.details["tile"] == "B"
        tracker.flag_dump_candidates("B")
        active = tracker.get_active_tiles()
        bs = [t for t in active if t.letter == "B"]
        assert len(bs) == 3
        assert all(t.is_dump_candidate for t in bs)

    def test_one_exit_deflags_siblings_atomically(self, dump_stack) -> None:
        tracker, _, mapper, _, tiles, ocr, _, _ = dump_stack
        ts = make_tracker_state(tracker.get_active_tiles())
        stable_update(mapper, ts, ocr)
        tracker.flag_dump_candidates("B")
        zones_before = {
            "tile_B_b": tracker.record("tile_B_b").zone,
            "tile_B_c": tracker.record("tile_B_c").zone,
        }
        tracker.confirm_dump("tile_B_a")
        assert tracker.record("tile_B_a") is None
        qb = tracker.record("tile_B_b")
        qc = tracker.record("tile_B_c")
        assert qb is not None and qc is not None
        assert qb.is_dump_candidate is False
        assert qc.is_dump_candidate is False
        assert qb.zone == zones_before["tile_B_b"]
        assert qc.zone == zones_before["tile_B_c"]

    def test_n_minus_1_tiles_revert_to_prior_zone(self, dump_stack, calibration_dict) -> None:
        tracker, bunch_tracker, mapper, dictionary, tiles, ocr, solver, _ = dump_stack
        ts = make_tracker_state(tracker.get_active_tiles())
        stable_update(mapper, ts, ocr)
        tracker.flag_dump_candidates("B")
        zones_before = {
            "tile_B_b": tracker.record("tile_B_b").zone,
            "tile_B_c": tracker.record("tile_B_c").zone,
        }
        tracker.confirm_dump("tile_B_a")
        assert tracker.record("tile_B_b").zone == zones_before["tile_B_b"]
        assert tracker.record("tile_B_c").zone == zones_before["tile_B_c"]
        remaining = [
            tracker.record(tid)
            for tid in ("tile_B_b", "tile_B_c", "hx", "hz", "hk")
        ]
        remaining = [t for t in remaining if t is not None]
        ocr2 = {t.tile_id: t.letter or "" for t in remaining}
        mapper2 = StateMapper(tracker, bunch_tracker, solver, calibration_dict)
        mapped, rec = mapper2.update(make_tracker_state(remaining), ocr2)
        assert_reasoning(rec)
        assert Counter(mapped.hand)["B"] == 2

    def test_occluded_flagged_tile_does_not_confirm(self, dump_stack) -> None:
        tracker, _, mapper, _, tiles, ocr, _, _ = dump_stack
        ts = make_tracker_state(tracker.get_active_tiles())
        stable_update(mapper, ts, ocr)
        tracker.flag_dump_candidates("B")
        qb = tracker.record("tile_B_b")
        assert qb is not None
        tracker._tracks["tile_B_b"] = replace(
            qb,
            state=TileState.OCCLUDED,
            is_occluded=True,
            is_dump_candidate=True,
        )
        tracker._hand_present = True
        tracker.hide_from_active("tile_B_b")
        assert tracker.record("tile_B_b").state == TileState.OCCLUDED
        assert tracker.record("tile_B_b").is_dump_candidate is True
        assert tracker.record("tile_B_a").is_dump_candidate is True
        assert tracker.record("tile_B_c").is_dump_candidate is True
        tracker._hand_present = False
        tracker.confirm_dump("tile_B_b")
        assert tracker.record("tile_B_b") is None

    def test_solver_recomputes_after_confirmed_dump_3_tile_draw(
        self, dump_stack, calibration_dict
    ) -> None:
        tracker, bunch_tracker, mapper, dictionary, tiles, ocr, solver, grid_validator = dump_stack
        ts = make_tracker_state(tracker.get_active_tiles())
        stable_update(mapper, ts, ocr)
        tracker.flag_dump_candidates("B")
        tracker.confirm_dump("tile_B_a")
        bunch_tracker.sync_observed_hand(["B", "B", "X", "Z", "K"])
        bunch_tracker.draw_to_hand(["A", "T"])
        bunch_tracker.place_tiles(["A", "T"])
        assert grid_validator.place_letter(20, 20, "A")
        assert grid_validator.place_letter(20, 21, "T")
        assert grid_validator.validate_board()
        pre_bunch = bunch_tracker.bunch_size()
        drawn = bunch_tracker.peek_next(3)
        bunch_tracker.dump("B")
        bunch_tracker.draw_from_bunch(drawn)
        rest = [
            t
            for t in (
                tracker.record("tile_B_b"),
                tracker.record("tile_B_c"),
                tracker.record("hx"),
                tracker.record("hz"),
                tracker.record("hk"),
            )
            if t is not None
        ]
        rest.extend(
            [
                make_tile("bw0", "A", Zone.BOARD, bbox=board_bbox(20, 20)),
                make_tile("bw1", "T", Zone.BOARD, bbox=board_bbox(20, 21)),
                make_tile("d1", drawn[0], Zone.HAND, bbox=(400, 440, 18, 18)),
                make_tile("d2", drawn[1], Zone.HAND, bbox=(430, 440, 18, 18)),
                make_tile("d3", drawn[2], Zone.HAND, bbox=(460, 440, 18, 18)),
            ]
        )
        for t in rest:
            tracker._tracks[t.tile_id] = t
        ocr3 = {t.tile_id: t.letter or "" for t in rest}
        mapper3 = StateMapper(tracker, bunch_tracker, solver, calibration_dict)
        mapped, rec = stable_update(
            mapper3, make_tracker_state(tracker.get_active_tiles()), ocr3
        )
        assert mapped.bunch_size == pre_bunch - 3
        assert rec.action == "PLACE"
        assert_reasoning(rec)
        w = rec.details["word"]
        assert any(L in w for L in drawn)
        _grid_from_mapped_and_place(dictionary, mapped, rec)


class TestPeelScenario:
    def test_peel_draws_one_from_bunch(
        self, wordlist_path: str, calibration_dict: dict
    ) -> None:
        mock_tracker = MagicMock()
        bunch_tracker = BunchTracker()
        dictionary = Dictionary(wordlist_path)
        grid_validator = GridValidator(dictionary)
        solver = SolverLoop(bunch_tracker, grid_validator, PlacementScorer(dictionary), dictionary)
        mapper = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)

        word = "RATE"
        hand_rest = "ABCD"
        all_letters = list(word + hand_rest)
        _bootstrap_hand(bunch_tracker, all_letters)
        tiles: list[TileRecord] = []
        ocr: dict[str, str] = {}
        for i, ch in enumerate(word):
            tid = f"b{i}"
            tiles.append(make_tile(tid, ch, Zone.BOARD, bbox=board_bbox(0, i)))
            ocr[tid] = ch
        for j, ch in enumerate(hand_rest):
            tid = f"h{j}"
            tiles.append(
                make_tile(tid, ch, Zone.HAND, bbox=(410 + j * 24, 410, 18, 18))
            )
            ocr[tid] = ch
        for i, ch in enumerate(word):
            assert grid_validator.place_letter(0, i, ch)
        bunch_tracker.place_tiles(list(word))

        pre_bunch = bunch_tracker.bunch_size()
        next_letter = bunch_tracker.peek_next(1)[0]
        bunch_tracker.draw_from_bunch([next_letter])
        tiles.append(
            make_tile("peel1", next_letter, Zone.HAND, bbox=(520, 410, 18, 18))
        )
        ocr["peel1"] = next_letter

        ts = make_tracker_state(tiles)
        mapped, rec = mapper.update(ts, ocr)
        assert mapped.bunch_size == pre_bunch - 1
        assert_reasoning(rec)
        if rec.action == "PLACE":
            _grid_from_mapped_and_place(dictionary, mapped, rec)


class TestGameOver:
    def test_game_over_no_moves_empty_bunch(
        self, wordlist_path: str, calibration_dict: dict
    ) -> None:
        mock_tracker = MagicMock()
        dictionary = Dictionary(wordlist_path)
        bunch_tracker = BunchTracker()
        grid_validator = GridValidator(dictionary)
        solver = SolverLoop(bunch_tracker, grid_validator, PlacementScorer(dictionary), dictionary)

        hand = ["Q", "X", "Z"]
        placed: list[str] = []
        bag = bunch_tracker.peek_bunch()
        hand_ct = Counter(h.upper() for h in hand)
        for L, n in hand_ct.items():
            bag[L] -= n
        for L in sorted(bag.elements()):
            if sum(bag.values()) <= 0:
                break
            placed.append(L)
            bag[L] -= 1
        draw_all = list(hand_ct.elements()) + placed
        bunch_tracker.draw_to_hand(draw_all)
        bunch_tracker.place_tiles(placed)
        bunch_tracker.sync_observed_hand([h.upper() for h in hand])
        assert bunch_tracker.bunch_size() == 0

        mapper = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)
        tiles = [
            make_tile("a", "Q", Zone.HAND, bbox=(400, 400, 18, 18)),
            make_tile("b", "X", Zone.HAND, bbox=(430, 400, 18, 18)),
            make_tile("c", "Z", Zone.HAND, bbox=(460, 400, 18, 18)),
        ]
        ocr = {"a": "Q", "b": "X", "c": "Z"}
        mapped, rec = mapper.update(make_tracker_state(tiles), ocr)
        assert rec.action == "GAME_OVER"
        assert_reasoning(rec)


class TestBunchAccuracy:
    def test_sequence_counts(self, wordlist_path: str, calibration_dict: dict) -> None:
        mock_tracker = MagicMock()
        dictionary = Dictionary(wordlist_path)
        bunch_tracker = BunchTracker()
        grid_validator = GridValidator(dictionary)
        solver = SolverLoop(bunch_tracker, grid_validator, PlacementScorer(dictionary), dictionary)
        mapper = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)

        dumped = 0

        def assert_formula(h: int, b: int, d: int) -> None:
            exp = expected_bunch(h, b, d)
            assert 0 <= exp <= 144
            assert bunch_tracker.bunch_size() == exp

        letters21 = list("RATE" + "BCDEFGHIJKLMNOP" + "AA")
        _bootstrap_hand(bunch_tracker, letters21)
        assert_formula(21, 0, dumped)

        tiles: list[TileRecord] = []
        ocr: dict[str, str] = {}
        for i, ch in enumerate("RATE"):
            tiles.append(make_tile(f"b{i}", ch, Zone.BOARD, bbox=board_bbox(10, 10 + i)))
            ocr[f"b{i}"] = ch
        for j, ch in enumerate("BCDEFGHIJKLMNOPAA"):
            tiles.append(
                make_tile(
                    f"h{j}",
                    ch,
                    Zone.HAND,
                    bbox=(420 + (j % 6) * 18, 420, 16, 16),
                )
            )
            ocr[f"h{j}"] = ch
        for i, ch in enumerate("RATE"):
            assert grid_validator.place_letter(10, 10 + i, ch)
        bunch_tracker.place_tiles(list("RATE"))
        assert_formula(17, 4, dumped)

        # Place 5 more from hand (FGHIJ) as isolated cells — no length-2 runs
        five = list("FGHIJ")
        singles = [(12, 10), (14, 12), (16, 14), (18, 16), (20, 18)]
        for ch, (row, col) in zip(five, singles):
            assert grid_validator.place_letter(row, col, ch)
            bunch_tracker.place_tiles([ch])
        for i, ch in enumerate(five):
            idx = next(j for j, t in enumerate(tiles) if t.zone == Zone.HAND and t.letter == ch)
            r, c = singles[i]
            tiles[idx] = replace(
                tiles[idx],
                tile_id=f"x{i}",
                zone=Zone.BOARD,
                bbox=board_bbox(r, c),
            )
        ocr = {t.tile_id: t.letter or "" for t in tiles}

        mapped, rec = mapper.update(make_tracker_state(tiles), ocr)
        assert_reasoning(rec)
        assert mapped.bunch_size == bunch_tracker.bunch_size()
        assert_formula(len(mapped.hand), len(mapped.grid), dumped)

        # Dump one tile from hand, draw 3 (letter must still be in rack)
        dletter = sorted(mapped.hand)[0]
        drawn3 = bunch_tracker.peek_next(3)
        bunch_tracker.dump(dletter)
        bunch_tracker.draw_from_bunch(drawn3)
        dumped = 1
        removed = False
        new_tiles: list[TileRecord] = []
        for t in tiles:
            if not removed and t.zone == Zone.HAND and t.letter == dletter:
                removed = True
                continue
            new_tiles.append(t)
        for i, ch in enumerate(drawn3):
            new_tiles.append(
                make_tile(f"dr{i}", ch, Zone.HAND, bbox=(400 + i * 28, 450, 18, 18))
            )
        ocr2 = {t.tile_id: t.letter or "" for t in new_tiles}
        mapper2 = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)
        mapped2, rec2 = mapper2.update(make_tracker_state(new_tiles), ocr2)
        assert_reasoning(rec2)
        assert mapped2.bunch_size == bunch_tracker.bunch_size()
        assert_formula(len(mapped2.hand), len(mapped2.grid), dumped)

        # Peel: one tile from bunch into hand
        pre_b = bunch_tracker.bunch_size()
        peel_letter = bunch_tracker.peek_next(1)[0]
        bunch_tracker.draw_from_bunch([peel_letter])
        new_tiles.append(
            make_tile("peel", peel_letter, Zone.HAND, bbox=(520, 450, 18, 18))
        )
        ocr3 = {t.tile_id: t.letter or "" for t in new_tiles}
        mapper3 = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)
        mapped3, rec3 = mapper3.update(make_tracker_state(new_tiles), ocr3)
        assert_reasoning(rec3)
        assert bunch_tracker.bunch_size() == pre_b - 1
        assert_formula(len(mapped3.hand), len(mapped3.grid), dumped)

        # Place 3 more from hand (isolated cells)
        take = mapped3.hand[:3]
        spots = [(22, 10), (24, 12), (26, 14)]
        for ch, (row3, c0) in zip(take, spots):
            assert grid_validator.place_letter(row3, c0, ch)
            bunch_tracker.place_tiles([ch])
            idx = next(j for j, t in enumerate(new_tiles) if t.zone == Zone.HAND and t.letter == ch)
            new_tiles[idx] = replace(
                new_tiles[idx],
                zone=Zone.BOARD,
                bbox=board_bbox(row3, c0),
            )
        ocr4 = {t.tile_id: t.letter or "" for t in new_tiles}
        mapper4 = StateMapper(mock_tracker, bunch_tracker, solver, calibration_dict)
        mapped4, rec4 = mapper4.update(make_tracker_state(new_tiles), ocr4)
        assert_reasoning(rec4)
        assert_formula(len(mapped4.hand), len(mapped4.grid), dumped)


class TestReasoningText:
    def test_all_actions_non_empty_reasoning(
        self, wordlist_path: str, calibration_dict: dict
    ) -> None:
        dictionary = Dictionary(wordlist_path)

        # PLACE (anchor)
        mock_t = MagicMock()
        bt = BunchTracker()
        gv = GridValidator(dictionary)
        sol = SolverLoop(bt, gv, PlacementScorer(dictionary), dictionary)
        mp = StateMapper(mock_t, bt, sol, calibration_dict)
        letters = ["R"] * 3 + ["A"] * 3 + ["T"] * 3 + ["E"] * 3 + ["N"] * 8 + ["I"]
        _bootstrap_hand(bt, letters)
        tiles = [
            make_tile(f"id{i}", letters[i], Zone.HAND, bbox=(400 + i, 400, 16, 16))
            for i in range(21)
        ]
        ocr = {f"id{i}": letters[i] for i in range(21)}
        _, r1 = stable_update(mp, make_tracker_state(tiles), ocr)
        assert r1.action == "PLACE"
        assert_reasoning(r1)

        # PLACE (extension): board + hand
        bt2 = BunchTracker()
        gv2 = GridValidator(dictionary)
        sol2 = SolverLoop(bt2, gv2, PlacementScorer(dictionary), dictionary)
        mp2 = StateMapper(mock_t, bt2, sol2, calibration_dict)
        w = "RATE"
        tail = "ABCDE"
        all_l = list(w + tail)
        _bootstrap_hand(bt2, all_l)
        t2: list[TileRecord] = []
        o2: dict[str, str] = {}
        for i, ch in enumerate(w):
            t2.append(make_tile(f"B{i}", ch, Zone.BOARD, bbox=board_bbox(0, i)))
            o2[f"B{i}"] = ch
        for j, ch in enumerate(tail):
            t2.append(make_tile(f"H{j}", ch, Zone.HAND, bbox=(410, 410 + j * 18, 16, 16)))
            o2[f"H{j}"] = ch
        for i, ch in enumerate(w):
            assert gv2.place_letter(0, i, ch)
        bt2.place_tiles(list(w))
        _, r2 = mp2.update(make_tracker_state(t2), o2)
        assert r2.action == "PLACE"
        assert_reasoning(r2)

        # DUMP
        bt3 = BunchTracker()
        gv3 = GridValidator(dictionary)
        sol3 = SolverLoop(bt3, gv3, PlacementScorer(dictionary), dictionary)
        mp3 = StateMapper(mock_t, bt3, sol3, calibration_dict)
        _bootstrap_hand(bt3, ["Q", "Q", "X", "Z", "K", "J"])
        t3 = [
            make_tile("u0", "Q", Zone.HAND, bbox=(400, 400, 18, 18)),
            make_tile("u1", "Q", Zone.HAND, bbox=(420, 400, 18, 18)),
            make_tile("u2", "X", Zone.HAND, bbox=(440, 400, 18, 18)),
            make_tile("u3", "Z", Zone.HAND, bbox=(460, 400, 18, 18)),
            make_tile("u4", "K", Zone.HAND, bbox=(480, 400, 18, 18)),
            make_tile("u5", "J", Zone.HAND, bbox=(500, 400, 18, 18)),
        ]
        o3 = {x.tile_id: x.letter or "" for x in t3}
        _, r3 = stable_update(mp3, make_tracker_state(t3), o3)
        assert r3.action == "DUMP"
        assert_reasoning(r3)

        # RESTRUCTURE — matches solver unit test: AB + CD rows, hand EEE, bunch leaves 20 in pile
        bt4 = _tracker_for_hand_and_bunch_size(["E", "E", "E"], 20)
        gv4 = GridValidator(dictionary)
        _place_word_on_grid(gv4, "AB", 0, 0, "H")
        _place_word_on_grid(gv4, "CD", 2, 0, "H")
        sol4 = SolverLoop(bt4, gv4, PlacementScorer(dictionary), dictionary)
        mp4 = StateMapper(mock_t, bt4, sol4, calibration_dict)
        t4 = [
            make_tile("p00", "A", Zone.BOARD, bbox=board_bbox(0, 0)),
            make_tile("p01", "B", Zone.BOARD, bbox=board_bbox(0, 1)),
            make_tile("p20", "C", Zone.BOARD, bbox=board_bbox(2, 0)),
            make_tile("p21", "D", Zone.BOARD, bbox=board_bbox(2, 1)),
            make_tile("he1", "E", Zone.HAND, bbox=(300, 300, 18, 18)),
            make_tile("he2", "E", Zone.HAND, bbox=(330, 300, 18, 18)),
            make_tile("he3", "E", Zone.HAND, bbox=(360, 300, 18, 18)),
        ]
        o4 = {x.tile_id: x.letter or "" for x in t4}
        _, r4 = stable_update(mp4, make_tracker_state(t4), o4)
        assert r4.action == "RESTRUCTURE"
        assert_reasoning(r4)

        # GAME_OVER
        mock_go = MagicMock()
        bt5 = BunchTracker()
        gv5 = GridValidator(dictionary)
        sol5 = SolverLoop(bt5, gv5, PlacementScorer(dictionary), dictionary)
        hand_go = ["Q", "X", "Z"]
        placed_go: list[str] = []
        bag = bt5.peek_bunch()
        hc = Counter(hand_go)
        for L, n in hc.items():
            bag[L] -= n
        for L in sorted(bag.elements()):
            if sum(bag.values()) <= 0:
                break
            placed_go.append(L)
            bag[L] -= 1
        bt5.draw_to_hand(list(hc.elements()) + placed_go)
        bt5.place_tiles(placed_go)
        bt5.sync_observed_hand([x.upper() for x in hand_go])
        mp5 = StateMapper(mock_go, bt5, sol5, calibration_dict)
        tg = [
            make_tile("g0", "Q", Zone.HAND, bbox=(400, 400, 18, 18)),
            make_tile("g1", "X", Zone.HAND, bbox=(420, 400, 18, 18)),
            make_tile("g2", "Z", Zone.HAND, bbox=(440, 400, 18, 18)),
        ]
        _, r5 = mp5.update(make_tracker_state(tg), {"g0": "Q", "g1": "X", "g2": "Z"})
        assert r5.action == "GAME_OVER"
        assert_reasoning(r5)
