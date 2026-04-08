"""Unit tests for core.solver.SolverLoop."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.bunch import BunchTracker
from core.grid import GridValidator
from core.scorer import PlacementScorer, SCRABBLE_VALUES
from core.solver import (
    SolverLoop,
    _best_placement_for_hand,
    _peek_first_n_from_counter,
    _simulate_post_dump_hand,
)
from data.dictionary import Dictionary


class PeelBunch(BunchTracker):
    """peek_next(1) returns Z for peel simulation (post-dump uses peek_bunch + returned tile)."""

    def peek_next(self, n: int = 3) -> list[str]:
        if n == 1:
            return ["Z"]
        return ["A", "A", "A"]


class StubDictionary:
    """No wordlist file; set-based words + multiset get_words_from_letters."""

    def __init__(self, words: set[str]) -> None:
        self._words = {w.upper() for w in words}

    def is_word(self, word: str) -> bool:
        return word.upper() in self._words

    def is_prefix(self, prefix: str) -> bool:
        p = prefix.upper()
        if not p:
            return bool(self._words)
        return any(w.startswith(p) for w in self._words)

    def get_words_from_letters(self, available_letters: list[str]) -> list[str]:
        counts: Counter[str] = Counter()
        for ch in available_letters:
            u = ch.upper()
            if len(u) == 1 and "A" <= u <= "Z":
                counts[u] += 1
        found: set[str] = set()
        for w in self._words:
            if len(w) < 2:
                continue
            wc = Counter(w)
            if all(counts[c] >= wc[c] for c in wc):
                found.add(w)
        return sorted(found)


def _tracker_for_hand_and_bunch_size(hand: list[str], bunch_size: int) -> BunchTracker:
    """Real BunchTracker with exact ``bunch_size`` after sync_observed_hand(hand)."""
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


def _tracker_with_exact_remaining(hand: list[str], remaining: Counter[str]) -> BunchTracker:
    """BunchTracker whose bunch multiset equals ``remaining`` after sync_observed_hand(hand)."""
    bt = BunchTracker()
    bag = bt.peek_bunch()
    hand_ct = Counter(h.upper() for h in hand)
    for letter, n in hand_ct.items():
        bag[letter] -= n
    rem = remaining.copy()
    for letter, n in rem.items():
        assert bag[letter] >= n, f"{letter} not available in remainder for bunch {remaining}"
    placed = bag.copy()
    for letter, n in rem.items():
        placed[letter] -= n
    assert all(placed[k] >= 0 for k in placed)
    placed_list: list[str] = []
    for letter in sorted(placed.keys()):
        while placed[letter] > 0:
            placed_list.append(letter)
            placed[letter] -= 1
    draw_all = list(hand_ct.elements()) + placed_list
    bt.draw_to_hand(draw_all)
    bt.place_tiles(placed_list)
    bt.sync_observed_hand([h.upper() for h in hand])
    assert bt.peek_bunch() == Counter({k: v for k, v in remaining.items() if v > 0})
    return bt


def _solver_with(
    words: set[str],
    bunch: BunchTracker | MagicMock | None = None,
    grid: GridValidator | None = None,
) -> tuple[SolverLoop, StubDictionary, GridValidator, BunchTracker | MagicMock]:
    d = StubDictionary(words)
    g = grid or GridValidator(d)
    b = bunch if bunch is not None else BunchTracker()
    sc = PlacementScorer(d)
    return SolverLoop(b, g, sc, d), d, g, b


def _place_word_on_grid(grid: GridValidator, word: str, row0: int, col0: int, direction: str) -> None:
    for i, ch in enumerate(word):
        r, c = (row0, col0 + i) if direction == "H" else (row0 + i, col0)
        assert grid.place_letter(r, c, ch)
    assert grid.validate_board()


def _apply_place_recommendation(grid: GridValidator, rec) -> None:
    """Place only empty cells for a PLACE recommendation (shared anchors may already exist)."""
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


def test_empty_board_anchor_prefers_higher_scrabble_word() -> None:
    words = {"ZOO", "CAT", "AT", "ZO", "OO"}
    d = StubDictionary(words)
    sol, _, _, bunch = _solver_with(words)
    hand = ["Z", "O", "O", "C", "A", "T"]
    bunch.sync_observed_hand(hand)
    rec = sol.on_state_change(hand)
    assert rec.action == "PLACE"
    assert rec.details["word"] == "ZOO"
    want = PlacementScorer(d).score_anchor("ZOO", hand)
    assert rec.score == pytest.approx(want)
    assert "ZOO" in rec.reasoning
    assert "12" in rec.reasoning or "score" in rec.reasoning.lower()


def test_subsequent_placement_validates_and_recommends() -> None:
    words = {"AT", "CAT", "CA", "HI", "IT"}
    bunch = BunchTracker()
    bunch.peek_next = lambda n=3: ["X", "X", "X"]  # type: ignore[method-assign]
    bunch.peek_bunch = lambda: Counter({"X": 100})  # type: ignore[method-assign]
    sol, _, grid, bunch = _solver_with(words, bunch=bunch)
    bunch.sync_observed_hand(["C", "A", "T"])
    rec0 = sol.on_state_change(["C", "A", "T"])
    assert rec0.action == "PLACE"
    w = rec0.details["word"]
    r, c, dr = rec0.details["row"], rec0.details["col"], rec0.details["direction"]
    grid.place_letter(r, c, w[0])
    if dr == "H":
        for i, ch in enumerate(w[1:], start=1):
            grid.place_letter(r, c + i, ch)
    else:
        for i, ch in enumerate(w[1:], start=1):
            grid.place_letter(r + i, c, ch)
    bunch.sync_observed_hand(["I", "T"])
    rec1 = sol.on_state_change(["I", "T"])
    assert rec1.action == "PLACE"
    assert rec1.details["word"] in {"AT", "IT"}
    assert rec1.details["word"] in rec1.reasoning


def test_dump_when_post_dump_scores_higher() -> None:
    # No QQ so one Q stays awkward; dumping X yields QQZZZ with strong ZZZ anchor vs QX now.
    words = {"QX", "ZZZ", "ZZ", "ZOO", "OO", "ZO", "AA", "AB", "BA"}
    bunch = BunchTracker()
    bunch.sync_observed_hand(["Q", "Q", "X"])
    bunch.peek_bunch = lambda: Counter({"Z": 100})  # type: ignore[method-assign]
    sol, d, grid, _ = _solver_with(words, bunch=bunch)
    hand = ["Q", "Q", "X"]
    rec = sol.on_state_change(hand)
    assert rec.action == "DUMP"
    assert rec.details["tile"] == "X"
    dn = min(3, bunch.bunch_size())
    sim, _ = _simulate_post_dump_hand(hand, "X", bunch, dn)
    _, post = _best_placement_for_hand(sim, grid, d, PlacementScorer(d))
    assert post > float("-inf")
    assert rec.score == pytest.approx(post)
    assert "DUMP" in rec.reasoning and "X" in rec.reasoning


def test_placement_when_post_dump_scores_lower() -> None:
    """Three Z/O tiles only — no stranded peel path; mock draw weakens post-dump vs ZOO."""
    words = {"ZOO", "ZO", "OO", "CAT", "AT", "CA", "QQ"}
    bunch = BunchTracker()
    bunch.sync_observed_hand(["Z", "O", "O"])
    bunch.peek_next = lambda n=3: ["B", "B", "B"]  # type: ignore[method-assign]
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    hand = ["Z", "O", "O"]
    rec = sol.on_state_change(hand)
    assert rec.action == "PLACE"
    assert rec.details["word"] == "ZOO"


def test_dump_score_matches_scorer_no_extra_penalty() -> None:
    """DUMP recommendation.score equals PlacementScorer best post-dump only."""
    words = {"QI", "XI", "IN", "IT", "HI", "ER", "RE", "EE", "TO", "OT", "GO", "OG", "ZOO", "OO", "ZO"}
    d = StubDictionary(words)
    grid = GridValidator(d)
    bunch = BunchTracker()
    bunch.sync_observed_hand(["Q", "I", "E", "E", "E", "E"])
    sc = PlacementScorer(d)
    sol = SolverLoop(bunch, grid, sc, d)
    hand = ["Q", "I", "E", "E", "E", "E"]
    dump_tile = min(hand, key=lambda x: (SCRABBLE_VALUES.get(x.upper(), 0), x.upper()))
    sim, _ = _simulate_post_dump_hand(hand, dump_tile, bunch, min(3, bunch.bunch_size()))
    _, expected_post = _best_placement_for_hand(sim, grid, d, sc)
    rec = sol.on_state_change(hand)
    if rec.action == "DUMP":
        assert rec.score == pytest.approx(expected_post)
        assert "not added to these scores" in rec.reasoning


def test_single_stranded_peel_beats_dump_and_place() -> None:
    """No placement with Q alone; peel adds Z -> QZ; post-dump simulation weaker than peel."""
    words = {"QZ", "ZQ", "ZZ"}
    d = StubDictionary(words)
    grid = GridValidator(d)
    bunch = PeelBunch()
    bunch.sync_observed_hand(["Q"])
    sol = SolverLoop(bunch, grid, PlacementScorer(d), d)
    hand = ["Q"]
    rec = sol.on_state_change(hand)
    assert rec.action == "PEEL"
    assert "stranded" in rec.reasoning.lower() or "Q" in rec.reasoning


def test_two_stranded_never_peel() -> None:
    words = {"EE", "QQ", "ZZ", "ZZZ", "ZOO", "OO", "ZO"}
    bunch = BunchTracker()
    bunch.sync_observed_hand(["E", "E", "Q", "Q"])
    bunch.peek_next = lambda n=3: ["Z", "Z", "Z"]  # type: ignore[method-assign]
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change(["E", "E", "Q", "Q"])
    assert rec.action != "PEEL"


def test_no_placements_bunch_nonempty_recommends_dump() -> None:
    words = {"AB", "BA"}
    bunch = BunchTracker()
    bunch.sync_observed_hand(["E", "E", "E"])
    bunch.peek_next = lambda n=3: ["A", "B", "C"]  # type: ignore[method-assign]
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change(["E", "E", "E"])
    assert rec.action == "DUMP"
    assert len(rec.reasoning) > 10


def test_no_placements_bunch_empty_game_over() -> None:
    words = {"AB", "BA"}
    bunch = MagicMock()
    bunch.sync_observed_hand = MagicMock()
    bunch.bunch_size = MagicMock(return_value=0)
    bunch.peek_next = MagicMock(return_value=[])
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change(["X", "X", "X"])
    assert rec.action == "GAME_OVER"
    assert "GAME" in rec.reasoning.upper()
    assert "bunch" in rec.reasoning.lower() or "empty" in rec.reasoning.lower()


def test_reasoning_contains_concrete_data() -> None:
    words = {"ZOO", "ZO", "OO"}
    bunch = BunchTracker()
    bunch.sync_observed_hand(["Z", "O", "O"])
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change(["Z", "O", "O"])
    assert len(rec.reasoning) > 15
    assert any(c.isdigit() for c in rec.reasoning)


@pytest.fixture
def twl_path(tmp_path: Path) -> str:
    p = tmp_path / "twl.txt"
    p.write_text("AA\nAB\nBA\n", encoding="utf-8")
    return str(p)


def test_solver_accepts_real_dictionary_subclass(twl_path: str) -> None:
    d = Dictionary(twl_path)
    g = GridValidator(d)
    b = BunchTracker()
    b.sync_observed_hand(["A", "A", "B"])
    sol = SolverLoop(b, g, PlacementScorer(d), d)
    rec = sol.on_state_change(["A", "A", "B"])
    assert rec.action == "PLACE"
    assert rec.details["word"] in {"AA", "AB", "BA"}


def test_bunch_peek_next_deterministic_order() -> None:
    bt = BunchTracker()
    bt.sync_observed_hand(["Z"])
    first = sorted(bt.peek_bunch().keys())[0]
    assert bt.peek_next(3)[0] == first


def test_bunch_sync_observed_hand_sets_bunch() -> None:
    bt = BunchTracker()
    bt.sync_observed_hand(["A", "A", "E"])
    assert bt.hand_size() == 3
    assert bt.bunch_size() == 141


def test_dump_bunch_size_two_draw_and_reasoning_align() -> None:
    """DUMP uses draw_n=min(3, bunch_size): two tiles in preview, score, and reasoning when bunch has 2."""
    words = {"AB", "BA"}
    hand = ["E", "E", "E"]
    bunch = _tracker_for_hand_and_bunch_size(hand, 2)
    sol, d, grid, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change(hand)
    assert rec.action == "DUMP"
    assert len(rec.details["draw_preview"]) == 2
    assert "draws 3" not in rec.reasoning.lower()
    assert "draws 2" in rec.reasoning.lower()
    dt = rec.details["tile"]
    draw_n = min(3, bunch.bunch_size())
    assert draw_n == 2
    sim, drawn = _simulate_post_dump_hand([h.upper() for h in hand], dt, bunch, draw_n)
    assert drawn == rec.details["draw_preview"]
    assert Counter(sim) == Counter(rec.details["post_dump_hand"])
    _, post = _best_placement_for_hand(sim, grid, d, PlacementScorer(d))
    expected_score = float(post) if post > float("-inf") else 0.0
    assert rec.score == pytest.approx(expected_score)


def test_dump_bunch_size_one_draw_and_reasoning_align() -> None:
    """DUMP draws exactly one tile when bunch size is 1; details and scorer use draw_n=1."""
    words = {"AB", "BA"}
    hand = ["E", "E", "E"]
    bunch = _tracker_for_hand_and_bunch_size(hand, 1)
    sol, d, grid, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change(hand)
    assert rec.action == "DUMP"
    assert len(rec.details["draw_preview"]) == 1
    draw_n = min(3, bunch.bunch_size())
    assert draw_n == 1
    sim, drawn = _simulate_post_dump_hand([h.upper() for h in hand], rec.details["tile"], bunch, draw_n)
    assert drawn == rec.details["draw_preview"]
    assert Counter(sim) == Counter(rec.details["post_dump_hand"])
    assert "draws 1" in rec.reasoning.lower()
    assert "draws 2" not in rec.reasoning.lower()
    _, post = _best_placement_for_hand(sim, grid, d, PlacementScorer(d))
    expected_score = float(post) if post > float("-inf") else 0.0
    assert rec.score == pytest.approx(expected_score)


def test_dump_when_dump_tile_is_only_bunch_tile_post_hand_is_consistent() -> None:
    """Dumped tile re-enters bunch before draw; draw preview and post-dump multiset match simulation."""
    words: set[str] = set()
    hand = ["Q", "E", "T"]
    bunch = _tracker_with_exact_remaining(hand, Counter({"Q": 1}))
    d = StubDictionary(words)
    grid = GridValidator(d)
    sol = SolverLoop(bunch, grid, PlacementScorer(d), d)
    # Force dumping Q (fix #12): live _pick_dump_tile would dump E first by Scrabble value.
    with patch("core.solver._pick_dump_tile", return_value="Q"):
        rec = sol.on_state_change(hand)
    assert rec.action == "DUMP"
    assert rec.details["draw_preview"] == ["Q"]
    assert Counter(rec.details["post_dump_hand"]) == Counter(["E", "T", "Q"])
    draw_n = min(3, bunch.bunch_size())
    sim, drawn = _simulate_post_dump_hand([h.upper() for h in hand], "Q", bunch, draw_n)
    assert drawn == ["Q"]
    _, post = _best_placement_for_hand(sim, grid, d, PlacementScorer(d))
    expected_score = float(post) if post > float("-inf") else 0.0
    assert rec.score == pytest.approx(expected_score)


def test_bridged_runs_valid_merges_place_invalid_merges_fallback() -> None:
    """SolverLoop never recommends a bridge placement whose merged row fails dictionary validation."""
    words_ok = {"CA", "AT", "CAT"}
    d_ok = StubDictionary(words_ok)
    grid_ok = GridValidator(d_ok)
    assert grid_ok.place_letter(0, 0, "C")
    assert grid_ok.place_letter(0, 2, "T")
    assert grid_ok.validate_board()
    sol_ok = SolverLoop(BunchTracker(), grid_ok, PlacementScorer(d_ok), d_ok)
    rec_ok = sol_ok.on_state_change(["A"])
    assert rec_ok.action == "PLACE"
    assert rec_ok.details["word"] == "CAT"
    w = rec_ok.details["word"]
    r, c, dr = rec_ok.details["row"], rec_ok.details["col"], rec_ok.details["direction"]
    g_apply = GridValidator(d_ok)
    assert g_apply.place_letter(r, c, w[0])
    if dr == "H":
        for i, ch in enumerate(w[1:], start=1):
            assert g_apply.place_letter(r, c + i, ch)
    else:
        for i, ch in enumerate(w[1:], start=1):
            assert g_apply.place_letter(r + i, c, ch)
    assert g_apply.validate_board()

    words_bad: set[str] = set()
    d_bad = StubDictionary(words_bad)
    grid_bad = GridValidator(d_bad)
    assert grid_bad.place_letter(0, 0, "C")
    assert grid_bad.place_letter(0, 2, "G")
    assert grid_bad.validate_board()
    sol_bad = SolverLoop(BunchTracker(), grid_bad, PlacementScorer(d_bad), d_bad)
    rec_bad = sol_bad.on_state_change(["A"])
    assert rec_bad.action != "PLACE"


def test_parallel_vertical_adjacency_rejects_invalid_incidental_runs() -> None:
    """Parallel-adjacent vertical extensions that create invalid perpendicular runs are not recommended."""
    words = {"AB", "BG", "HI", "IF", "CD"}
    d = StubDictionary(words)
    grid = GridValidator(d)
    _place_word_on_grid(grid, "AB", 0, 0, "H")
    sol = SolverLoop(BunchTracker(), grid, PlacementScorer(d), d)
    rec_bad = sol.on_state_change(["C", "D", "G"])
    assert rec_bad.action == "PLACE"
    assert rec_bad.details["word"] == "BG"
    g_apply = GridValidator(d)
    _place_word_on_grid(g_apply, "AB", 0, 0, "H")
    _apply_place_recommendation(g_apply, rec_bad)
    rec_good = sol.on_state_change(["G"])
    assert rec_good.action == "PLACE"
    assert rec_good.details["word"] == "BG"
    g_final = GridValidator(d)
    _place_word_on_grid(g_final, "AB", 0, 0, "H")
    _apply_place_recommendation(g_final, rec_good)


class PeelWeakBunch(BunchTracker):
    """Peel sees X (no helpful word); post-dump uses real bunch multiset."""

    def peek_next(self, n: int = 3) -> list[str]:
        if n == 1:
            return ["X"]
        return super().peek_next(n)


def test_peel_vs_dump_single_stranded_prefers_higher_score_never_peel_if_two_stranded() -> None:
    """With one stranded tile, PEEL wins only when peel_score beats post-dump; two stranded tiles never PEEL."""
    words_peel = {"QZ", "ZZ", "ZZZ", "ZQ"}
    bunch_peel = PeelBunch()
    bunch_peel.sync_observed_hand(["Q"])
    sol_peel, _, _, _ = _solver_with(words_peel, bunch=bunch_peel)
    rec_peel = sol_peel.on_state_change(["Q"])
    assert rec_peel.action == "PEEL"
    assert "stranded" in rec_peel.reasoning.lower()
    assert "peel" in rec_peel.reasoning.lower() or "PEEL" in rec_peel.reasoning

    words_dump = {"QZ", "ZZ", "ZZZ", "ZQ"}
    hand_q = ["Q"]
    bunch_dump = PeelWeakBunch()
    bunch_dump.sync_observed_hand(hand_q)
    sol_dump, _, _, _ = _solver_with(words_dump, bunch=bunch_dump)
    rec_dump = sol_dump.on_state_change(hand_q)
    assert rec_dump.action == "DUMP"

    words_2 = {"EE", "QQ", "ZZ", "ZZZ", "ZOO", "OO", "ZO"}
    bunch_2 = BunchTracker()
    bunch_2.sync_observed_hand(["E", "E", "Q", "Q"])
    bunch_2.peek_next = lambda n=3: ["Z", "Z", "Z"]  # type: ignore[method-assign]
    sol_2, _, _, _ = _solver_with(words_2, bunch=bunch_2)
    assert sol_2.on_state_change(["E", "E", "Q", "Q"]).action != "PEEL"


def test_peel_not_surfaced_when_any_placement_exists() -> None:
    """One stranded tile does not trigger PEEL if another tile still has a valid placement."""
    words = {"AT", "TA", "QZ"}
    d = StubDictionary(words)
    grid = GridValidator(d)
    _place_word_on_grid(grid, "AT", 0, 0, "H")
    bunch = _tracker_for_hand_and_bunch_size(["Q", "T", "A"], 0)
    sol = SolverLoop(bunch, grid, PlacementScorer(d), d)
    rec = sol.on_state_change(["Q", "T", "A"])
    assert rec.action == "PLACE"
    assert rec.details["word"] in {"AT", "TA"}
    g_apply = GridValidator(d)
    _place_word_on_grid(g_apply, "AT", 0, 0, "H")
    _apply_place_recommendation(g_apply, rec)


def test_restructure_recommended_when_cheaper_than_dump() -> None:
    """With no placements, a low-cost non-AP restructure beats DUMP when bunch is non-empty."""
    words = {"AB", "CD"}
    d = StubDictionary(words)
    grid = GridValidator(d)
    _place_word_on_grid(grid, "AB", 0, 0, "H")
    _place_word_on_grid(grid, "CD", 2, 0, "H")
    bunch = _tracker_for_hand_and_bunch_size(["E", "E", "E"], 20)
    sol = SolverLoop(bunch, grid, PlacementScorer(d), d)
    rec = sol.on_state_change(["E", "E", "E"])
    assert rec.action == "RESTRUCTURE"
    assert "word" in rec.details
    assert rec.details["word"] in {"AB", "CD"}
    assert "restructure cost" in rec.reasoning.lower() or "Restructure cost" in rec.reasoning
    assert "dump" in rec.reasoning.lower()
    g2 = GridValidator(d)
    remove = set(tuple(p) for p in rec.details["positions"])
    for (r, c), ch in grid.get_board_state().items():
        if (r, c) not in remove:
            assert g2.place_letter(r, c, ch)
    assert g2.validate_board()


def test_restructure_skipped_all_articulation_points_falls_through_to_dump() -> None:
    """When the only restructure candidates are expensive articulation points, solver chooses DUMP."""
    words = {"AB", "BA"}
    d = StubDictionary(words)
    grid = GridValidator(d)
    _place_word_on_grid(grid, "AB", 0, 0, "H")
    hand = ["E", "E", "E"]
    bunch = _tracker_for_hand_and_bunch_size(hand, 15)
    sol = SolverLoop(bunch, grid, PlacementScorer(d), d)
    rec = sol.on_state_change(hand)
    assert rec.action == "DUMP"
    from core.scorer import RestructureScorer

    rs = RestructureScorer(d)
    cands = rs.score_restructure(dict(grid.get_board_state()), [h.upper() for h in hand], d)
    assert cands, "board should yield restructure candidates so DUMP implies rejection, not skip"


def test_real_dictionary_and_bunch_integration_no_peek_patches(tmp_path: Path) -> None:
    """Real Dictionary file and BunchTracker: PLACE a known word, then DUMP/PEEL with honest peek order."""
    lines = [
        "HELLO",
        "HELL",
        "ELLO",
        "EL",
        "HE",
        "LO",
        "EH",
        "HO",
        "OLE",
        "BINGO",
        "RING",
        "GRIN",
        "BRING",
        "TILE",
        "LITE",
        "RITE",
        "TIER",
        "NOTE",
        "TONE",
        "QUIZ",
        "ZIP",
        "ZAP",
        "GAP",
        "MAP",
        "NAP",
        "RAP",
        "SAP",
        "TAP",
        "YAP",
        "WAR",
        "RAW",
        "WAS",
        "SAW",
        "SEW",
        "NEW",
        "NET",
        "TEN",
        "ANT",
        "ART",
        "RAT",
        "TAR",
        "CAT",
        "ACT",
        "BAT",
        "TAB",
        "AT",
        "IT",
        "IN",
        "ON",
        "NO",
        "GO",
        "SO",
        "TO",
        "UP",
        "US",
    ]
    p = tmp_path / "solver_integration_words.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    d = Dictionary(str(p))
    bt = BunchTracker()
    hello = ["H", "E", "L", "L", "O"]
    bt.draw_to_hand(hello)
    assert bt.is_valid_state()
    g = GridValidator(d)
    sol = SolverLoop(bt, g, PlacementScorer(d), d)
    rec1 = sol.on_state_change(hello)
    assert rec1.action == "PLACE"
    w1 = rec1.details["word"]
    assert d.is_word(w1)
    assert bt.is_valid_state()

    bt2 = BunchTracker()
    draw2 = ["Q", "Q", "J", "J", "X"]
    bt2.draw_to_hand(draw2)
    g2 = GridValidator(d)
    sol2 = SolverLoop(bt2, g2, PlacementScorer(d), d)
    rec2 = sol2.on_state_change(draw2)
    assert rec2.action in {"DUMP", "PEEL"}
    if rec2.action == "DUMP":
        preview = rec2.details.get("draw_preview", [])
        bc = bt2.peek_bunch().copy()
        bc[rec2.details["tile"]] = bc.get(rec2.details["tile"], 0) + 1
        expected = _peek_first_n_from_counter(bc, min(3, bt2.bunch_size()))
        assert preview == expected
    assert bt2.is_valid_state()


def test_empty_hand_with_bunch_game_over_safe() -> None:
    """Empty hand and empty grid with tiles still in bunch returns GAME_OVER without crashing."""
    words = {"AB", "BA"}
    bunch = _tracker_for_hand_and_bunch_size([], 50)
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change([])
    assert rec.action == "GAME_OVER"
    r = rec.reasoning.lower()
    assert "0" in rec.reasoning
    assert "bunch" in r
    assert str(bunch.bunch_size()) in rec.reasoning or "50" in rec.reasoning


def test_wait_start_empty_hand_before_round() -> None:
    words = {"AB", "BA"}
    bunch = _tracker_for_hand_and_bunch_size([], 50)
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change([], game_active=False)
    assert rec.action == "WAIT_START"
    assert rec.details["hand_count"] == 0
    assert rec.details["phase"] == "deal_tiles"


def test_wait_start_partial_rack() -> None:
    words = {"AB", "BA"}
    bunch = _tracker_for_hand_and_bunch_size(["A", "B", "C"], 50)
    sol, _, _, _ = _solver_with(words, bunch=bunch)
    rec = sol.on_state_change(["a", "b", "c"], game_active=False)
    assert rec.action == "WAIT_START"
    assert rec.details["hand_count"] == 3
    assert rec.details["phase"] == "dealing"


def test_wait_start_full_rack_press_g_prompt() -> None:
    words = {"AB", "BA", "CAT", "AT", "ACT"}
    bt = BunchTracker()
    drawn: list[str] = []
    for _ in range(21):
        nxt = bt.peek_next(1)
        assert nxt
        drawn.append(nxt[0])
        bt.draw_to_hand([nxt[0]])
    bt.sync_observed_hand(drawn)
    sol, _, _, _ = _solver_with(words, bunch=bt)
    rec = sol.on_state_change(drawn, game_active=False)
    assert rec.action == "WAIT_START"
    assert rec.details["hand_count"] == 21
    assert rec.details["phase"] == "press_g"


def test_wait_start_skipped_when_grid_nonempty() -> None:
    """If letters are already on the validator grid, do not block the solver behind WAIT_START."""
    words = {"CAT", "AT", "TA", "ACT"}
    bunch = _tracker_for_hand_and_bunch_size([], 50)
    sol, _, g, _ = _solver_with(words, bunch=bunch)
    assert g.place_letter(0, 0, "C")
    assert g.place_letter(0, 1, "A")
    assert g.place_letter(0, 2, "T")
    rec = sol.on_state_change([], game_active=False)
    assert rec.action == "GAME_OVER"
