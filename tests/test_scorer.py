"""Unit tests for core.scorer.PlacementScorer and RestructureScorer."""

from __future__ import annotations

from collections import Counter

import pytest

from core.scorer import (
    Grid,
    PlacementScorer,
    RestructureScorer,
    SCRABBLE_VALUES,
    _isolation_nonparticipating_tile_count,
    _scrabble_sum,
)


class StubDictionary:
    """Set-based stub with is_word, is_prefix, and get_words_from_letters."""

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


@pytest.fixture
def placement_dict_flex() -> StubDictionary:
    """Words so E,R,T is flexible; G,T,T much less so (only GT)."""
    return StubDictionary(
        {
            "ER",
            "ET",
            "RT",
            "ERT",
            "GT",
        }
    )


@pytest.fixture
def placement_dict_isolation() -> StubDictionary:
    """No Q-words; ER etc. for contrast."""
    return StubDictionary({"ER", "RE", "EE"})


def test_anchor_higher_scrabble_wins() -> None:
    d = StubDictionary({"AA", "BB", "AB", "BA"})
    s = PlacementScorer(d)
    # ZOO all from hand: Z=10+O+O vs AAA: 1+1+1
    z = s.score_anchor("ZOO", [])
    a = s.score_anchor("AAA", [])
    assert z > a


def test_anchor_tiebreaker_longer_wins() -> None:
    d = StubDictionary(set())
    s = PlacementScorer(d)
    # Same Scrabble sum (B+A = 4, E+E+E+E = 4); longer word wins only via 0.1 * len
    short_w, long_w = "BA", "EEEE"
    assert _scrabble_sum(short_w) == _scrabble_sum(long_w)
    short = s.score_anchor(short_w, [])
    long = s.score_anchor(long_w, [])
    assert long > short


def test_flexibility_ert_beats_gtt(placement_dict_flex: StubDictionary) -> None:
    s = PlacementScorer(placement_dict_flex)
    grid: Grid = {}
    # Same scrabble from hand (E+N = 1+1); flexibility differs for E,R,T vs G,T,T
    w = "EN"
    score_ert = s.score_placement(w, (0, 0), "H", grid, ["E", "R", "T"], placement_dict_flex)
    score_gtt = s.score_placement(w, (0, 0), "H", grid, ["G", "T", "T"], placement_dict_flex)
    assert score_ert > score_gtt


def test_isolation_penalty_q_without_words(placement_dict_isolation: StubDictionary) -> None:
    s = PlacementScorer(placement_dict_isolation)
    grid: Grid = {}
    assert _isolation_nonparticipating_tile_count(placement_dict_isolation, ["Q"]) == 1
    assert _isolation_nonparticipating_tile_count(placement_dict_isolation, ["E", "E"]) == 0
    # Same scrabble from hand; Q is isolated (no Q in any word from stub), E tiles both appear in EE
    hand_q = ["Q"]
    hand_ee = ["E", "E"]
    score_q = s.score_placement("EE", (0, 0), "H", grid, hand_q, placement_dict_isolation)
    score_ee = s.score_placement("EE", (0, 0), "H", grid, hand_ee, placement_dict_isolation)
    assert score_ee > score_q


def test_parallel_vertical_scores_lower_than_l_shape() -> None:
    d = StubDictionary({"AB", "CD", "EF", "GH", "DD", "GG", "DP"})
    s = PlacementScorer(d)
    # Existing vertical stack col 0; parallel-adjacent second vertical in col 1 (no shape bonus)
    base: Grid = {(0, 0): "E", (1, 0): "F"}
    parallel = s.score_placement("GH", (0, 1), "V", base, [], d)
    # Same word vertical in col 3 — col 2 empty, so not parallel-adjacent to col 0 (+1.5 shape bonus)
    spaced = s.score_placement("GH", (0, 3), "V", base, [], d)
    assert spaced - parallel == pytest.approx(1.5)
    # Horizontal extension off vertical anchor DD: bonus applies; parallel-adjacent vertical GG does not
    base_d: Grid = {(0, 0): "D", (1, 0): "D"}
    parallel_gg = s.score_placement("GG", (0, 1), "V", base_d, [], d)
    horizontal_dp = s.score_placement("DP", (1, 0), "H", base_d, [], d)
    assert horizontal_dp > parallel_gg


def test_crossing_tile_zero_scrabble_component() -> None:
    d = StubDictionary({"AB", "BC", "ABC", "AC"})
    s = PlacementScorer(d)
    # B already at (0,1); extend horizontal "AB" from (0,0) — A from hand, B on board
    grid: Grid = {(0, 1): "B"}
    score = s.score_placement("AB", (0, 0), "H", grid, [], d)
    assert score == pytest.approx(float(SCRABBLE_VALUES["A"]) + 1.5)  # A only + shape bonus (H placement)


def test_restructure_single_word_is_articulation_high_cost() -> None:
    d = StubDictionary({"AB", "CD"})
    r = RestructureScorer(d)
    grid: Grid = {(0, 0): "A", (0, 1): "B"}
    out = r.score_restructure(grid, [], d)
    assert len(out) == 1
    c = out[0]
    assert c.is_articulation_point is True
    assert c.cost == pytest.approx(20.0)  # 10 * 2 tiles
    assert c.reasoning


def test_restructure_linear_chain_middle_is_articulation() -> None:
    d = StubDictionary({"AB", "BC", "CD", "ABC", "BCD"})
    r = RestructureScorer(d)
    # AB row0 col0-1, BC col1 row0-1, CD row1 col1-2
    grid: Grid = {
        (0, 0): "A",
        (0, 1): "B",
        (1, 1): "C",
        (1, 2): "D",
    }
    out = r.score_restructure(grid, [], d)
    by_word = {c.word: c for c in out}
    assert by_word["BC"].is_articulation_point is True
    assert by_word["BC"].stranded_tile_count >= 1
    assert by_word["AB"].is_articulation_point is False
    assert by_word["CD"].is_articulation_point is False


def test_restructure_degree_one_not_articulation_on_cycle_tail() -> None:
    # 2x2 core + vertical BDE (col 1) + horizontal EF; EF shares only E with BDE → degree 1, not articulation
    d3 = StubDictionary({"AB", "CD", "AC", "BDE", "EF"})
    r3 = RestructureScorer(d3)
    grid3: Grid = {
        (0, 0): "A",
        (0, 1): "B",
        (1, 0): "C",
        (1, 1): "D",
        (2, 1): "E",
        (2, 2): "F",
    }
    out3 = r3.score_restructure(grid3, [], d3)
    ef = next((c for c in out3 if c.word == "EF"), None)
    assert ef is not None
    assert ef.connection_count == 1
    assert ef.is_articulation_point is False


def test_restructure_cost_ascending_and_reasoning() -> None:
    d = StubDictionary({"AB", "BC", "CD", "XY", "YZ"})
    r = RestructureScorer(d)
    grid: Grid = {
        (0, 0): "A",
        (0, 1): "B",
        (1, 1): "C",
        (1, 2): "D",
        (10, 0): "X",
        (10, 1): "Y",
        (11, 1): "Z",
    }
    out = r.score_restructure(grid, ["M"], d)
    costs = [c.cost for c in out]
    assert costs == sorted(costs)
    assert len(out) <= 5
    assert all(c.reasoning.strip() for c in out)


def test_placement_empty_hand_after_finite() -> None:
    d = StubDictionary({"ZZ", "ZY"})
    s = PlacementScorer(d)
    score = s.score_placement("Z", (0, 0), "H", {}, [], d)
    assert isinstance(score, float)
    assert score == score  # finite (not NaN)


def test_restructure_no_two_plus_letter_runs_returns_empty() -> None:
    d = StubDictionary({"AB", "CD"})
    r = RestructureScorer(d)
    grid: Grid = {(0, 0): "A", (10, 10): "B", (5, 5): "C"}
    assert r.score_restructure(grid, [], d) == []


def test_placement_zero_tiles_from_hand_scores_flex_and_shape_only() -> None:
    d = StubDictionary(set())
    s = PlacementScorer(d)
    grid: Grid = {(0, 0): "A", (0, 1): "B"}
    score = s.score_placement("AB", (0, 0), "H", grid, [], d)
    assert score == pytest.approx(1.5)


def test_restructure_all_candidates_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """No realistic board marks every run as AP with stranded>3; force exclusion path."""
    d = StubDictionary({"AB", "CD", "AC", "BD"})
    r = RestructureScorer(d)
    grid: Grid = {
        (0, 0): "A",
        (0, 1): "B",
        (1, 0): "C",
        (1, 1): "D",
    }
    monkeypatch.setattr("core.scorer._tarjan_articulation_points", lambda n, adj: [True] * n)
    monkeypatch.setattr("core.scorer._stranded_tile_count", lambda *args, **kwargs: 5)
    assert r.score_restructure(grid, [], d) == []
