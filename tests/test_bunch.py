"""Unit tests for core.bunch.BunchTracker."""

from __future__ import annotations

import pytest
from collections import Counter

from core.bunch import BunchTracker

TOTAL_TILES = 144


def test_initial_state_bunch_144_hand_and_placed_empty() -> None:
    bt = BunchTracker()
    assert bt.bunch_size() == TOTAL_TILES
    assert bt.hand_size() == 0
    assert bt.peek_hand() == Counter()
    assert bt.is_valid_state()
    # Valid partition with hand=0 and bunch=144 forces placed and dumped empty.


def test_draw_to_hand_moves_tiles_bunch_decrements() -> None:
    bt = BunchTracker()
    letters = ["A", "A", "E", "I", "O"]
    before_bunch = bt.bunch_size()
    bt.draw_to_hand(letters)
    assert bt.hand_size() == len(letters)
    assert bt.bunch_size() == before_bunch - len(letters)
    assert bt.peek_hand() == Counter(letters)
    assert bt.is_valid_state()


def test_draw_to_hand_raises_when_tile_not_in_bunch() -> None:
    bt = BunchTracker()
    for _ in range(13):
        bt.draw_to_hand(["A"])
    with pytest.raises(ValueError, match="not enough 'A' in bunch"):
        bt.draw_to_hand(["A"])


def test_place_tiles_moves_hand_to_placed() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["B", "A", "N", "A", "N", "A"])
    bt.place_tiles(["B", "A", "N"])
    assert bt.peek_hand() == Counter({"A": 2, "N": 1})
    assert bt.hand_size() == 3
    assert bt.bunch_size() == TOTAL_TILES - 6
    assert bt.is_valid_state()


def test_place_tiles_raises_when_tile_not_in_hand() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["A", "B"])
    with pytest.raises(ValueError, match="not enough 'Z' in hand"):
        bt.place_tiles(["Z"])


def test_return_to_hand_moves_placed_to_hand() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["C", "A", "T"])
    bt.place_tiles(["C", "A"])
    bt.return_to_hand(["A"])
    assert bt.peek_hand() == Counter({"A": 1, "T": 1})
    assert bt.is_valid_state()


def test_dump_moves_one_tile_hand_to_dumped() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["Q", "U", "I"])
    bt.dump("Q")
    assert bt.peek_hand() == Counter({"U": 1, "I": 1})
    assert bt.is_valid_state()


def test_dump_raises_when_tile_not_in_hand() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["A"])
    with pytest.raises(ValueError, match="not enough 'Z' in hand"):
        bt.dump("Z")


def test_draw_from_bunch_draws_into_hand() -> None:
    bt = BunchTracker()
    # 21 tiles respecting per-letter caps (e.g. only 2 X in the set).
    start = ["X"] * 2 + ["Y"] * 3 + ["Z"] * 2 + ["A"] * 13 + ["E"] * 1
    assert len(start) == 21
    bt.draw_to_hand(start)
    # Y and Z are exhausted by the opening draw; draw from letters still in bunch.
    bt.draw_from_bunch(["E", "E", "I"])
    assert bt.hand_size() == 24
    assert bt.peek_hand() == Counter(
        {"X": 2, "Y": 3, "Z": 2, "A": 13, "E": 3, "I": 1}
    )
    assert bt.bunch_size() == TOTAL_TILES - 24
    assert bt.is_valid_state()


def test_draw_from_bunch_raises_when_bunch_lacks_tile() -> None:
    bt = BunchTracker()
    for _ in range(18):
        bt.draw_to_hand(["E"])
    with pytest.raises(ValueError, match="not enough 'E' in bunch"):
        bt.draw_from_bunch(["E"])


def test_peek_bunch_and_peek_hand_return_copies() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["M", "O", "P"])
    pb = bt.peek_bunch()
    ph = bt.peek_hand()
    pb["A"] = 99
    ph["Z"] = 99
    assert bt.peek_bunch()["A"] == 13
    assert bt.peek_hand().get("Z", 0) == 0
    assert bt.peek_hand() == Counter({"M": 1, "O": 1, "P": 1})


def test_bunch_size_and_hand_size_through_stages() -> None:
    bt = BunchTracker()
    assert bt.bunch_size() == 144 and bt.hand_size() == 0
    bt.draw_to_hand(["E"] * 10)
    assert bt.bunch_size() == 134 and bt.hand_size() == 10
    bt.place_tiles(["E"] * 4)
    assert bt.bunch_size() == 134 and bt.hand_size() == 6
    bt.dump("E")
    assert bt.bunch_size() == 134 and bt.hand_size() == 5


def test_is_valid_state_true_after_valid_operations() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["S", "O", "L", "O"])
    bt.place_tiles(["S", "O"])
    bt.return_to_hand(["O"])
    bt.dump("L")
    bt.draw_from_bunch(["A", "B"])
    assert bt.is_valid_state()


def test_is_valid_state_false_when_internal_state_corrupted() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["A"])
    bt._hand["A"] += 50  # noqa: SLF001 — impossible surplus in hand
    assert bt.is_valid_state() is False


def test_full_dump_sequence() -> None:
    bt = BunchTracker()
    # 21 tiles within per-letter limits (e.g. N has 8, X has 2).
    start = ["D", "U", "M", "P"] + ["X"] * 2 + ["A"] * 13 + ["E"] * 2
    assert len(start) == 21
    bt.draw_to_hand(start)
    assert bt.hand_size() == 21
    assert bt.bunch_size() == TOTAL_TILES - 21

    bt.place_tiles(["D", "U", "M"])
    assert bt.hand_size() == 18

    bt.dump("P")
    assert bt.hand_size() == 17

    # Bunch has no A left after the opening draw; draw three Bs instead.
    bt.draw_from_bunch(["B", "B", "B"])
    assert bt.hand_size() == 20
    # Dump moves hand → dumped, not bunch; bunch only loses the 21 + 3 draws.
    assert bt.bunch_size() == TOTAL_TILES - 21 - 3
    assert bt.peek_hand()["P"] == 0
    assert bt.peek_hand()["B"] == 3
    assert bt.is_valid_state()


def test_full_peel_sequence() -> None:
    bt = BunchTracker()
    # Only 2 K in the set — P,E,E,L + 13 A + 4 E = 21 legal tiles.
    letters21 = ["P", "E", "E", "L"] + ["A"] * 13 + ["E"] * 4
    assert len(letters21) == 21
    bt.draw_to_hand(letters21)
    assert bt.bunch_size() == TOTAL_TILES - 21

    bt.place_tiles(letters21)
    assert bt.hand_size() == 0
    bunch_before_peel = bt.bunch_size()

    bt.draw_to_hand(["R"])
    assert bt.hand_size() == 1
    assert bt.bunch_size() == bunch_before_peel - 1
    assert bt.is_valid_state()


def test_draw_all_tiles_empties_bunch_completely() -> None:
    bt = BunchTracker()
    all_tiles = (
        ["A"] * 13
        + ["B"] * 3
        + ["C"] * 3
        + ["D"] * 6
        + ["E"] * 18
        + ["F"] * 3
        + ["G"] * 4
        + ["H"] * 3
        + ["I"] * 12
        + ["J"] * 2
        + ["K"] * 2
        + ["L"] * 5
        + ["M"] * 3
        + ["N"] * 8
        + ["O"] * 11
        + ["P"] * 3
        + ["Q"] * 2
        + ["R"] * 9
        + ["S"] * 6
        + ["T"] * 9
        + ["U"] * 6
        + ["V"] * 3
        + ["W"] * 3
        + ["X"] * 2
        + ["Y"] * 3
        + ["Z"] * 2
    )
    assert len(all_tiles) == TOTAL_TILES
    bt.draw_to_hand(all_tiles)
    assert bt.bunch_size() == 0
    assert bt.is_valid_state()
    assert bt.hand_size() == TOTAL_TILES


def test_lowercase_input_normalized_like_uppercase() -> None:
    bt = BunchTracker()
    before_bunch = bt.bunch_size()
    bt.draw_to_hand(["a", "e", "t"])
    assert bt.peek_hand() == Counter({"A": 1, "E": 1, "T": 1})
    assert bt.bunch_size() == before_bunch - 3
    assert bt.is_valid_state()


def test_return_to_hand_raises_when_tile_not_in_placed() -> None:
    bt = BunchTracker()
    bt.draw_to_hand(["K"])
    with pytest.raises(ValueError, match="not enough 'K' in placed"):
        bt.return_to_hand(["K"])


def test_full_dump_sequence_asserts_all_four_buckets_explicitly() -> None:
    bt = BunchTracker()
    # 21 tiles: 10 E, 8 N, 3 S (within per-letter caps).
    start = ["E"] * 10 + ["N"] * 8 + ["S"] * 3
    assert len(start) == 21
    bt.draw_to_hand(start)

    bt.place_tiles(["E"] * 10)
    assert bt.hand_size() == 11

    bt.dump("N")
    assert bt.hand_size() == 10

    bunch_before_draw_three = bt.bunch_size()
    peek_before = bt.peek_bunch()
    assert peek_before["A"] == 13

    bt.draw_from_bunch(["A", "A", "A"])

    assert bt.hand_size() == 13
    assert sum(bt._placed.values()) == 10  # noqa: SLF001 — no public peek_placed
    assert bt._placed == Counter({"E": 10})  # noqa: SLF001
    assert bt._dumped == Counter({"N": 1})  # noqa: SLF001 — no public peek_dumped
    assert bt.bunch_size() == bunch_before_draw_three - 3
    assert bt.peek_bunch()["A"] == peek_before["A"] - 3
    assert bt.peek_hand() == Counter({"N": 7, "S": 3, "A": 3})
    assert bt.is_valid_state()


def test_full_peel_sequence_uses_draw_from_bunch_by_name() -> None:
    bt = BunchTracker()
    # 12 A + 3 B + 3 C + 3 D = 21; leaves exactly 1 A in the bunch for the peel draw.
    letters21 = ["A"] * 12 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3
    assert len(letters21) == 21
    bt.draw_to_hand(letters21)
    bt.place_tiles(letters21)
    assert bt.hand_size() == 0

    bt.draw_from_bunch(["A"])

    assert bt.hand_size() == 1
    assert bt.peek_hand()["A"] == 1
    assert bt.bunch_size() == TOTAL_TILES - 21 - 1
    assert bt.is_valid_state()
