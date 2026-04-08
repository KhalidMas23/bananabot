"""Unit tests for core.grid.GridValidator."""

from __future__ import annotations

import pytest

from core.grid import GridValidator


class MockDictionary:
    """Minimal dict with is_word / is_prefix for grid tests (no file I/O)."""

    def __init__(self, words: set[str]) -> None:
        self._words = {w.upper() for w in words}

    def is_word(self, word: str) -> bool:
        return word.upper() in self._words

    def is_prefix(self, prefix: str) -> bool:
        p = prefix.upper()
        if not p:
            return bool(self._words)
        return any(w.startswith(p) for w in self._words)


@pytest.fixture
def simple_dict() -> MockDictionary:
    # Every contiguous run of 2+ letters must be is_word at placement time, so incremental
    # row builds need each growing prefix (length >= 2) in the mock.
    return MockDictionary(
        {
            "CA",
            "AT",
            "CAT",
            "CATS",
            "TR",
            "RI",
            "IC",
            "CK",
            "TRI",
            "TRIC",
            "TRICK",
            "TRICKS",
            "TRICKST",
            "TRICKSTE",
            "TRICKSTER",
            "AP",
            "APP",
            "APPL",
            "APPLE",
            "HI",
            "IF",
            "AB",
            "CD",
            "AT",
            "HE",
            "HAT",
            "IT",
            "XI",
            "US",
            "ZA",
            "QI",
        }
    )


def test_place_letter_empty_grid_succeeds(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "A") is True
    assert g.get_letter(0, 0) == "A"


def test_place_letter_occupied_fails_no_overwrite(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "X") is True
    assert g.place_letter(0, 0, "Y") is False
    assert g.get_letter(0, 0) == "X"


def test_remove_letter_returns_letter_and_clears(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    g.place_letter(1, 2, "Z")
    assert g.remove_letter(1, 2) == "Z"
    assert g.get_letter(1, 2) is None


def test_remove_letter_empty_returns_none(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.remove_letter(0, 0) is None


def test_single_isolated_letter_valid(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(5, -3, "Q") is True
    assert g.validate_board() is True


def test_single_letter_horizontal_neighbor_invalid_fails(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "C") is True
    assert g.place_letter(0, 1, "Z") is False
    assert g.get_letter(0, 1) is None
    assert g.get_letter(0, 0) == "C"


def test_single_letter_vertical_neighbor_invalid_fails(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "C") is True
    assert g.place_letter(1, 0, "Z") is False
    assert g.get_letter(1, 0) is None


def test_extend_valid_word_cat_to_cats(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "C") is True
    assert g.place_letter(0, 1, "A") is True
    assert g.place_letter(0, 2, "T") is True
    assert g.place_letter(0, 3, "S") is True
    assert g.get_board_state() == {(0, 0): "C", (0, 1): "A", (0, 2): "T", (0, 3): "S"}


def test_extend_to_invalid_sequence_rolls_back(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "C") is True
    assert g.place_letter(0, 1, "A") is True
    assert g.place_letter(0, 2, "T") is True
    assert g.place_letter(0, 3, "Z") is False
    assert g.get_letter(0, 3) is None
    assert g.validate_board() is True


def test_trick_to_trickster(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    word = "TRICK"
    for i, ch in enumerate(word):
        assert g.place_letter(0, i, ch) is True
    for j, ch in enumerate("STER", start=len(word)):  # S T E R → TRICKSTER
        assert g.place_letter(0, j, ch) is True
    assert g.validate_board() is True


def test_applextra_fails_board_stays_apple(simple_dict: MockDictionary) -> None:
    """After APPLE, extending along the row toward APPLEXTRA fails; board stays APPLE."""
    g = GridValidator(simple_dict)
    for i, ch in enumerate("APPLE"):
        assert g.place_letter(0, i, ch) is True
    # Shared E at col 4: next segment is X T R A → full run APPLEXTRA (invalid, like APPLEXTRA).
    assert g.place_letter(0, 5, "X") is False
    assert g.get_board_state() == {(0, i): "APPLE"[i] for i in range(5)}


def test_incidental_invalid_two_letter_vertical_fails() -> None:
    # Row AT at (0,0),(0,1); column 0 reads top-to-bottom A then Z → "AZ".
    d = MockDictionary({"AT"})
    g = GridValidator(d)
    assert g.place_letter(0, 0, "A") is True
    assert g.place_letter(0, 1, "T") is True
    assert g.place_letter(1, 0, "Z") is False
    assert g.get_letter(1, 0) is None


def test_incidental_valid_two_letter_vertical_succeeds() -> None:
    d = MockDictionary({"AT", "AZ"})
    g = GridValidator(d)
    assert g.place_letter(0, 0, "A") is True
    assert g.place_letter(0, 1, "T") is True
    assert g.place_letter(1, 0, "Z") is True
    assert g.validate_board() is True


def test_parallel_adjacency_invalid_perpendicular_fails() -> None:
    d = MockDictionary({"AB", "CD"})
    g = GridValidator(d)
    assert g.place_letter(0, 0, "A") is True
    assert g.place_letter(0, 1, "B") is True
    assert g.place_letter(1, 0, "C") is False
    assert g.get_letter(1, 0) is None


def test_parallel_adjacency_valid_perpendicular_succeeds() -> None:
    d = MockDictionary({"HI", "IF"})
    g = GridValidator(d)
    assert g.place_letter(0, 0, "H") is True
    assert g.place_letter(0, 1, "I") is True
    assert g.place_letter(1, 0, "I") is True
    assert g.place_letter(1, 1, "F") is True
    assert g.validate_board() is True


def test_bridging_merged_run_must_be_word(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "C") is True
    assert g.place_letter(0, 2, "T") is True
    assert g.place_letter(0, 1, "A") is True
    assert g.get_board_state() == {(0, 0): "C", (0, 1): "A", (0, 2): "T"}


def test_bridging_invalid_merge_rolls_back(simple_dict: MockDictionary) -> None:
    d = MockDictionary({"CAT", "DOG"})
    g = GridValidator(d)
    assert g.place_letter(0, 0, "C") is True
    assert g.place_letter(0, 2, "G") is True
    assert g.place_letter(0, 1, "A") is False
    assert g.get_letter(0, 1) is None
    assert g.get_board_state() == {(0, 0): "C", (0, 2): "G"}


def test_get_board_state_is_copy(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    g.place_letter(0, 0, "X")
    state = g.get_board_state()
    state[(9, 9)] = "Y"
    assert g.get_letter(9, 9) is None


def test_is_empty_fresh_and_after_place(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.is_empty() is True
    g.place_letter(0, 0, "A")
    assert g.is_empty() is False


def test_get_bounds(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    g.place_letter(-2, 5, "A")
    g.place_letter(3, -1, "B")
    assert g.get_bounds() == (-2, 3, -1, 5)


def test_get_bounds_empty_raises(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    with pytest.raises(ValueError, match="empty board"):
        g.get_bounds()


def test_validate_board_matches_full_scan_after_placements(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    for i, ch in enumerate("HI"):
        g.place_letter(0, i, ch)
    for i, ch in enumerate("IF"):
        g.place_letter(1, i, ch)
    assert g.validate_board() is True


def test_place_letter_lowercase_normalized(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "a") is True
    assert g.get_letter(0, 0) == "A"


def test_place_letter_invalid_char_fails(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    assert g.place_letter(0, 0, "1") is False
    assert g.place_letter(0, 0, "ab") is False


def test_remove_letter_then_board_still_valid(simple_dict: MockDictionary) -> None:
    g = GridValidator(simple_dict)
    for i, ch in enumerate("CAT"):
        g.place_letter(0, i, ch)
    g.remove_letter(0, 2)
    assert g.validate_board() is True
    assert g.get_letter(0, 2) is None


def test_dictionary_integration_twl_if_present() -> None:
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "data" / "wordlists" / "TWL06.txt"
    if not p.is_file():
        pytest.skip("TWL06 wordlist not present")
    from data.dictionary import Dictionary

    d = Dictionary(str(p))
    g = GridValidator(d)
    assert g.place_letter(0, 0, "Q") is True
    assert g.place_letter(0, 1, "I") is True
    assert g.validate_board() is True
