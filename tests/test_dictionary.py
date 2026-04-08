"""Unit tests for data.dictionary.Dictionary."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.dictionary import DEFAULT_WORDLIST_PATH, Dictionary

FIXTURE_WORDS = [
    "AA",
    "AI",
    "AH",
    "CAT",
    "CATS",
    "CAR",
    "CART",
    "TRICK",
    "TRICKSTER",
    "QI",
]


@pytest.fixture
def small_wordlist_path(tmp_path: Path) -> str:
    p = tmp_path / "fixture_words.txt"
    p.write_text("\n".join(FIXTURE_WORDS) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def dict_small(small_wordlist_path: str) -> Dictionary:
    return Dictionary(small_wordlist_path)


def test_is_word_true_for_known_words_including_two_letter(dict_small: Dictionary) -> None:
    assert dict_small.is_word("AA")
    assert dict_small.is_word("QI")
    assert dict_small.is_word("AI")
    assert dict_small.is_word("AH")
    assert dict_small.is_word("TRICK")
    assert dict_small.is_word("TRICKSTER")


def test_is_word_false_for_non_words_and_partial(dict_small: Dictionary) -> None:
    assert not dict_small.is_word("TRIC")
    assert not dict_small.is_word("XYZ")
    assert not dict_small.is_word("APPLE")
    assert not dict_small.is_word("TRICKS")


def test_is_prefix_true_when_completions_exist(dict_small: Dictionary) -> None:
    assert dict_small.is_prefix("TRIC")
    assert dict_small.is_prefix("TRICK")
    assert dict_small.is_prefix("TRICKS")
    assert dict_small.is_prefix("C")
    assert dict_small.is_prefix("CA")


def test_is_prefix_false_when_no_completions(dict_small: Dictionary) -> None:
    assert not dict_small.is_prefix("ZZ")
    assert not dict_small.is_prefix("ZZZ")
    assert not dict_small.is_prefix("XYZ")


def test_is_prefix_true_for_terminal_leaf(dict_small: Dictionary) -> None:
    assert dict_small.is_prefix("QI")


def test_get_words_from_letters_known_hand(dict_small: Dictionary) -> None:
    got = dict_small.get_words_from_letters(["C", "A", "R", "T"])
    assert got == ["CAR", "CART", "CAT"]


def test_get_words_from_letters_respects_multiset(dict_small: Dictionary) -> None:
    assert dict_small.get_words_from_letters(["A"]) == []
    assert dict_small.get_words_from_letters(["A", "A"]) == ["AA"]
    assert dict_small.get_words_from_letters(["C", "A", "T"]) == ["CAT"]
    assert dict_small.get_words_from_letters(["C", "A", "T", "S"]) == ["CAT", "CATS"]


def test_get_words_from_letters_includes_two_letter_when_formable(dict_small: Dictionary) -> None:
    hand = ["A", "A", "I", "H", "Q", "I"]
    got = set(dict_small.get_words_from_letters(hand))
    assert {"AA", "AI", "AH", "QI"}.issubset(got)


def test_case_insensitive_is_word(dict_small: Dictionary) -> None:
    assert dict_small.is_word("cat")
    assert dict_small.is_word("Cat")
    assert dict_small.is_word("CAT")
    assert dict_small.is_word("qi")
    assert dict_small.is_word("Qi")


def test_case_insensitive_is_prefix(dict_small: Dictionary) -> None:
    assert dict_small.is_prefix("tri")
    assert dict_small.is_prefix("TriC")


def test_case_insensitive_get_words_from_letters(dict_small: Dictionary) -> None:
    assert dict_small.get_words_from_letters(["c", "a", "t"]) == ["CAT"]


def test_missing_wordlist_raises_file_not_found(tmp_path: Path) -> None:
    missing = str(tmp_path / "no_such_wordlist.txt")
    with pytest.raises(FileNotFoundError, match="Wordlist file not found"):
        Dictionary(missing)


def test_default_wordlist_path_points_under_data() -> None:
    assert "wordlists" in DEFAULT_WORDLIST_PATH.replace("\\", "/")
    assert DEFAULT_WORDLIST_PATH.endswith("TWL06.txt")
