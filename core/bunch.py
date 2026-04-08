"""Deterministic tile accounting for solo Bananagrams (full known set)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def _standard_tile_distribution() -> Counter[str]:
    """Return the official 144-tile Bananagrams letter counts."""
    return Counter(
        {
            "A": 13,
            "B": 3,
            "C": 3,
            "D": 6,
            "E": 18,
            "F": 3,
            "G": 4,
            "H": 3,
            "I": 12,
            "J": 2,
            "K": 2,
            "L": 5,
            "M": 3,
            "N": 8,
            "O": 11,
            "P": 3,
            "Q": 2,
            "R": 9,
            "S": 6,
            "T": 9,
            "U": 6,
            "V": 3,
            "W": 3,
            "X": 2,
            "Y": 3,
            "Z": 2,
        }
    )


def _counts_from_letters(letters: Iterable[str]) -> Counter[str]:
    """Build a Counter from single-letter strings; validate each entry."""
    out: Counter[str] = Counter()
    for raw in letters:
        if not isinstance(raw, str) or len(raw) != 1:
            raise ValueError(f"expected single-character letter string, got {raw!r}")
        letter = raw.upper()
        if letter not in _LETTERS:
            raise ValueError(f"not a Bananagrams letter: {raw!r}")
        out[letter] += 1
    return out


_LETTERS = frozenset(_standard_tile_distribution().keys())


class BunchTracker:
    """Ledger for tiles: bunch = full_set - hand - placed - dumped (derived, not stored)."""

    def __init__(self) -> None:
        self._full_set: Counter[str] = _standard_tile_distribution()
        self._hand: Counter[str] = Counter()
        self._placed: Counter[str] = Counter()
        self._dumped: Counter[str] = Counter()

    def _current_bunch(self) -> Counter[str]:
        bunch = self._full_set.copy()
        bunch.subtract(self._hand)
        bunch.subtract(self._placed)
        bunch.subtract(self._dumped)
        return bunch

    def _require_subset(
        self, available: Counter[str], need: Counter[str], location: str
    ) -> None:
        for letter, n in need.items():
            if n <= 0:
                continue
            have = available[letter]
            if have < n:
                raise ValueError(
                    f"not enough {letter!r} in {location} "
                    f"(need {n}, have {have})"
                )

    def draw_to_hand(self, letters: list[str]) -> None:
        """Move tiles from the bunch into the hand."""
        need = _counts_from_letters(letters)
        bunch = self._current_bunch()
        self._require_subset(bunch, need, "bunch")
        self._hand.update(need)

    def place_tiles(self, letters: list[str]) -> None:
        """Move tiles from the hand onto the board (placed)."""
        need = _counts_from_letters(letters)
        self._require_subset(self._hand, need, "hand")
        self._hand.subtract(need)
        self._placed.update(need)

    def return_to_hand(self, letters: list[str]) -> None:
        """Move tiles from placed back into the hand."""
        need = _counts_from_letters(letters)
        self._require_subset(self._placed, need, "placed")
        self._placed.subtract(need)
        self._hand.update(need)

    def dump(self, letter: str) -> None:
        """Move one tile from the hand into the dumped pool."""
        need = _counts_from_letters([letter])
        self._require_subset(self._hand, need, "hand")
        self._hand.subtract(need)
        self._dumped.update(need)

    def draw_from_bunch(self, letters: list[str]) -> None:
        """Move tiles from the bunch into the hand (e.g. after a peel or post-dump draw)."""
        self.draw_to_hand(letters)

    def peek_bunch(self) -> Counter[str]:
        """Return a copy of current bunch contents (read-only for callers)."""
        return self._current_bunch().copy()

    def peek_hand(self) -> Counter[str]:
        """Return a copy of the hand contents (read-only for callers)."""
        return self._hand.copy()

    def bunch_size(self) -> int:
        """Count of tiles still in the bunch."""
        return sum(self._current_bunch().values())

    def peek_next(self, n: int = 3) -> list[str]:
        """Deterministic next ``n`` tiles from the bunch (A..Z order), or fewer if bunch is smaller."""
        if n <= 0:
            return []
        bunch = self._current_bunch()
        out: list[str] = []
        for letter in sorted(bunch.keys()):
            take = min(bunch[letter], n - len(out))
            out.extend([letter] * take)
            if len(out) >= n:
                break
        return out

    def sync_observed_hand(self, letters: list[str]) -> None:
        """Set hand multiset from an observed hand (vision). Bunch is derived; placed/dumped unchanged."""
        need = _counts_from_letters(letters)
        max_avail = self._full_set.copy()
        max_avail.subtract(self._placed)
        max_avail.subtract(self._dumped)
        self._require_subset(max_avail, need, "available (full - placed - dumped)")
        self._hand = need

    def hand_size(self) -> int:
        """Count of tiles in the hand."""
        return sum(self._hand.values())

    def is_valid_state(self) -> bool:
        """True if all 144 tiles are accounted for with no negative counts."""
        if sum(self._full_set.values()) != 144:
            return False
        bunch = self._current_bunch()
        if any(v < 0 for v in bunch.values()):
            return False
        for letter in self._full_set:
            total = self._full_set[letter]
            split = (
                self._hand[letter]
                + self._placed[letter]
                + self._dumped[letter]
                + bunch[letter]
            )
            if split != total:
                return False
        for letter in self._hand:
            if letter not in self._full_set and self._hand[letter]:
                return False
        for letter in self._placed:
            if letter not in self._full_set and self._placed[letter]:
                return False
        for letter in self._dumped:
            if letter not in self._full_set and self._dumped[letter]:
                return False
        return True
