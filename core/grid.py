"""Sparse 2D grid with dictionary validation of contiguous letter runs (Bananagrams)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.dictionary import Dictionary


def _normalize_letter(letter: str) -> str | None:
    if len(letter) != 1:
        return None
    u = letter.upper()
    if not ("A" <= u <= "Z"):
        return None
    return u


class GridValidator:
    """Letters on an unbounded sparse grid; every 2+ contiguous run in a row/column must be a word."""

    def __init__(self, dictionary: Dictionary) -> None:
        self._dictionary = dictionary
        self._cells: dict[tuple[int, int], str] = {}

    def place_letter(self, row: int, col: int, letter: str) -> bool:
        ch = _normalize_letter(letter)
        if ch is None:
            return False
        if (row, col) in self._cells:
            return False
        self._cells[row, col] = ch
        if not self.validate_affected([(row, col)]):
            del self._cells[row, col]
            return False
        return True

    def remove_letter(self, row: int, col: int) -> str | None:
        return self._cells.pop((row, col), None)

    def validate_board(self) -> bool:
        if not self._cells:
            return True
        rows = {r for r, _ in self._cells}
        cols = {c for _, c in self._cells}
        for r in rows:
            if not self._validate_row(r):
                return False
        for c in cols:
            if not self._validate_col(c):
                return False
        return True

    def validate_affected(self, changed_positions: list[tuple[int, int]]) -> bool:
        rows = {r for r, _ in changed_positions}
        cols = {c for _, c in changed_positions}
        for r in rows:
            if not self._validate_row(r):
                return False
        for c in cols:
            if not self._validate_col(c):
                return False
        return True

    def get_letter(self, row: int, col: int) -> str | None:
        return self._cells.get((row, col))

    def get_board_state(self) -> dict[tuple[int, int], str]:
        return dict(self._cells)

    def is_empty(self) -> bool:
        return len(self._cells) == 0

    def get_bounds(self) -> tuple[int, int, int, int]:
        if not self._cells:
            raise ValueError("cannot get bounds of an empty board")
        rows = [r for r, _ in self._cells]
        cols = [c for _, c in self._cells]
        return min(rows), max(rows), min(cols), max(cols)

    def _validate_row(self, row: int) -> bool:
        cols = sorted(c for r, c in self._cells if r == row)
        return self._validate_axis_segments(cols, lambda c: self._cells[row, c])

    def _validate_col(self, col: int) -> bool:
        rows_sorted = sorted(r for r, c in self._cells if c == col)
        return self._validate_axis_segments(rows_sorted, lambda r: self._cells[r, col])

    def _validate_axis_segments(
        self,
        sorted_indices: list[int],
        letter_at: Callable[[int], str],
    ) -> bool:
        i = 0
        n = len(sorted_indices)
        while i < n:
            j = i + 1
            while j < n and sorted_indices[j] == sorted_indices[j - 1] + 1:
                j += 1
            segment = sorted_indices[i:j]
            if len(segment) >= 2:
                s = "".join(letter_at(idx) for idx in segment)
                if not self._dictionary.is_word(s):
                    return False
            i = j
        return True
