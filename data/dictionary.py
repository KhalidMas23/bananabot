"""Trie-backed dictionary for fast word and prefix lookup (Bananagrams solver)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def _default_wordlist_path() -> str:
    return str(Path(__file__).resolve().parent / "wordlists" / "TWL06.txt")


DEFAULT_WORDLIST_PATH: str = _default_wordlist_path()


class _TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.is_end: bool = False


class Dictionary:
    """In-memory prefix trie loaded from a plaintext wordlist (one word per line)."""

    def __init__(self, wordlist_path: str = DEFAULT_WORDLIST_PATH) -> None:
        path = Path(wordlist_path)
        if not path.is_file():
            raise FileNotFoundError(f"Wordlist file not found: {path.resolve()}")

        self._root = _TrieNode()
        with path.open(encoding="utf-8", errors="replace", newline=None) as f:
            for line in f:
                word = line.strip().upper()
                if word:
                    self._insert(word)

    def _insert(self, word: str) -> None:
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, _TrieNode())
        node.is_end = True

    @staticmethod
    def _normalize_query(s: str) -> str | None:
        if not s:
            return None
        upper = s.upper()
        for c in upper:
            if not ("A" <= c <= "Z"):
                return None
        return upper

    def _walk(self, s: str) -> _TrieNode | None:
        key = self._normalize_query(s)
        if key is None:
            return None
        node = self._root
        for ch in key:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def is_word(self, word: str) -> bool:
        """Return True if word is in the dictionary. Case-insensitive."""
        node = self._walk(word)
        return node is not None and node.is_end

    def is_prefix(self, prefix: str) -> bool:
        """Return True if any dictionary word starts with this prefix. Used to prune search early."""
        if not prefix:
            return self._has_any_word()
        return self._walk(prefix) is not None

    def _has_any_word(self) -> bool:
        stack = [self._root]
        while stack:
            n = stack.pop()
            if n.is_end:
                return True
            stack.extend(n.children.values())
        return False

    def get_words_from_letters(self, available_letters: list[str]) -> list[str]:
        """Return all valid dictionary words formable from available_letters (multiset — repeats matter).

        Used by scorer to evaluate hand flexibility after a placement.
        """
        counts: Counter[str] = Counter()
        for ch in available_letters:
            u = ch.upper()
            if len(u) != 1 or not ("A" <= u <= "Z"):
                continue
            counts[u] += 1

        found: set[str] = set()

        def dfs(node: _TrieNode, path: list[str]) -> None:
            if node.is_end and len(path) >= 2:
                found.add("".join(path))
            for letter, child in node.children.items():
                c = counts[letter]
                if c <= 0:
                    continue
                counts[letter] = c - 1
                path.append(letter)
                dfs(child, path)
                path.pop()
                counts[letter] = c

        dfs(self._root, [])
        return sorted(found)
