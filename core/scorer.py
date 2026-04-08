"""Placement and restructure scoring for the Bananagrams solver (no validation — caller validates)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.dictionary import Dictionary

Grid = dict[tuple[int, int], str]

SCRABBLE_VALUES: dict[str, int] = {
    "A": 1,
    "B": 3,
    "C": 3,
    "D": 2,
    "E": 1,
    "F": 4,
    "G": 2,
    "H": 4,
    "I": 1,
    "J": 8,
    "K": 5,
    "L": 1,
    "M": 3,
    "N": 1,
    "O": 1,
    "P": 3,
    "Q": 10,
    "R": 1,
    "S": 1,
    "T": 1,
    "U": 1,
    "V": 4,
    "W": 4,
    "X": 8,
    "Y": 4,
    "Z": 10,
}

ISOLATION_PENALTY_WEIGHT = 2.0
ANTI_PARALLEL_VERTICAL_BONUS = 1.5  # applied unless parallel-adjacent vertical stack detected
ANCHOR_LENGTH_TIEBREAKER = 0.1

# CONTEXT.md: base restructure cost = stranded subtree size + connection count of the candidate word.
# Words that are articulation points multiply that base by AP_PENALTY_MULTIPLIER so a run that looks
# like a single cheap connector but disconnects the board is not ranked as easy to move (numeric
# behavior unchanged from the historical literal 10.0 factor).
AP_PENALTY_MULTIPLIER = 10.0


@dataclass(frozen=True)
class WordRun:
    """One contiguous 2+ letter run on the board (horizontal or vertical)."""

    word: str
    positions: tuple[tuple[int, int], ...]
    direction: str  # 'H' or 'V'

    @property
    def pos_set(self) -> frozenset[tuple[int, int]]:
        return frozenset(self.positions)


def _scrabble_sum(letters: str) -> int:
    return sum(SCRABBLE_VALUES.get(ch.upper(), 0) for ch in letters if ch.upper() in SCRABBLE_VALUES)


def _placement_positions(position: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    r0, c0 = position
    if direction == "H":
        return [(r0, c0 + i) for i in range(length)]
    if direction == "V":
        return [(r0 + i, c0) for i in range(length)]
    raise ValueError("direction must be 'H' or 'V'")


def _has_parallel_adjacent_vertical_stack(grid: Grid, new_positions: list[tuple[int, int]], new_col: int) -> bool:
    """True if this vertical placement sits beside another vertical run of 2+ with no horizontal gap (same rows)."""

    rows = sorted(r for r, c in new_positions if c == new_col)
    if not rows:
        return False

    def vertical_run_len_at(row: int, col: int) -> int:
        if grid.get((row, col)) is None:
            return 0
        r = row
        while grid.get((r - 1, col)) is not None:
            r -= 1
        start = r
        r = row
        while grid.get((r + 1, col)) is not None:
            r += 1
        return r - start + 1

    for r in rows:
        for dc in (-1, 1):
            nc = new_col + dc
            if grid.get((r, nc)) is None:
                continue
            if vertical_run_len_at(r, nc) >= 2:
                return True
    return False


def _flexibility_count(dictionary: Dictionary, hand_after: list[str]) -> int:
    getter = getattr(dictionary, "get_words_from_letters", None)
    if callable(getter):
        return len(getter(hand_after))
    return len(_words_from_letters_trie(dictionary, hand_after))


def _words_from_letters_trie(dictionary: Dictionary, available_letters: list[str]) -> list[str]:
    """Fallback when get_words_from_letters is missing: enumerate via prefix checks (slower)."""
    counts: Counter[str] = Counter()
    for ch in available_letters:
        u = ch.upper()
        if len(u) == 1 and "A" <= u <= "Z":
            counts[u] += 1

    found: set[str] = set()

    def dfs(prefix: str) -> None:
        if len(prefix) >= 2 and dictionary.is_word(prefix):
            found.add(prefix)
        if not dictionary.is_prefix(prefix):
            return
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            c = counts[letter]
            if c <= 0:
                continue
            counts[letter] = c - 1
            dfs(prefix + letter)
            counts[letter] = c

    dfs("")
    return sorted(found)


def _isolation_nonparticipating_tile_count(dictionary: Dictionary, hand_after: list[str]) -> int:
    """Count hand tiles whose letter appears in no word from get_words_from_letters(hand_after)."""
    words = _flexibility_words(dictionary, hand_after)
    isolated = 0
    for ch in hand_after:
        u = ch.upper()
        if len(u) != 1 or not ("A" <= u <= "Z"):
            continue
        if not any(u in w for w in words):
            isolated += 1
    return isolated


def _flexibility_words(dictionary: Dictionary, hand_after: list[str]) -> list[str]:
    getter = getattr(dictionary, "get_words_from_letters", None)
    if callable(getter):
        return list(getter(hand_after))
    return _words_from_letters_trie(dictionary, hand_after)


def _extract_word_runs(grid: Grid) -> list[WordRun]:
    runs: list[WordRun] = []
    if not grid:
        return runs

    rows = defaultdict(list)
    cols = defaultdict(list)
    for (r, c), ch in grid.items():
        u = ch.upper()
        if len(u) == 1 and "A" <= u <= "Z":
            rows[r].append(c)
            cols[c].append(r)

    for r, cs in rows.items():
        cs_sorted = sorted(cs)
        i = 0
        while i < len(cs_sorted):
            j = i + 1
            while j < len(cs_sorted) and cs_sorted[j] == cs_sorted[j - 1] + 1:
                j += 1
            seg = cs_sorted[i:j]
            if len(seg) >= 2:
                positions = tuple((r, c) for c in seg)
                word = "".join(grid[r, c] for c in seg)
                runs.append(WordRun(word=word.upper(), positions=positions, direction="H"))
            i = j

    for c, rs in cols.items():
        rs_sorted = sorted(rs)
        i = 0
        while i < len(rs_sorted):
            j = i + 1
            while j < len(rs_sorted) and rs_sorted[j] == rs_sorted[j - 1] + 1:
                j += 1
            seg = rs_sorted[i:j]
            if len(seg) >= 2:
                positions = tuple((r, c) for r in seg)
                word = "".join(grid[r, c] for r in seg)
                runs.append(WordRun(word=word.upper(), positions=positions, direction="V"))
            i = j

    return runs


def _build_word_graph(runs: list[WordRun]) -> list[set[int]]:
    """Undirected adjacency by shared grid cell."""
    n = len(runs)
    adj: list[set[int]] = [set() for _ in range(n)]
    pos_to_runs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, run in enumerate(runs):
        for p in run.positions:
            pos_to_runs[p].append(i)
    for idxs in pos_to_runs.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                u, v = idxs[a], idxs[b]
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _tarjan_articulation_points(n: int, adj: list[set[int]]) -> list[bool]:
    if n == 0:
        return []
    if n == 1:
        return [True]

    visited = [False] * n
    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    ap = [False] * n
    time_ref = [0]

    def dfs(u: int) -> None:
        children = 0
        visited[u] = True
        time_ref[0] += 1
        disc[u] = low[u] = time_ref[0]
        for v in adj[u]:
            if not visited[v]:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and children > 1:
                    ap[u] = True
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if not visited[i]:
            dfs(i)
    return ap


def _component_tile_counts_after_removal(
    n: int,
    adj: list[set[int]],
    removed: int,
    runs: list[WordRun],
) -> list[int]:
    """Sizes (unique board cells) of each connected component in G \\ {removed}."""
    remaining = [i for i in range(n) if i != removed]
    seen = [False] * n
    sizes: list[int] = []

    def tiles_for_component(start: int) -> int:
        stack = [start]
        seen[start] = True
        cells: set[tuple[int, int]] = set()
        while stack:
            u = stack.pop()
            cells.update(runs[u].positions)
            for v in adj[u]:
                if v == removed or seen[v]:
                    continue
                seen[v] = True
                stack.append(v)
        return len(cells)

    for i in remaining:
        if not seen[i]:
            sizes.append(tiles_for_component(i))
    return sizes


def _stranded_tile_count(n: int, adj: list[set[int]], removed: int, runs: list[WordRun]) -> int:
    if n <= 1:
        return 0
    comps = _component_tile_counts_after_removal(n, adj, removed, runs)
    if len(comps) <= 1:
        return 0
    total = sum(comps)
    largest = max(comps)
    return total - largest


@dataclass
class RestructureCandidate:
    word: str
    positions: list[tuple[int, int]]
    direction: str
    cost: float
    is_articulation_point: bool
    stranded_tile_count: int
    connection_count: int
    reasoning: str


class PlacementScorer:
    """Scores candidate placements. Does not validate — GridValidator is used by the caller."""

    def __init__(self, dictionary: Dictionary) -> None:
        self._dictionary = dictionary

    def score_anchor(self, word: str, hand: list[str]) -> float:  # noqa: ARG002 — API for solver
        w = word.upper()
        return float(_scrabble_sum(w) + ANCHOR_LENGTH_TIEBREAKER * len(w))

    def score_placement(
        self,
        word: str,
        position: tuple[int, int],
        direction: str,
        grid: Grid,
        hand_after: list[str],
        dictionary: Dictionary | None = None,
    ) -> float:
        """Score a non-anchor placement (caller validates the board separately).

        Combines Scrabble value for letters taken from the hand, hand flexibility
        (count of dictionary words formable from ``hand_after``), minus an isolation
        penalty: each tile in ``hand_after`` whose letter appears in no word returned
        by ``get_words_from_letters(hand_after)`` counts toward that penalty, scaled by
        ``ISOLATION_PENALTY_WEIGHT``.

        Applies a bonus to placements that do not form a parallel vertical stack adjacent
        to an existing vertical word. This discourages high-incidental-constraint board
        shapes without requiring explicit L/Z geometry detection.
        """
        d = dictionary if dictionary is not None else self._dictionary
        w = word.upper()
        cells = _placement_positions(position, direction, len(w))

        from_hand_value = 0
        for i, cell in enumerate(cells):
            ch = w[i]
            existing = grid.get(cell)
            if existing is not None and existing.upper() == ch:
                continue
            from_hand_value += SCRABBLE_VALUES.get(ch, 0)

        flex = float(_flexibility_count(d, hand_after))
        isolated = float(_isolation_nonparticipating_tile_count(d, hand_after))
        score = float(from_hand_value) + flex - ISOLATION_PENALTY_WEIGHT * isolated

        shape_bonus = ANTI_PARALLEL_VERTICAL_BONUS
        if direction == "V" and len(cells) >= 1:
            col = cells[0][1]
            if _has_parallel_adjacent_vertical_stack(grid, cells, col):
                shape_bonus = 0.0
        score += shape_bonus

        return score


class RestructureScorer:
    """Ranks word runs to remove when the board may need restructuring."""

    def __init__(self, dictionary: Dictionary) -> None:
        self._dictionary = dictionary

    def score_restructure(
        self,
        grid: Grid,
        hand: list[str],
        dictionary: Dictionary | None = None,
    ) -> list[RestructureCandidate]:
        d = dictionary if dictionary is not None else self._dictionary
        runs = _extract_word_runs(grid)
        n = len(runs)
        if n == 0:
            return []

        adj = _build_word_graph(runs)
        ap = _tarjan_articulation_points(n, adj)

        raw: list[tuple[float, RestructureCandidate]] = []

        for i, run in enumerate(runs):
            conn = len(adj[i])
            if n == 1:
                stranded = 0
                is_ap = True
                base_cost = float(len(run.positions))
                cost = AP_PENALTY_MULTIPLIER * base_cost
                exclude = False
            else:
                is_ap = ap[i]
                stranded = _stranded_tile_count(n, adj, i, runs)
                base_cost = float(stranded + conn)
                exclude = bool(is_ap and stranded > 3)
                cost = AP_PENALTY_MULTIPLIER * base_cost if is_ap and not exclude else base_cost

            if exclude:
                continue

            letters_free = list(run.word)
            extended = [ch.upper() for ch in hand if len(ch) == 1 and "A" <= ch.upper() <= "Z"]
            extended.extend(letters_free)
            n_opts = _flexibility_count(d, extended)

            ap_phrase = "an articulation point" if is_ap else "not an articulation point"
            reasoning = (
                f"Move {run.word} (cost {cost:g}): {conn} connection"
                f"{'s' if conn != 1 else ''}, {ap_phrase}. "
                f"Freeing {','.join(letters_free)} gives {n_opts} placement options with current hand."
            )

            cand = RestructureCandidate(
                word=run.word,
                positions=list(run.positions),
                direction=run.direction,
                cost=cost,
                is_articulation_point=is_ap,
                stranded_tile_count=stranded,
                connection_count=conn,
                reasoning=reasoning,
            )
            raw.append((cost, cand))

        raw.sort(key=lambda t: (t[0], t[1].word))
        return [c for _, c in raw[:5]]
