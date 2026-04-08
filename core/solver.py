"""Greedy solver loop: placements, DUMP, PEEL, GAME_OVER, RESTRUCTURE, WAIT_START (Bananagrams v1).

Pre-round: when ``game_active`` is False and the solver grid is still empty, the loop
returns ``WAIT_START`` so an empty observed hand is not mistaken for GAME OVER (e.g.
right after camera calibration before tiles are dealt).

Vision / mapper contract (BunchTracker ledger):
- ``sync_observed_hand`` updates only the hand multiset from vision; it does not move
  tiles into ``placed`` or ``dumped``.
- The caller must keep the ledger consistent with reality: call ``place_tiles`` when
  tiles move from hand to board, ``dump`` when a dump is confirmed, and keep
  ``sync_observed_hand`` aligned with the visible hand. Otherwise ``bunch_size`` and
  post-dump simulation (which uses ``peek_bunch()``) will be wrong. ``sync_observed_hand``
  only updates hand letters, not placed or dumped state.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from core.bunch import BunchTracker
from core.grid import GridValidator
from core.scorer import SCRABBLE_VALUES, PlacementScorer, RestructureScorer

GridState = dict[tuple[int, int], str]

# Standard Bananagrams opening rack (2–4 players). Solo / bot uses the same count.
STARTING_RACK_TILES = 21


@runtime_checkable
class DictionaryProtocol(Protocol):
    def is_word(self, word: str) -> bool: ...

    def is_prefix(self, prefix: str) -> bool: ...

    def get_words_from_letters(self, letters: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class SolverRecommendation:
    action: Literal["PLACE", "DUMP", "PEEL", "GAME_OVER", "RESTRUCTURE", "WAIT_START"]
    details: dict
    score: float
    reasoning: str


@dataclass
class _Placement:
    word: str
    row: int
    col: int
    direction: str
    hand_after: list[str]
    score: float
    letters_from_hand: list[str]


def _peek_first_n_from_counter(bunch_counter: Counter[str], n: int) -> list[str]:
    """Deterministic next ``n`` tiles: A..Z order, same rule as ``BunchTracker.peek_next``."""
    if n <= 0:
        return []
    out: list[str] = []
    for letter in sorted(bunch_counter.keys()):
        take = min(bunch_counter[letter], n - len(out))
        out.extend([letter] * take)
        if len(out) >= n:
            break
    return out


def _simulate_post_dump_hand(
    hand: list[str], dump_tile: str, bunch: BunchTracker, draw_n: int
) -> tuple[list[str], list[str]]:
    """Return (hand after dump + draws, drawn letters). Dump tile re-enters bunch before draw."""
    bc = bunch.peek_bunch()
    bc[dump_tile] = bc.get(dump_tile, 0) + 1
    drawn = _peek_first_n_from_counter(bc, draw_n)
    return _hand_without_one(hand, dump_tile) + drawn, drawn


def _normalize_hand(hand: list[str]) -> list[str]:
    return [h.upper() for h in hand if isinstance(h, str) and len(h) == 1 and "A" <= h.upper() <= "Z"]


def _hand_without_one(hand: list[str], letter: str) -> list[str]:
    u = letter.upper()
    out: list[str] = []
    removed = False
    for h in hand:
        if not removed and h.upper() == u:
            removed = True
            continue
        out.append(h.upper())
    return out


def _pick_dump_tile(hand: list[str]) -> str:
    """Lowest Scrabble value; tie-break by letter A..Z."""
    best = hand[0].upper()
    best_v = SCRABBLE_VALUES.get(best, 0)
    for h in hand[1:]:
        u = h.upper()
        v = SCRABBLE_VALUES.get(u, 0)
        if v < best_v or (v == best_v and u < best):
            best, best_v = u, v
    return best


def _clone_grid(grid: GridValidator, dictionary: DictionaryProtocol) -> GridValidator:
    g = GridValidator(dictionary)
    for (r, c), ch in sorted(grid.get_board_state().items()):
        if not g.place_letter(r, c, ch):
            raise RuntimeError("failed to clone grid state")
    return g


def _place_word_on_grid(g: GridValidator, word: str, row0: int, col0: int, direction: str) -> bool:
    for i, ch in enumerate(word):
        r, c = (row0, col0 + i) if direction == "H" else (row0 + i, col0)
        existing = g.get_letter(r, c)
        if existing is not None:
            if existing != ch:
                return False
            continue
        if not g.place_letter(r, c, ch):
            return False
    return g.validate_board()


def _hand_after_using(hand: Counter[str], used_from_hand: list[str]) -> list[str]:
    h = hand.copy()
    for ch in used_from_hand:
        u = ch.upper()
        h[u] -= 1
        if h[u] < 0:
            raise ValueError("hand multiset underflow")
    out: list[str] = []
    for letter in sorted(h.keys()):
        out.extend([letter] * h[letter])
    return out


def _dedupe_placements(raw: list[_Placement]) -> list[_Placement]:
    seen: set[tuple[str, frozenset[tuple[int, int]], str]] = set()
    out: list[_Placement] = []
    for p in raw:
        cells = []
        for i, ch in enumerate(p.word):
            r, c = (p.row, p.col + i) if p.direction == "H" else (p.row + i, p.col)
            cells.append((r, c))
        key = (p.word.upper(), frozenset(cells), p.direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _generate_anchor_placements(
    hand: list[str],
    grid: GridValidator,
    dictionary: DictionaryProtocol,
    scorer: PlacementScorer,
) -> list[_Placement]:
    if not grid.is_empty():
        return []
    words = dictionary.get_words_from_letters(hand)
    hand_ct = Counter(hand)
    out: list[_Placement] = []
    for w in words:
        if len(w) < 2:
            continue
        need = Counter(w)
        if any(need[x] > hand_ct[x] for x in need):
            continue
        used_from_hand = list(w)
        hand_after = _hand_after_using(hand_ct, used_from_hand)
        for direction in ("H", "V"):
            g2 = GridValidator(dictionary)
            row0, col0 = 0, 0
            if not _place_word_on_grid(g2, w, row0, col0, direction):
                continue
            score = float(scorer.score_anchor(w, hand))
            out.append(
                _Placement(
                    word=w,
                    row=row0,
                    col=col0,
                    direction=direction,
                    hand_after=hand_after,
                    score=score,
                    letters_from_hand=used_from_hand,
                )
            )
    return _dedupe_placements(out)


def _generate_extension_placements(
    hand: list[str],
    grid: GridValidator,
    dictionary: DictionaryProtocol,
    scorer: PlacementScorer,
) -> list[_Placement]:
    if grid.is_empty():
        return []
    cells = grid.get_board_state()
    pool: list[str] = list(hand)
    for ch in cells.values():
        pool.append(ch)
    words = dictionary.get_words_from_letters(pool)
    hand_ct = Counter(hand)
    out: list[_Placement] = []
    for w in words:
        if len(w) < 2:
            continue
        for (br, bc), letter in cells.items():
            for k, wk in enumerate(w):
                if wk != letter:
                    continue
                for direction in ("H", "V"):
                    if direction == "H":
                        row0, col0 = br, bc - k
                    else:
                        row0, col0 = br - k, bc
                    used_from_hand: list[str] = []
                    ok = True
                    for i, ch in enumerate(w):
                        r, c = (row0, col0 + i) if direction == "H" else (row0 + i, col0)
                        ex = cells.get((r, c))
                        if ex is not None:
                            if ex != ch:
                                ok = False
                                break
                        else:
                            used_from_hand.append(ch)
                    if not ok:
                        continue
                    need = Counter(used_from_hand)
                    if any(need[x] > hand_ct[x] for x in need):
                        continue
                    if not used_from_hand:
                        continue
                    g2 = _clone_grid(grid, dictionary)
                    if not _place_word_on_grid(g2, w, row0, col0, direction):
                        continue
                    hand_after = _hand_after_using(hand_ct, used_from_hand)
                    grid_before: GridState = dict(grid.get_board_state())
                    sc = float(
                        scorer.score_placement(
                            w,
                            (row0, col0),
                            direction,
                            grid_before,
                            hand_after,
                            dictionary,
                        )
                    )
                    out.append(
                        _Placement(
                            word=w,
                            row=row0,
                            col=col0,
                            direction=direction,
                            hand_after=hand_after,
                            score=sc,
                            letters_from_hand=used_from_hand,
                        )
                    )
    return _dedupe_placements(out)


def _placeable_letters(placements: list[_Placement]) -> set[str]:
    # Stranded detection operates on letter multisets, not unique tile IDs.
    # Two physical tiles of the same letter are indistinguishable here.
    # Per-tile tracking (vision IDs) is handled upstream in mapper.py.
    s: set[str] = set()
    for p in placements:
        for ch in p.letters_from_hand:
            s.add(ch.upper())
    return s


def _stranded_tiles(hand: list[str], placeable: set[str]) -> list[str]:
    return [h for h in hand if h.upper() not in placeable]


def _stranded_pair_can_form_word(dictionary: DictionaryProtocol, stranded: list[str]) -> bool:
    n = len(stranded)
    for i in range(n):
        for j in range(i + 1, n):
            wds = dictionary.get_words_from_letters([stranded[i].upper(), stranded[j].upper()])
            if any(len(x) >= 2 for x in wds):
                return True
    return False


def _scrabble_sum_letters(letters: Iterable[str]) -> int:
    return sum(SCRABBLE_VALUES.get(ch.upper(), 0) for ch in letters)


def _count_placements_for_hand(
    hand: list[str],
    grid: GridValidator,
    dictionary: DictionaryProtocol,
    scorer: PlacementScorer,
) -> int:
    return len(
        _generate_anchor_placements(hand, grid, dictionary, scorer)
        + _generate_extension_placements(hand, grid, dictionary, scorer)
    )


def _best_placement_for_hand(
    hand: list[str],
    grid: GridValidator,
    dictionary: DictionaryProtocol,
    scorer: PlacementScorer,
) -> tuple[_Placement | None, float]:
    raw = _generate_anchor_placements(hand, grid, dictionary, scorer) + _generate_extension_placements(
        hand, grid, dictionary, scorer
    )
    if not raw:
        return None, float("-inf")
    raw.sort(key=lambda p: (-p.score, -len(p.word), p.word, p.row, p.col, p.direction))
    best = raw[0]
    return best, best.score


def _dump_net_tiles(draw_n: int) -> int:
    """Net tiles added to hand after dump: draw_n - 1."""
    return max(0, draw_n - 1)


def _restructure_beats_dump(best_restructure_cost: float, draw_n: int) -> bool:
    """Prefer restructure when its cost is below the dump-equivalent threshold (tuned to draw size)."""
    net = _dump_net_tiles(draw_n)
    threshold = 4.0 - float(net)
    return best_restructure_cost < threshold


def _wait_start_recommendation(hand_n: list[str]) -> SolverRecommendation:
    """Pre-round UX: empty board, round not yet armed — do not treat empty hand as GAME_OVER."""
    n = len(hand_n)
    if n == 0:
        phase = "deal_tiles"
        body = (
            f"No tiles in the hand zone yet. Deal {STARTING_RACK_TILES} tiles from the bunch into the hand area. "
            f"Solving stays paused until you press G with a full rack."
        )
    elif n < STARTING_RACK_TILES:
        phase = "dealing"
        body = (
            f"The camera sees {n}/{STARTING_RACK_TILES} tile(s) in the hand zone. "
            f"Finish dealing, then press G when the rack is full."
        )
    elif n == STARTING_RACK_TILES:
        phase = "press_g"
        body = (
            f"Full rack ({STARTING_RACK_TILES} tiles) detected. Press G to start the game and enable move advice."
        )
    else:
        phase = "too_many"
        body = (
            f"The camera sees {n} tiles; open with exactly {STARTING_RACK_TILES}. "
            "Remove extras from the hand zone, then press G when the count matches."
        )
    summary = {
        "deal_tiles": "WAIT — set up rack",
        "dealing": "WAIT — dealing",
        "press_g": "WAIT — press G to start",
        "too_many": "WAIT — adjust rack",
    }[phase]
    reasoning = f"Action: {summary}\n{body}"
    return SolverRecommendation(
        action="WAIT_START",
        details={"phase": phase, "hand_count": n},
        score=0.0,
        reasoning=reasoning,
    )


class SolverLoop:
    """Greedy immediate-move recommendations from bunch + grid + dictionary state."""

    def __init__(
        self,
        bunch: BunchTracker,
        grid: GridValidator,
        scorer: PlacementScorer,
        dictionary: DictionaryProtocol,
        restructure_scorer: RestructureScorer | None = None,
    ) -> None:
        self._bunch = bunch
        self._grid = grid
        self._scorer = scorer
        self._dictionary = dictionary
        self._restructure_scorer = restructure_scorer or RestructureScorer(dictionary)

    def on_state_change(self, hand: list[str], *, game_active: bool = True) -> SolverRecommendation:
        hand_n = _normalize_hand(hand)
        self._bunch.sync_observed_hand(hand_n)

        if not game_active and self._grid.is_empty():
            return _wait_start_recommendation(hand_n)

        placements = _generate_anchor_placements(
            hand_n, self._grid, self._dictionary, self._scorer
        ) + _generate_extension_placements(hand_n, self._grid, self._dictionary, self._scorer)
        placements.sort(key=lambda p: (-p.score, -len(p.word), p.word, p.row, p.col, p.direction))
        best_place = placements[0] if placements else None
        current_score = best_place.score if best_place else float("-inf")

        bunch_n = self._bunch.bunch_size()
        dump_tile = _pick_dump_tile(hand_n) if hand_n else None

        draw_n = min(3, bunch_n) if bunch_n else 0

        post_dump_hand_sim: list[str] = []
        post_dump_drawn: list[str] = []
        post_dump_score = float("-inf")
        post_dump_best: _Placement | None = None
        if hand_n and dump_tile is not None and draw_n > 0:
            post_dump_hand_sim, post_dump_drawn = _simulate_post_dump_hand(
                hand_n, dump_tile, self._bunch, draw_n
            )
            post_dump_best, post_dump_score = _best_placement_for_hand(
                post_dump_hand_sim, self._grid, self._dictionary, self._scorer
            )

        placeable = _placeable_letters(placements)
        stranded = _stranded_tiles(hand_n, placeable)
        stranded_count = len(stranded)

        peel_score = float("-inf")
        peel_best: _Placement | None = None
        if stranded_count == 1 and bunch_n >= 1:
            next1 = self._bunch.peek_next(1)
            if next1:
                peel_hand = hand_n + next1
                peel_placements = _generate_anchor_placements(
                    peel_hand, self._grid, self._dictionary, self._scorer
                ) + _generate_extension_placements(
                    peel_hand, self._grid, self._dictionary, self._scorer
                )
                peel_placements.sort(
                    key=lambda p: (-p.score, -len(p.word), p.word, p.row, p.col, p.direction)
                )
                if peel_placements:
                    peel_best = peel_placements[0]
                    peel_score = peel_best.score

        n_placements_now = len(placements)
        n_placements_post = (
            _count_placements_for_hand(post_dump_hand_sim, self._grid, self._dictionary, self._scorer)
            if post_dump_hand_sim
            else 0
        )
        net_dump = _dump_net_tiles(draw_n)

        # Terminal: no placements, empty bunch
        if best_place is None and bunch_n == 0:
            return SolverRecommendation(
                action="GAME_OVER",
                details={},
                score=0.0,
                reasoning=(
                    "Action: GAME OVER\nReason: No valid placements exist and bunch is empty "
                    f"({len(hand_n)} tiles in hand). Board is complete or stuck with no legal moves remaining."
                ),
            )

        # No placements but bunch has tiles → restructure, else PEEL vs DUMP
        if best_place is None and bunch_n > 0:
            if not hand_n:
                return SolverRecommendation(
                    action="GAME_OVER",
                    details={},
                    score=float(current_score if current_score > float("-inf") else 0.0),
                    reasoning=(
                        f"Action: GAME OVER\nReason: Empty hand with no valid placements; bunch has {bunch_n} "
                        f"tiles. Best placement score computed: "
                        f"{current_score if current_score > float('-inf') else 0.0:.2f}."
                    ),
                )
            dt = dump_tile if dump_tile is not None else hand_n[0]
            sim_hand, drawn = _simulate_post_dump_hand(hand_n, dt, self._bunch, draw_n)
            after_best, after_score = _best_placement_for_hand(
                sim_hand, self._grid, self._dictionary, self._scorer
            )
            after_pl = _generate_anchor_placements(
                sim_hand, self._grid, self._dictionary, self._scorer
            ) + _generate_extension_placements(sim_hand, self._grid, self._dictionary, self._scorer)
            n_opts = len(after_pl)
            asc = after_score if after_best is not None else 0.0

            grid_state: GridState = dict(self._grid.get_board_state())
            if grid_state:
                re_cands = self._restructure_scorer.score_restructure(
                    grid_state, hand_n, self._dictionary
                )
                if re_cands and _restructure_beats_dump(re_cands[0].cost, draw_n):
                    cand = re_cands[0]
                    ap_phrase = "an articulation point" if cand.is_articulation_point else "not an articulation point"
                    return SolverRecommendation(
                        action="RESTRUCTURE",
                        details={
                            "word": cand.word,
                            "positions": cand.positions,
                            "direction": cand.direction,
                            "cost": cand.cost,
                        },
                        score=float(-cand.cost),
                        reasoning=(
                            f"Action: RESTRUCTURE — move {cand.word}\n"
                            f"Reason: No clean placement for current hand. Moving {cand.word} "
                            f"({cand.connection_count} connection(s), {ap_phrase}). "
                            f"Restructure cost: {cand.cost:g}. "
                            f"DUMP alternative would draw {len(drawn)} tile(s) (net +{net_dump} to hand at bunch size {bunch_n}). "
                            f"Valid placements now: {n_placements_now}. Estimated post-dump: {n_opts}."
                        ),
                    )

            if stranded_count == 1 and peel_best is not None and peel_score > post_dump_score:
                nxt = self._bunch.peek_next(1)
                return SolverRecommendation(
                    action="PEEL",
                    score=float(peel_score),
                    details={"peek_tile": nxt[0] if nxt else None},
                    reasoning=(
                        f"Action: PEEL\nReason: 1 stranded tile ({stranded[0]}). No placement for the full hand. "
                        f"After drawing next tile {nxt[0] if nxt else '?'}, best valid placement scores {peel_score:.2f} "
                        f"vs post-dump best {post_dump_score:.2f}. "
                        f"Valid placements now: {n_placements_now}. Post-dump hand would have {n_opts} valid placement(s) "
                        f"after drawing {len(drawn)} tile(s)."
                    ),
                )

            return SolverRecommendation(
                action="DUMP",
                details={"tile": dt, "draw_preview": drawn, "post_dump_hand": sim_hand},
                score=float(asc),
                reasoning=(
                    f"Action: DUMP {dt}\nReason: No valid placement exists for the current hand "
                    f"({''.join(hand_n)}). Post-dump hand draws {len(drawn)} tile(s) (bunch has {bunch_n} remaining). "
                    f"Valid placements now: {n_placements_now}. Estimated post-dump: {n_opts}. "
                    f"Dumping gains net +{net_dump} tile(s) to hand. Best score after: {asc:.2f}."
                ),
            )

        stuck_pairs = (
            stranded_count >= 2
            and not _stranded_pair_can_form_word(self._dictionary, stranded)
            and bunch_n > 0
        )

        if stuck_pairs and dump_tile is not None:
            sim_hand, drawn = _simulate_post_dump_hand(hand_n, dump_tile, self._bunch, draw_n)
            after_pl = _generate_anchor_placements(
                sim_hand, self._grid, self._dictionary, self._scorer
            ) + _generate_extension_placements(sim_hand, self._grid, self._dictionary, self._scorer)
            ps = after_pl[0].score if after_pl else float("-inf")
            n_opts = len(after_pl)
            return SolverRecommendation(
                action="DUMP",
                details={"tile": dump_tile, "draw_preview": drawn, "forced": True, "post_dump_hand": sim_hand},
                score=float(ps if ps > float("-inf") else 0.0),
                reasoning=(
                    f"Action: DUMP {dump_tile}\nReason: {stranded_count} stranded tiles "
                    f"({''.join(stranded)}) share no valid two-tile dictionary combination; hand is stuck. "
                    f"Post-dump hand draws {len(drawn)} tile(s) (bunch has {bunch_n} remaining). "
                    f"Fastest path: dump lowest-value tile {dump_tile} (Scrabble value "
                    f"{SCRABBLE_VALUES.get(dump_tile, 0)} for selection only). Post-dump best score "
                    f"{ps:.2f} from {n_opts} valid placement(s)."
                ),
            )

        # Build competing actions (scores are PlacementScorer outputs only)
        candidates: list[tuple[str, float, dict, str]] = []

        if best_place is not None:
            burned = "".join(best_place.letters_from_hand)
            scr = _scrabble_sum_letters(best_place.letters_from_hand)
            rem = len(best_place.hand_after)
            flex_after = len(self._dictionary.get_words_from_letters(best_place.hand_after))
            place_reason = (
                f"Action: Place {best_place.word} at row {best_place.row}, col {best_place.col} "
                f"({'horizontal' if best_place.direction == 'H' else 'vertical'})\n"
                f"Reason: Burns {len(burned)} tiles ({','.join(burned)}), Scrabble sum on new squares {scr}. "
                f"Placement score {best_place.score:.2f}. Remaining hand has {rem} tiles with "
                f"{flex_after} words formable from letters (flexibility proxy). "
                f"{stranded_count} stranded tile(s) in current hand before this move."
            )
            candidates.append(
                (
                    "PLACE",
                    best_place.score,
                    {
                        "word": best_place.word,
                        "row": best_place.row,
                        "col": best_place.col,
                        "direction": best_place.direction,
                        "tiles_used": best_place.letters_from_hand,
                        "hand_after": best_place.hand_after,
                    },
                    place_reason,
                )
            )

        # Score-based DUMP vs PLACE only on an empty grid: with extensions, prefer playing
        # onto the board when a legal placement exists (see integration / Bananagrams flow).
        # Anchor-phase "dump for better draw" remains allowed when the board is still empty.
        allow_dump = (
            self._grid.is_empty()
            and bunch_n > 0
            and draw_n > 0
            and post_dump_best is not None
            and dump_tile is not None
            and post_dump_score > current_score
        )
        if stranded_count == 1 and allow_dump and best_place is None:
            allow_dump = post_dump_score > peel_score

        if allow_dump and dump_tile is not None:
            gain = n_placements_post - n_placements_now
            dump_reason = (
                f"Action: DUMP {dump_tile}\nReason: Post-dump best placement score {post_dump_score:.2f} "
                f"beats current best {current_score:.2f}. Post-dump hand draws {len(post_dump_drawn)} tile(s) "
                f"(bunch has {bunch_n} remaining); hand size after: {len(post_dump_hand_sim)}. "
                f"Valid placements now: {n_placements_now}. Estimated post-dump: {n_placements_post}. "
                f"Net change in placement options: {gain:+d}. "
                f"Scrabble value of {dump_tile} was used only to pick the dump tile, not added to these scores."
            )
            if stranded_count == 1:
                dump_reason += (
                    f" Single stranded tile: peel upside {peel_score:.2f} is lower than post-dump "
                    f"{post_dump_score:.2f}."
                )
            candidates.append(
                (
                    "DUMP",
                    post_dump_score,
                    {
                        "tile": dump_tile,
                        "draw_preview": list(post_dump_drawn),
                        "post_dump_hand": list(post_dump_hand_sim),
                    },
                    dump_reason,
                )
            )

        if not candidates:
            if best_place:
                bp = current_score if current_score > float("-inf") else 0.0
                return SolverRecommendation(
                    action="PLACE",
                    details={
                        "word": best_place.word,
                        "row": best_place.row,
                        "col": best_place.col,
                        "direction": best_place.direction,
                    },
                    score=best_place.score,
                    reasoning=(
                        f"Action: Place (fallback)\nReason: No alternative candidates. "
                        f"Hand size {len(hand_n)}, bunch size {bunch_n}, best placement score {bp:.2f}."
                    ),
                )
            bp = current_score if current_score > float("-inf") else 0.0
            return SolverRecommendation(
                action="GAME_OVER",
                details={},
                score=0.0,
                reasoning=(
                    f"Action: GAME OVER\nReason: No legal actions. Hand size {len(hand_n)}, bunch size {bunch_n}, "
                    f"best placement score {bp:.2f}."
                ),
            )

        action, score, details, reasoning = max(candidates, key=lambda t: (t[1], t[0] == "PLACE"))
        return SolverRecommendation(
            action=action,  # type: ignore[arg-type]
            details=details,
            score=score,
            reasoning=reasoning,
        )
