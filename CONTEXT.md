# Bananagrams Solver — Project Context

## Overview
A real-time Bananagrams solver for solo play. As letters are added to the hand, the solver identifies the optimal board arrangement incrementally — fitting new letters into the existing grid with minimal restructuring rather than resolving from scratch each time.

## Scope
**In scope (v1):**
- Solo play only
- Full tile set known upfront (144 tiles)
- Greedy immediate-move optimization
- Python logic layer
- Multiple dictionary support (TWL, SOWPODS)

**Explicitly out of scope for v1:**
- Multiplayer (planned v2 — see extension notes below)
- Probabilistic bunch tracking
- Lookahead / multi-move optimization
- Visual board (planned after Python logic is complete)

---

## Stack
- **Language:** Python
- **Editor:** Cursor
- **Visual layer:** To be added later (likely React)
- **Dictionaries:** Plain .txt wordlists (TWL06, SOWPODS)

---

## Project Structure
```
bananagrams/
├── core/
│   ├── grid.py          ← GridValidator
│   ├── scorer.py        ← PlacementScorer
│   ├── bunch.py         ← BunchTracker
│   └── solver.py        ← SolverLoop
├── data/
│   ├── dictionary.py    ← dictionary loader/interface
│   └── wordlists/       ← TWL, SOWPODS etc as .txt files
├── tests/
│   ├── test_grid.py
│   ├── test_scorer.py
│   ├── test_bunch.py
│   └── test_solver.py
└── main.py              ← entry point, game loop
```

## Build Order
1. `bunch.py` — simplest, no dependencies, establishes tile set
2. `dictionary.py` — loads wordlist into trie/DAWG structure
3. `grid.py` — depends on dictionary for validation
4. `scorer.py` — depends on grid and dictionary
5. `solver.py` — depends on everything

---

## Core Architecture

### BunchTracker (`bunch.py`)
Tracks the full tile set and what has been drawn, placed, or dumped.

```
bunch = full set - hand - placed tiles - dumped tiles
```

Always deterministic in solo mode — full tile set is known at game start. Designed as a standalone object so it can be swapped for a probabilistic implementation in multiplayer without touching other modules.

**Bananagrams tile distribution (144 total):**
```
A(13), B(3), C(3), D(6), E(18), F(3), G(4), H(3), I(12),
J(2), K(2), L(5), M(3), N(8), O(11), P(3), Q(2), R(9),
S(6), T(9), U(6), V(3), W(3), X(2), Y(3), Z(2)
```

### Dictionary (`dictionary.py`)
Loads wordlist into a DAWG (Directed Acyclic Word Graph) or trie for fast prefix lookup. Supports swappable dictionaries — TWL for Scrabble-standard, SOWPODS for international.

### GridValidator (`grid.py`)
**The grid is a pure 2D array:**
```
(row, col) → letter | empty
```
No word objects, no ownership, no terminal tile flags. Letters and positions only.

**Validation runs after every placement:**
1. Find all affected rows and columns from the placement
2. Scan each for contiguous runs (bounded by empty cells or board edge)
3. Every contiguous run of 2+ letters must be a valid dictionary word
4. Single isolated letters (no neighbors in either direction) are legal placeholders
5. No concept of word identity — only "is this contiguous sequence valid?"

**This approach cleanly handles:**
- Word extension (TRICK → TRICKSTER): finds full run, validates once
- Illegal extension (APPLE + EXTRA = APPLEXTRA): finds full run, fails validation
- Incidental 2-letter words: caught in perpendicular column scan
- Parallel adjacency: caught because every column of every new letter is scanned
- Bridging two existing runs: merged into one longer run, validated as unit

**Placement rules:**
- New word must share a tile with an existing word via a matching letter
- Shared tile letter must be valid in its position in both words simultaneously
- No parallel adjacency unless all incidental perpendicular runs are valid
- After any placement, scan ALL affected rows and columns — not just the new word

### PlacementScorer (`scorer.py`)
Scores candidate placements using:

**Anchor word scoring (first word):**
```
Score = sum of Scrabble tile values of letters used
```
High Scrabble value = rare/awkward letter = prioritize burning early.
Tiebreaker: word that uses more tiles wins.

**Subsequent placement scoring:**
```
Score = Scrabble value of letters used
      + flexibility score of remaining hand
      - isolation penalty of remaining hand
```
Flexibility = how many valid words can be formed from remaining hand letters after placement. A placement that scores well but leaves an inflexible hand (e.g. G, T, T) is penalized.

**Score-validate-commit loop:**
```
1. Score word candidate
2. Tentatively place on grid
3. Run GridValidator on full board
4. If invalid → discard, try next candidate
5. If valid → commit, score next word
```
Placements are never scored in isolation — grid validity is always checked before committing.

**Board shape preference:**
Parallel vertical words off an anchor create too many incidental horizontal word constraints. L-shaped and Z-shaped builds are preferred as they reduce incidental conflict surface.

### Restructure Scoring
When a new letter cannot place cleanly, the solver evaluates minimum-cost restructuring:

- Board modeled as a graph: words are nodes, intersections are edges
- **Articulation points** (via Tarjan's algorithm) are heavily penalized — removing them disconnects the graph
- Restructure cost = stranded subtree size + connection count of moved word
- Lowest cost restructure is recommended

**Key principle:** A word with one connection is not automatically cheap to move — if it is an articulation point, removing it strands everything beyond it.

### SolverLoop (`solver.py`)
Greedy, immediate-move optimization. On every state change:

```
1. Update bunch
2. Score all valid placements for current hand
3. Score all valid dumps
4. Compare best placement vs best dump
5. Recommend highest scoring immediate action
```

**DUMP evaluation:**
```
Dump candidate = lowest placement value tile in hand
Post-dump hand = current hand - 1 dumped tile + 3 known tiles from bunch
If post-dump best placement > current best placement → recommend dump
```

**PEEL vs DUMP decision:**
```
If stranded tiles = 1:
    Evaluate peel first — new tile may combine with stranded tile
If stranded tiles = 2+:
    Evaluate dump — fewer stranded tiles faster is better
If stranded tiles share no valid combinations:
    Dump lowest value tile immediately
```

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Grid as pure 2D letter array | Simpler, catches all incidental word conflicts naturally |
| No word ownership on tiles | Word identity emerges from validation, not placement |
| Scrabble values for anchor scoring | High value = rare = burn early, already a solved problem |
| Leftover hand flexibility in scoring | A word using 5 tiles but leaving G,T,T is worse than 4 tiles leaving E,T,T |
| Score-validate-commit loop | Individual word scores don't predict combined board validity |
| Tarjan's for restructure | O(V+E) articulation point detection, fast even on full board |
| Greedy immediate optimization | Simple, fast, correct for solo. Lookahead added in future iteration |
| BunchTracker as isolated object | Swap deterministic → probabilistic for multiplayer without touching other modules |

---

## Lessons from Algorithm Walkthrough
These emerged from manually walking through a full 21-tile game:

1. **Parallel vertical words off an anchor create too many incidental row constraints** — prefer L/Z shaped builds
2. **Placement order affects validity** — word A valid + word B valid ≠ A and B valid together
3. **Leftover hand flexibility must score into placement** — not just tiles placed count
4. **Single stranded tile = peel, not dump** — new tile likely combines with it
5. **Incidental words are caught at the row/column scan level** — not at placement logic level

---

## Multiplayer Extension Notes (v2)
Solo is built first. Multiplayer adds a new BunchTracker implementation only — all other modules unchanged.

**Additional v2 variables:**
```
Bunch:
- known_remaining      ← solo (deterministic)
- probable_remaining   ← multiplayer (estimated)
- drawn_by_others      ← multiplayer (count only, not identity)
```

**Observable in multiplayer:**
- Peel counts per opponent (everyone draws together)
- Dump counts per opponent (visible action)
- Net hand size estimates per opponent

**Not observable:**
- Which letters opponents hold
- Which specific letter was dumped back
- What 3 tiles came back from an opponent's dump

Dump letter treated as unknown — tracking it is possible but exceeds real-game bandwidth.

```
probable_remaining = known_remaining 
                   - (peel_count × 1 per opponent) 
                   - (dump_count × net 2 per opponent)
```

---

## Bananagrams Rules Reference
- 144 tiles total
- Solo starting hand: 21 tiles
- **PEEL:** Complete your board → draw 1 tile from bunch
- **DUMP:** Return 1 tile → draw 3 from bunch (can do anytime)
- **Win:** All tiles placed when bunch is empty
- Every contiguous letter sequence on the board (horizontal and vertical) must be a valid word
- 2-letter words are valid (per Scrabble dictionary)
- Single isolated letters are legal placeholders
