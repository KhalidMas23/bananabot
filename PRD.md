# Product Requirements Document
# Bananagrams Solver — v1.0

---

## 1. Overview

### 1.1 Product Summary
A real-time Bananagrams solver that uses a laptop camera to read the physical game board and player hand, then overlays optimal play decisions on a live digital grid display. The solver tells the player what to place, where to place it, or when to dump — instantly, as the game evolves.

### 1.2 Problem Statement
Bananagrams requires simultaneous spatial reasoning, vocabulary recall, and strategic tile management under time pressure. The solver acts as a real-time co-pilot — reading the physical board via camera, tracking every tile, and surfacing the optimal next action so the player can focus on execution.

### 1.3 Target Users
- Primary: Solo players practicing or learning strategy
- Secondary: Word game enthusiasts and developers interested in the technical implementation ("nerds")

### 1.4 Version Scope
- **v1.0:** Solo play, full tile set known upfront, greedy immediate-move optimization, Python logic + OpenCV vision
- **v2.0:** Multiplayer extension (see Section 8)

---

## 2. Success Metrics

### 2.1 Primary Metric
**Near-instant decision output** — whether a DUMP action or a full board state recommendation, the solver must respond fast enough to feel real-time during active play. Target: sub-500ms from tile state change to decision display.

### 2.2 Secondary Metrics
- Valid board states only — solver never recommends an illegal placement
- Calibration completes in under 2 minutes on first launch
- Tile tracking maintains continuity across occlusion events without ID reassignment
- Dictionary lookup does not bottleneck solver response time

---

## 3. Core Features

### 3.1 Camera Vision System
The primary input mechanism. No manual tile entry required.

**Camera setup:**
- Single static camera — laptop camera or equivalent
- Fixed perspective, straight down or shallow fixed angle
- One-time calibration on first launch, saved to config
- Calibration uses a printed reference sheet to correct perspective distortion

**Zone definition (set during calibration):**
```
Zone 1: Board area — placed tiles, active grid
Zone 2: Hand area — unplaced tiles available to player
```
Player defines both zone boundaries by dragging corners on screen during calibration. Zones saved to config, never touched again unless camera moves.

**Tile tracking:**
Every face-up tile is assigned a unique ID the moment it enters the camera FOV:
```
tile_001: A — first seen hand zone, frame 142
tile_002: E — first seen board zone, frame 156
```

State transitions driven purely by vision:
```
Enters FOV              → assign unique ID, classify to zone
Hand zone → board zone  → tile placed, update grid position  
Board zone → hand zone  → tile picked up, update hand
Leaves FOV (no hand)    → confirmed dumped, removed from bunch
Leaves FOV (with hand)  → confirmed dumped, removed from bunch
Occluded by hand        → state frozen, tile presumed present
New tile enters FOV     → assign new unique ID
```

**Two tracker types run simultaneously:**
```
Tile trackers  — follow individual letter tiles by unique ID
Hand tracker   — follows player's hand/fingers
```

**Occlusion handling:**
A tile that disappears while the hand tracker is present is treated as occluded, not dumped. Tile state is frozen until it reappears or the hand leaves without it (confirmed dump). A tile that disappears with no hand present is immediately marked dumped.

**Partial visibility:**
Tiles partially in frame are treated as present. Face-down bunch tiles are always distinguishable from face-up played tiles — no ambiguity at frame boundaries.

**Bunch calculation:**
```
bunch = all_ever_seen_tiles - currently_visible_tiles
```
Automatically maintained. No manual dump logging required.

### 3.2 Letter Recognition (OCR)
- Reads face-up tiles only within defined zones
- Identifies letter on each detected tile
- Feeds letter identity + zone position to solver
- Library: OpenCV + Tesseract or equivalent

### 3.3 Grid Validator
The core of the algorithm. Stateless, pure function.

**Grid model:**
```
(row, col) → letter | empty
```
No word objects, no ownership, no terminal tile flags. Letters and positions only.

**Validation rules:**
1. Find all affected rows and columns from a placement
2. Scan each for contiguous runs (bounded by empty cells or board edge)
3. Every contiguous run of 2+ letters must be a valid dictionary word
4. Single isolated letters (no neighbors in either direction) are legal placeholders
5. Every affected row AND column is scanned — not just the new word's axis

**Handles correctly:**
- Word extension (TRICK → TRICKSTER): full run found, validated once
- Illegal extension (APPLE + EXTRA = APPLEXTRA): full run found, fails
- Incidental 2-letter words: caught in perpendicular scan
- Parallel adjacency conflicts: caught in column-by-column scan
- Bridging two runs: merged into one, validated as unit

**Score-validate-commit loop:**
```
1. Score candidate placement
2. Tentatively place on grid
3. Run full GridValidator
4. If invalid → discard, try next candidate
5. If valid → commit
```
Placements are never accepted without full board validation.

### 3.4 Placement Scorer

**Anchor word (first word) scoring:**
```
Score = sum of Scrabble tile values of letters used
```
High Scrabble value = rare/inflexible letter = burn early.
Tiebreaker: word using more tiles wins.

**Scrabble tile values:**
```
A=1, B=3, C=3, D=2, E=1, F=4, G=2, H=4, I=1, J=8, K=5,
L=1, M=3, N=1, O=1, P=3, Q=10, R=1, S=1, T=1, U=1,
V=4, W=4, X=8, Y=4, Z=10
```

**Subsequent placement scoring:**
```
Score = Scrabble value of letters used
      + flexibility score of remaining hand
      - isolation penalty of remaining hand
```
Flexibility = number of valid words constructable from remaining hand after placement. A placement leaving an inflexible hand (e.g. G, T, T) is penalized relative to one leaving a flexible hand (e.g. E, R, T).

**Board shape preference:**
Parallel vertical words off an anchor create too many incidental horizontal constraints. L-shaped and Z-shaped board builds are preferred and scored higher.

### 3.5 Restructure Scorer
When a new tile cannot place cleanly, the solver evaluates minimum-cost board restructuring.

**Model:**
- Board represented as a graph: words are nodes, intersections are edges
- Articulation points detected via Tarjan's algorithm (O(V+E))
- Articulation point = word whose removal disconnects the graph → heavily penalized

**Restructure cost:**
```
Cost = stranded subtree size + connection count of candidate word
```
Lowest cost restructure surfaced as recommendation. Articulation points are never recommended for removal unless no other option exists.

### 3.6 Bunch Tracker
Deterministic in v1 — full tile set always known.

```
bunch = full set - hand - placed tiles - dumped tiles
```

**Bananagrams tile distribution (144 total):**
```
A(13), B(3), C(3), D(6), E(18), F(3), G(4), H(3), I(12),
J(2), K(2), L(5), M(3), N(8), O(11), P(3), Q(2), R(9),
S(6), T(9), U(6), V(3), W(3), X(2), Y(3), Z(2)
```

Designed as an isolated object — swappable for probabilistic implementation in v2 without touching other modules.

### 3.7 Solver Loop
Greedy, immediate-move optimization. Runs on every detected state change.

```
On every state change:
1. Update bunch
2. Score all valid placements for current hand
3. Score all valid dumps
4. Compare best placement vs best dump
5. Surface highest scoring immediate action
```

**DUMP evaluation:**
```
Dump candidate  = lowest placement value tile in hand
Post-dump hand  = current hand - 1 dumped + 3 known tiles from bunch
Recommend dump if post-dump best placement > current best placement
```

**PEEL vs DUMP decision:**
```
Stranded tiles = 1    → evaluate peel first (new tile likely combines)
Stranded tiles = 2+   → evaluate dump
No valid combinations → dump lowest value tile immediately
```

**Error state:**
If no valid placement exists and no dump is possible (bunch empty), the game is over. Solver surfaces game over state — no further recommendations.

### 3.8 Display
- Live camera feed shown alongside a separate digital grid representation
- Digital grid updates snap to new state (no animation)
- Solver recommendation overlaid as text on the display
- Reasoning text available as an expandable/toggleable panel for users who want to understand the decision

**Reasoning output example:**
```
Action: DUMP T
Reason: T has no valid placements with current hand or board. 
        Post-dump hand (G, R, E) scores 3 valid placements 
        vs 0 for current hand. Dumping gains net +2 tiles.
```

```
Action: Place NEURAL at row 3
Reason: Burns 5 tiles (E,U,R,A,L), scores 6pts. 
        Remaining hand (G,T,T) has dump fallback. 
        No incidental word conflicts detected.
```

### 3.9 Dictionary
- **v1.0:** TWL (Tournament Word List / Scrabble standard) only
- Loaded into DAWG (Directed Acyclic Word Graph) or trie for fast prefix lookup
- Dictionary does not bottleneck solver response time
- **Future:** Selectable dictionary (SOWPODS, Oxford, etc.) as a settings option

---

## 4. Technical Architecture

```
bananagrams/
├── core/
│   ├── grid.py          ← GridValidator
│   ├── scorer.py        ← PlacementScorer + RestructureScorer
│   ├── bunch.py         ← BunchTracker
│   └── solver.py        ← SolverLoop
├── vision/
│   ├── camera.py        ← feed capture + calibration
│   ├── tracker.py       ← tile IDs + hand tracker
│   ├── ocr.py           ← letter reading
│   └── mapper.py        ← vision state → grid + hand + bunch
├── overlay/
│   ├── renderer.py      ← digital grid display
│   └── display.py       ← decision text + reasoning panel
├── data/
│   ├── dictionary.py    ← DAWG/trie loader
│   └── wordlists/       ← TWL.txt, future dictionaries
├── tests/
│   ├── test_grid.py
│   ├── test_scorer.py
│   ├── test_bunch.py
│   ├── test_solver.py
│   └── test_vision.py
└── main.py              ← entry point, game loop
```

**Build order:**
1. `bunch.py`
2. `dictionary.py`
3. `grid.py`
4. `scorer.py`
5. `solver.py`
6. `vision/` modules
7. `overlay/` modules
8. `main.py` integration

**Key libraries:**
- OpenCV — camera capture, perspective correction, tile detection, hand tracking
- Tesseract (via pytesseract) — OCR for letter recognition
- Python standard library — core logic, no heavy dependencies in core/

---

## 5. User Flow

### 5.1 First Launch — Calibration
```
1. App launches, camera feed displayed
2. User prompted to place calibration sheet in frame
3. Perspective correction applied and saved
4. User drags to define board zone boundary
5. User drags to define hand zone boundary
6. Calibration saved to config
7. Ready to play
```

### 5.2 Game Flow
```
1. Player draws 21 tiles, places hand tiles in hand zone
2. Vision system detects and IDs all tiles, reads letters
3. Solver computes optimal first word (anchor)
4. Digital grid displays recommended board state
5. Reasoning panel shows why
6. Player executes recommendation on physical board
7. Camera detects placement, updates grid state
8. Player draws next tile (PEEL), places in hand zone
9. Solver recomputes, surfaces next recommendation
10. Repeat until all tiles placed (win) or no moves possible (game over)
```

### 5.3 DUMP Flow
```
1. Solver recommends DUMP with specific tile identified
2. Player picks up recommended tile from hand zone
3. Tile tracker follows tile out of FOV → confirmed dumped
4. Bunch updates automatically
5. Player draws 3 tiles, places in hand zone
6. Vision assigns new IDs to incoming tiles
7. Solver recomputes with new hand
```

---

## 6. Out of Scope (v1.0)
- Multiplayer
- Probabilistic bunch tracking
- Lookahead / multi-move optimization
- Animated board transitions
- Mobile camera support
- Cloud dictionary APIs
- User accounts or game history
- AR overlay on physical board

---

## 7. Open Questions / TBDs
- Exact performance threshold (currently: "near instant", target 500ms)
- OCR library final selection (Tesseract vs alternatives)
- Calibration sheet design (printed pattern vs dynamic on-screen guide)
- Reasoning panel — always visible or toggle?
- Hand tracker implementation — color-based, contour-based, or ML model?

---

## 8. v2.0 Multiplayer Extension
All v1 core modules unchanged. Multiplayer adds a new BunchTracker implementation only.

**Additional variables:**
```
probable_remaining = known_remaining
                   - (peel_count × 1 per opponent)
                   - (dump_count × net 2 per opponent)
```

**Observable in multiplayer:**
- Peel counts per opponent (everyone draws simultaneously)
- Dump counts per opponent (visible action)
- Net hand size estimates per opponent

**Not observable:**
- Which specific letters opponents hold
- Which letter an opponent dumped
- What 3 tiles came back from opponent dump

Opponent dump letter treated as unknown — tracking it exceeds real-game cognitive bandwidth and is not worth modeling.
