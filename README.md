# 🍌 BananaBot

A real-time Bananagrams solver that watches your game through a camera and tells you exactly what to play.

BananaBot uses computer vision to read your tiles, tracks every piece on the board, and surfaces the optimal move — whether that's placing a word, restructuring the board, or dumping a tile — instantly, as the game evolves.

---

## Demo

> 📸 *Screenshot / GIF coming soon*

---

## How It Works

BananaBot runs two things simultaneously:

**Vision pipeline**
Your laptop camera watches two zones you define at setup — the board and your hand. Every tile is assigned a unique tracker the moment it enters frame. Letters are read via OCR. Tiles that leave the frame are automatically marked as dumped and subtracted from the bunch. Occlusion (your hand covering a tile) is handled gracefully — the tile isn't lost, just frozen until it reappears.

**Solver**
The board is modeled as a pure 2D grid — no word ownership, no terminal flags, just letters and positions. After every tile change, the solver scans every affected row and column for contiguous letter runs and validates each against the dictionary. It then scores all valid placements using Scrabble tile values (rarer letters burned first), evaluates whether a DUMP would unlock better options, and surfaces the highest-value immediate action with a plain-English explanation of why.

---

## Features

- 🎥 **Camera input** — no typing, no manual entry, just play
- 🔤 **Live tile tracking** — unique ID per tile, survives occlusion
- 🧠 **Real-time solver** — greedy immediate-move optimization
- ♻️ **DUMP evaluation** — knows when to cut losses and redraw
- 🗺️ **Board restructure scoring** — recommends minimum-cost reorganizations using graph theory (Tarjan's algorithm for articulation points)
- 📖 **Reasoning panel** — explains every decision for those who want to learn
- 📋 **Dictionary support** — TWL (Scrabble standard) out of the box, more coming
- 🧪 **Fully tested** — unit + integration test suite across all modules

---

## Architecture

```
bananagrams/
├── core/
│   ├── grid.py          ← GridValidator (pure 2D array, contiguous run validation)
│   ├── scorer.py        ← PlacementScorer + RestructureScorer (Tarjan's)
│   ├── bunch.py         ← BunchTracker (deterministic tile set management)
│   └── solver.py        ← SolverLoop (greedy immediate-move)
├── vision/
│   ├── camera.py        ← feed capture + one-time calibration
│   ├── tracker.py       ← tile + hand tracking
│   ├── ocr.py           ← letter recognition (pytesseract)
│   └── mapper.py        ← vision state → game state
├── overlay/
│   ├── renderer.py      ← digital grid display
│   └── display.py       ← recommendation + reasoning panel
├── data/
│   ├── dictionary.py    ← DAWG/trie loader
│   └── wordlists/       ← TWL.txt
├── tests/
└── main.py
```

---

## Setup

### Requirements
- Python 3.10+
- Laptop or external camera
- Physical Bananagrams tile set
- A printed calibration sheet (see `/docs/calibration_sheet.pdf`)

### Install

```bash
git clone https://github.com/yourusername/bananabot.git
cd bananabot
pip install -r requirements.txt
```

### First Launch — Calibration

On first run, BananaBot will walk you through a one-time calibration:

1. Place the calibration sheet flat in frame
2. BananaBot corrects for your camera's perspective angle
3. Drag to define your **board zone** and **hand zone**
4. Config is saved — never do this again unless your camera moves

```bash
python main.py
```

---

## Playing

1. Draw your 21 starting tiles, lay them flat in the **hand zone**
2. BananaBot reads your letters and computes the optimal first word
3. Place the word on the board in the **board zone**
4. Draw the next tile, place it in the hand zone
5. BananaBot updates in real time — follow its recommendations
6. Toggle the reasoning panel any time to see why it made its call

**Tile tracking is fully automatic:**
- Place a tile on the board → tracked as placed
- Pick a tile back up → tracked as in hand
- Dump a tile → tracked as dumped, bunch updates instantly
- New tiles drawn → new trackers assigned automatically

---

## Solver Logic

### Anchor Word (First Play)
Scored by sum of Scrabble tile values of letters used — rare and awkward letters (Q, X, Z, H, M, P) are burned first. Ties broken by number of tiles placed.

### Subsequent Placements
```
Score = Scrabble value of tiles used
      + flexibility of remaining hand
      - isolation penalty of remaining hand
```
A placement that scores well but leaves you with G, T, T is penalized vs one that leaves E, R, T.

### DUMP Decision
```
Dump candidate  = lowest placement-value tile in hand
Post-dump hand  = current hand - 1 + 3 known tiles from bunch
Recommend dump  = post-dump best placement > current best placement
```
Since it's solo play, the full tile set is known upfront — BananaBot knows exactly what 3 tiles you'd draw from a dump.

### Restructure
When a new tile can't place cleanly, BananaBot evaluates minimum-cost board reorganization:
- Board modeled as a graph (words = nodes, intersections = edges)
- Articulation points detected via Tarjan's algorithm
- Restructure cost = stranded subtree size + word connection count
- Safest restructure always recommended first

### Validation
Every placement is validated before committing. The grid is scanned row by row and column by column — every contiguous letter run must be a valid dictionary word. This catches illegal extensions (APPLEXTRA), incidental words from parallel placement, and merged runs from bridging two existing words.

---

## Roadmap

- [x] Solo play — full tile set known upfront
- [x] Greedy immediate-move solver
- [x] Camera vision + tile tracking
- [ ] Selectable dictionaries (SOWPODS, Oxford)
- [ ] Multiplayer mode — probabilistic bunch tracking as opponents draw and dump
- [ ] Mobile camera support
- [ ] Visual board history / replay

---

## Contributing

PRs welcome. If you're picking up a feature, check the open issues first and reference `CONTEXT.md` and `PRD.md` in the repo root — they explain the design decisions and why things are built the way they are.

```bash
# Run tests
python -m pytest tests/
```

Please keep core modules (`grid.py`, `scorer.py`, `bunch.py`, `solver.py`) free of vision and display logic. Separation of concerns is load-bearing here.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgements

- [TWL06](https://www.wordgamedictionary.com/twl06/) — Tournament Word List
- [OpenCV](https://opencv.org/) — computer vision
- [pytesseract](https://github.com/madmaze/pytesseract) — OCR
- Bananagrams — the game that started this

---

*Built because losing at Bananagrams is unacceptable.*
