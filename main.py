"""
BananaBot entry point: calibration, module wiring, and the live display loop.

Business logic stays in core/, vision/, and overlay/; this file only integrates.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import cv2

from data.dictionary import Dictionary
from core.bunch import BunchTracker
from core.grid import GridValidator
from core.scorer import PlacementScorer
from core.solver import STARTING_RACK_TILES, SolverLoop
from vision.camera import CameraFeed
from vision.opencv_util import (
    HIGHGUI_UNAVAILABLE_HINT,
    cv2_highgui_available,
    safe_destroy_all_windows,
    safe_wait_key,
)
from vision.tracker import VisionTracker
from vision.ocr import OCRReader
from vision.mapper import StateMapper
from overlay.display import DisplayManager

# Virtual board resolution inside the calibrated board quadrilateral (mapper / overlay).
BOARD_GRID_ROWS = 30
BOARD_GRID_COLS = 30


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _calibration_json_ok(path: Path) -> bool:
    """Return True if file matches the shape expected by ``CameraFeed`` / ``CameraManager``."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        matrix = data.get("perspective_matrix")
        if not isinstance(matrix, list) or len(matrix) != 3:
            return False
        if any(not isinstance(row, list) or len(row) != 3 for row in matrix):
            return False
        for row in matrix:
            for v in row:
                float(v)
        ws = data.get("warp_shape")
        if not isinstance(ws, list) or len(ws) != 2:
            return False
        int(ws[0])
        int(ws[1])
        bz = data.get("board_zone")
        hz = data.get("hand_zone")
        if not isinstance(bz, list) or len(bz) != 4:
            return False
        if not isinstance(hz, list) or len(hz) != 4:
            return False
        for p in bz + hz:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return False
            float(p[0])
            float(p[1])
        if "camera_index" in data:
            int(data["camera_index"])
        return True
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _prepare_calibration_file(project_root: Path) -> Path:
    """Ensure ``config/`` exists; drop corrupt ``calibration.json`` so a fresh run can calibrate."""
    config_dir = project_root / "config"
    os.makedirs(config_dir, exist_ok=True)
    cal_path = config_dir / "calibration.json"
    if cal_path.is_file() and not _calibration_json_ok(cal_path):
        try:
            cal_path.unlink()
        except OSError:
            print(
                f"Invalid calibration.json could not be removed: {cal_path}",
                file=sys.stderr,
            )
    return cal_path


def _mapper_calibration(camera: CameraFeed, rows: int, cols: int) -> dict:
    """Build ``StateMapper`` calibration; bridges board quad to axis-aligned bounds + grid size."""
    bz = camera.board_zone
    if len(bz) != 4:
        raise ValueError("Board zone must have four points after calibration.")
    xs = [float(p[0]) for p in bz]
    ys = [float(p[1]) for p in bz]
    return {
        "board_zone_bounds": {
            "top_left": (min(xs), min(ys)),
            "bottom_right": (max(xs), max(ys)),
        },
        "grid_rows": rows,
        "grid_cols": cols,
    }


def _open_camera(project_root: Path) -> CameraFeed:
    cal_path = _prepare_calibration_file(project_root)
    # Full calibration every launch: camera mount / angle can change between sessions.
    return CameraFeed(
        config_path=cal_path,
        project_root=project_root,
        force_recalibrate=True,
    )


def _assert_camera_frame(camera: CameraFeed) -> None:
    for _ in range(45):
        raw, warped = camera.read_raw_and_warped()
        if raw is not None and warped is not None:
            return
        safe_wait_key(50)
    print("Camera not found — check connection", file=sys.stderr)
    raise SystemExit(1)


def _build_state_mapper(
    tracker: VisionTracker,
    bunch_tracker: BunchTracker,
    solver: SolverLoop,
    camera: CameraFeed,
) -> StateMapper:
    cal = _mapper_calibration(camera, BOARD_GRID_ROWS, BOARD_GRID_COLS)
    return StateMapper(tracker, bunch_tracker, solver, cal)


def main() -> None:
    project_root = _project_root()
    camera: CameraFeed | None = None

    try:
        twl_path = project_root / "data" / "wordlists" / "TWL.txt"
        if not twl_path.is_file():
            print("TWL.txt not found in data/wordlists/", file=sys.stderr)
            raise SystemExit(1)

        if not cv2_highgui_available():
            print(HIGHGUI_UNAVAILABLE_HINT, file=sys.stderr)
            raise SystemExit(1)

        dictionary = Dictionary(str(twl_path))
        bunch_tracker = BunchTracker()
        grid_validator = GridValidator(dictionary)
        placement_scorer = PlacementScorer(dictionary)
        solver_loop = SolverLoop(
            bunch_tracker, grid_validator, placement_scorer, dictionary
        )

        camera = _open_camera(project_root)
        _assert_camera_frame(camera)

        vision_tracker = VisionTracker(camera)
        ocr_reader = OCRReader()
        mapper = _build_state_mapper(
            vision_tracker, bunch_tracker, solver_loop, camera
        )
        display = DisplayManager()

        game_over_shown = False
        game_active = False

        while True:
            raw, warped = camera.read_raw_and_warped()
            if raw is None or warped is None:
                key = safe_wait_key(1)
                if key in (ord("c"), ord("C")):
                    try:
                        camera.run_calibration()
                        mapper = _build_state_mapper(
                            vision_tracker, bunch_tracker, solver_loop, camera
                        )
                        game_over_shown = False
                        game_active = False
                    except Exception:
                        traceback.print_exc(file=sys.stderr)
                should_quit = display.run_key_handler(key)
                if should_quit:
                    break
                continue

            panel = camera.draw_zones_on_source(raw)
            warp_mx = camera.warp_to_source_matrix

            vision_tracker.update(warped)
            tracker_state = {
                t.tile_id: t for t in vision_tracker.get_active_tiles()
            }
            ocr_output = ocr_reader.read(warped, vision_tracker)
            mapped, recommendation = mapper.update(
                tracker_state, ocr_output, game_active=game_active
            )
            grid_state = mapped.grid

            if recommendation is not None:
                display.update(
                    panel,
                    grid_state,
                    recommendation,
                    tracker_state,
                    warp_to_source_3x3=warp_mx,
                )
                if (
                    recommendation.action == "GAME_OVER"
                    and not game_over_shown
                ):
                    display.set_game_over(recommendation, grid_state)
                    game_over_shown = True

            key = safe_wait_key(1)
            if key in (ord("g"), ord("G")):
                if len(mapped.hand) == STARTING_RACK_TILES:
                    game_active = True
                    game_over_shown = False
            if key in (ord("c"), ord("C")):
                try:
                    camera.run_calibration()
                    mapper = _build_state_mapper(
                        vision_tracker, bunch_tracker, solver_loop, camera
                    )
                    game_over_shown = False
                    game_active = False
                except Exception:
                    traceback.print_exc(file=sys.stderr)
            should_quit = display.run_key_handler(key)
            if should_quit:
                break

    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
    finally:
        if camera is not None:
            camera.release()
        safe_destroy_all_windows()


if __name__ == "__main__":
    main()
