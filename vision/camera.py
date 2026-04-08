"""
Live camera capture with perspective correction and board/hand zone geometry.

The calibration **sheet** fixes the table **plane and camera angle** only. The
rectified image is sized to include the **whole camera frame** projected onto
that plane (not just the paper), so board/hand zones can cover the real play
surface. ``read_warped_frame`` / ``read_raw_and_warped`` feed detection and OCR;
the raw image from ``read_raw_and_warped`` is full FOV with overlays via
``warp_to_source_matrix``.

Interactive calibration needs OpenCV with GUI (``opencv-python``). Headless builds
are detected early; use :func:`vision.opencv_util.cv2_highgui_available` or a saved
``calibration.json`` from another machine.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision.opencv_util import (
    HIGHGUI_UNAVAILABLE_HINT,
    cv2_highgui_available,
    ensure_opencv_cv2_runtime,
    safe_destroy_all_windows,
    safe_destroy_window,
)

ensure_opencv_cv2_runtime()

# Window titles used during calibration — destroyed via release() or on interrupt.
_WIN_PERSPECTIVE = "Bananagrams calibration — perspective"
_WIN_BOARD_ZONE = "Bananagrams calibration — board zone"
_WIN_HAND_ZONE = "Bananagrams calibration — hand zone"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _order_quad_document(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_aspect_ratio(pts: np.ndarray) -> float:
    """Width/height of axis-aligned bounding box of the quad."""
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    if h < 1e-6:
        return 999.0
    return float(w) / float(h)


def _find_largest_sheet_quad(
    frame: np.ndarray,
    min_area_ratio: float = 0.10,
    min_aspect: float = 0.25,
    max_aspect: float = 4.0,
) -> np.ndarray | None:
    """
    Find the largest plausible quadrilateral (plain A4/Letter paper) in frame.

    Expects a light sheet on a darker playing surface so the paper edge contrasts
    with the background. Same contour pipeline as before: Canny edges, largest
    convex 4-gon passing area and aspect-ratio checks.
    Returns 4x2 float32 in TL, TR, BR, BL order, or None if confidence is low.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    best: np.ndarray | None = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * frame_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        ar = _quad_aspect_ratio(pts)
        if ar < min_aspect or ar > max_aspect:
            continue
        if area > best_area:
            best_area = area
            best = _order_quad_document(pts)

    return best


def _warp_output_size(src_quad: np.ndarray) -> tuple[int, int]:
    """Rectangle size (width, height) from edge lengths of the source quad."""
    (tl, tr, br, bl) = src_quad
    width_a = float(np.linalg.norm(tr - tl))
    width_b = float(np.linalg.norm(br - bl))
    max_width = int(max(width_a, width_b))
    height_a = float(np.linalg.norm(bl - tl))
    height_b = float(np.linalg.norm(br - tr))
    max_height = int(max(height_a, height_b))
    max_width = max(max_width, 1)
    max_height = max(max_height, 1)
    return max_width, max_height


def _perspective_from_quad(src_quad: np.ndarray) -> tuple[np.ndarray, int, int]:
    w, h = _warp_output_size(src_quad)
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(src_quad, dst)
    return m, w, h


def _expand_warp_to_table_plane(
    m_paper: np.ndarray,
    paper_w: int,
    paper_h: int,
    frame_w: int,
    frame_h: int,
    *,
    margin: float = 8.0,
) -> tuple[np.ndarray, int, int]:
    """
    Grow the rectified output so it covers the **entire camera frame** projected
    onto the same plane as the calibration sheet — not just the sheet patch.

    ``m_paper`` maps camera pixels into coordinates where the sheet occupies
    approximately [0, paper_w-1] × [0, paper_h-1]. Returns ``T @ m_paper`` and
    output size ``(big_w, big_h)`` so :func:`cv2.warpPerspective` shows the full
    table (or desk) visible in frame, while the sheet was only used to infer the
    plane / angle.
    """
    if frame_w < 2 or frame_h < 2 or paper_w < 1 or paper_h < 1:
        return m_paper.astype(np.float64), paper_w, paper_h

    corners_src = np.array(
        [
            [
                [0.0, 0.0],
                [float(frame_w - 1), 0.0],
                [float(frame_w - 1), float(frame_h - 1)],
                [0.0, float(frame_h - 1)],
            ]
        ],
        dtype=np.float32,
    )
    corners_dst = cv2.perspectiveTransform(corners_src, m_paper)
    xs = corners_dst[0, :, 0].astype(np.float64).flatten().tolist()
    ys = corners_dst[0, :, 1].astype(np.float64).flatten().tolist()
    xs.extend([0.0, float(paper_w - 1), float(paper_w - 1), 0.0])
    ys.extend([0.0, 0.0, float(paper_h - 1), float(paper_h - 1)])

    min_x = np.floor(min(xs) - margin)
    min_y = np.floor(min(ys) - margin)
    max_x = np.ceil(max(xs) + margin)
    max_y = np.ceil(max(ys) + margin)

    big_w = max(1, int(max_x - min_x))
    big_h = max(1, int(max_y - min_y))

    t = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    m_out = t @ m_paper.astype(np.float64)
    return m_out, big_w, big_h


def _open_capture(preferred_indices: list[int]) -> tuple[cv2.VideoCapture | None, int]:
    """Try indices in order; return (capture, index) or (None, -1)."""
    tried: set[int] = set()
    for idx in preferred_indices:
        if idx in tried:
            continue
        tried.add(idx)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if not ok:
            cap.release()
            continue
        return cap, idx
    return None, -1


def _zone_mouse_callback_factory(
    corners: list[list[float]], drag_radius: float = 24.0
) -> Any:
    state: dict[str, Any] = {"dragging": None}

    def on_event(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            best_i = None
            best_d2 = drag_radius**2
            for i, c in enumerate(corners):
                d2 = (c[0] - x) ** 2 + (c[1] - y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            state["dragging"] = best_i
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
            i = state["dragging"]
            corners[i][0] = float(x)
            corners[i][1] = float(y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = None

    return on_event


def _draw_perspective_calibration_overlay(
    hint: np.ndarray, *, detection_failed: bool
) -> None:
    """Multi-line instructions for perspective calibration (mutates ``hint``)."""
    if detection_failed:
        lines = [
            "Sheet not detected. Check that the paper is on a dark surface",
            "and fully in frame, then press SPACE to try again.",
        ]
    else:
        lines = [
            "Place A4/Letter on the TABLE — sheet is only for camera angle/plane.",
            "Next steps: board + hand zones can cover the full rectified table view.",
            "Dark surface under the sheet. Press SPACE to capture (ESC to cancel).",
        ]
    y = 24
    for line in lines:
        cv2.putText(
            hint,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 26


def _calibration_window_closed(window_name: str) -> bool:
    """True if the OpenCV window was destroyed (e.g. user closed it)."""
    try:
        return float(cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)) < 0
    except cv2.error:
        return True


def _draw_zone_ui(
    frame: np.ndarray,
    corners: list[list[float]],
    title: str,
    instruction: str,
) -> np.ndarray:
    vis = frame.copy()
    pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    for c in corners:
        cx, cy = int(c[0]), int(c[1])
        cv2.circle(vis, (cx, cy), 8, (0, 128, 255), -1)
        cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)
    y0 = 28
    cv2.putText(
        vis,
        title,
        (12, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for i, line in enumerate(instruction.split("\n")):
        cv2.putText(
            vis,
            line,
            (12, y0 + 28 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    return vis


@dataclass
class _CalibrationState:
    perspective_matrix: np.ndarray
    warp_w: int
    warp_h: int
    board_zone: list[tuple[float, float]]
    hand_zone: list[tuple[float, float]]
    camera_index: int


class CameraManager:
    """
    Opens the laptop camera, applies saved perspective correction, and exposes
    board/hand zones as quadrilaterals in warped (corrected) pixel coordinates.

    Use :meth:`read_raw_and_warped` in the live loop: raw frames show the full
    sensor FOV in the UI; warped frames feed detection and OCR on the rectified
    sheet.
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        *,
        project_root: Path | str | None = None,
        force_recalibrate: bool = False,
    ) -> None:
        """
        ``force_recalibrate`` (used by ``main`` on each boot) opens any saved
        ``calibration.json`` only to recover ``camera_index``, then runs the full
        interactive calibration and overwrites the file.
        """
        root = Path(project_root) if project_root is not None else _project_root()
        self._project_root = root
        self._config_path = (
            Path(config_path) if config_path is not None else root / "config" / "calibration.json"
        )

        self._cap: cv2.VideoCapture | None = None
        self._camera_index: int = -1
        self._matrix: np.ndarray | None = None
        self._matrix_inv: np.ndarray | None = None
        self._warp_w: int = 0
        self._warp_h: int = 0
        self._board_zone: list[tuple[float, float]] = []
        self._hand_zone: list[tuple[float, float]] = []

        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        if self._config_path.is_file() and not force_recalibrate:
            self._load_config_and_open_camera()
        else:
            if self._config_path.is_file():
                self._load_config_and_open_camera()
            if not cv2_highgui_available():
                raise RuntimeError(HIGHGUI_UNAVAILABLE_HINT)
            self._ensure_capture_for_calibration()
            self.run_calibration()

        if self._cap is None or not self._cap.isOpened():
            warnings.warn("Camera could not be opened; get_frame() will return None.")

    def _set_perspective(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float64)
        if self._matrix.shape != (3, 3):
            raise ValueError("perspective matrix must be 3x3")
        try:
            self._matrix_inv = np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError:
            self._matrix_inv = None
            warnings.warn(
                "Perspective matrix is singular; source-space overlays are disabled.",
                stacklevel=2,
            )

    @property
    def board_zone(self) -> list[tuple[float, float]]:
        return list(self._board_zone)

    @property
    def hand_zone(self) -> list[tuple[float, float]]:
        return list(self._hand_zone)

    @property
    def warp_to_source_matrix(self) -> np.ndarray | None:
        """
        3×3 inverse homography mapping rectified (warp) pixel coordinates to the
        raw camera frame. None if calibration is missing or singular.
        """
        return None if self._matrix_inv is None else self._matrix_inv.copy()

    def point_in_board_zone(self, x: float, y: float) -> bool:
        if len(self._board_zone) != 4:
            return False
        cnt = np.array(self._board_zone, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0

    def point_in_hand_zone(self, x: float, y: float) -> bool:
        if len(self._hand_zone) != 4:
            return False
        cnt = np.array(self._hand_zone, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of frame with board (green) and hand (blue) polygons drawn."""
        out = frame.copy()
        if len(self._board_zone) == 4:
            b = np.array(self._board_zone, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [b], isClosed=True, color=(0, 255, 0), thickness=2)
        if len(self._hand_zone) == 4:
            h = np.array(self._hand_zone, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [h], isClosed=True, color=(255, 0, 0), thickness=2)
        return out

    def draw_zones_on_source(self, frame_source: np.ndarray) -> np.ndarray:
        """
        Draw board (green) and hand (red) quads on a **raw** camera frame.

        Zone corners are stored in rectified coordinates; they are mapped through
        ``warp_to_source_matrix`` before drawing.
        """
        out = frame_source.copy()
        inv_m = self._matrix_inv
        if inv_m is None:
            return out

        def _draw_zone(corners: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
            if len(corners) != 4:
                return
            pts_w = np.array(corners, dtype=np.float32).reshape(1, 4, 2)
            pts_s = cv2.perspectiveTransform(pts_w, inv_m)
            pi = np.round(pts_s).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pi], isClosed=True, color=color, thickness=2)

        _draw_zone(self._board_zone, (0, 255, 0))
        _draw_zone(self._hand_zone, (255, 0, 0))
        return out

    def read_raw_and_warped(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Single camera grab: raw full-FOV frame plus rectified sheet view.

        Use this in the main loop so vision and display stay time-aligned.
        """
        if self._cap is None or not self._cap.isOpened():
            return None, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, None
        if self._matrix is None or self._warp_w < 1 or self._warp_h < 1:
            return frame, None
        warped = cv2.warpPerspective(
            frame,
            self._matrix,
            (self._warp_w, self._warp_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return frame, warped

    def read_raw_frame(self) -> np.ndarray | None:
        """One grab: raw frame only (does not return the paired rectified view)."""
        raw, _ = self.read_raw_and_warped()
        return raw

    def get_frame(self) -> np.ndarray | None:
        return self.read_warped_frame()

    def read_warped_frame(self) -> np.ndarray | None:
        """Rectified view from one fresh grab (prefer :meth:`read_raw_and_warped` in loops)."""
        _, warped = self.read_raw_and_warped()
        return warped

    def read_frame(self) -> np.ndarray | None:
        """Rectified frame; same as :meth:`read_warped_frame` (vision / OCR pipeline)."""
        return self.read_warped_frame()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        safe_destroy_all_windows()

    def run_calibration(self) -> None:
        """
        Run perspective and zone calibration, then save and apply the result.

        On cancel (ESC, closed calibration window, or KeyboardInterrupt), the
        previous calibration.json is left unchanged. On success, in-memory
        geometry and get_frame() use the new calibration immediately.
        """
        if not cv2_highgui_available():
            raise RuntimeError(HIGHGUI_UNAVAILABLE_HINT)
        self._ensure_capture_for_calibration()
        cap = self._cap
        if cap is None or not cap.isOpened():
            raise RuntimeError("Camera is not available for calibration.")
        cam_idx = self._camera_index
        state = self._collect_calibration_interactive(cap, cam_idx)
        self._save_config_atomic(state)
        self._apply_calibration_state(state)

    def _ensure_capture_for_calibration(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        order: list[int] = []
        if self._camera_index >= 0:
            order.append(self._camera_index)
        for i in (0, 1):
            if i not in order:
                order.append(i)
        cap, used_idx = _open_capture(order)
        if cap is None:
            raise RuntimeError("Could not open camera for calibration.")
        self._cap = cap
        self._camera_index = used_idx

    def _apply_calibration_state(self, state: _CalibrationState) -> None:
        self._set_perspective(state.perspective_matrix)
        self._warp_w = state.warp_w
        self._warp_h = state.warp_h
        self._board_zone = state.board_zone
        self._hand_zone = state.hand_zone
        self._camera_index = state.camera_index

    def _load_config_and_open_camera(self) -> None:
        with open(self._config_path, encoding="utf-8") as f:
            data = json.load(f)

        matrix = np.array(data["perspective_matrix"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("calibration.json: perspective_matrix must be 3x3")

        warp_shape = data.get("warp_shape")
        if warp_shape is None:
            raise ValueError(
                "calibration.json: warp_shape [width, height] is required for warping"
            )
        warp_w, warp_h = int(warp_shape[0]), int(warp_shape[1])

        board = [tuple(float(x) for x in p) for p in data["board_zone"]]
        hand = [tuple(float(x) for x in p) for p in data["hand_zone"]]
        if len(board) != 4 or len(hand) != 4:
            raise ValueError("calibration.json: board_zone and hand_zone need 4 points each")

        saved_idx = int(data.get("camera_index", 0))
        cap, used_idx = _open_capture([saved_idx, 0, 1])
        if cap is None:
            self._set_perspective(matrix)
            self._warp_w = warp_w
            self._warp_h = warp_h
            self._board_zone = board
            self._hand_zone = hand
            self._camera_index = saved_idx
            return

        if used_idx != saved_idx:
            warnings.warn(
                f"Camera index {saved_idx} from config failed to open; using index {used_idx}."
            )

        self._cap = cap
        self._set_perspective(matrix)
        self._warp_w = warp_w
        self._warp_h = warp_h
        self._board_zone = board
        self._hand_zone = hand
        self._camera_index = used_idx

    def _save_config_atomic(self, state: _CalibrationState) -> None:
        payload = {
            "perspective_matrix": state.perspective_matrix.astype(float).tolist(),
            "warp_shape": [state.warp_w, state.warp_h],
            "board_zone": [[float(x), float(y)] for x, y in state.board_zone],
            "hand_zone": [[float(x), float(y)] for x, y in state.hand_zone],
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "camera_index": state.camera_index,
        }
        dest = self._config_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="calibration.",
            suffix=".json.tmp",
            dir=dest.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        print(f"Calibration saved to {dest}")

    def _collect_calibration_interactive(
        self, cap: cv2.VideoCapture, cam_idx: int
    ) -> _CalibrationState:
        try:
            matrix, warp_w, warp_h = self._calibrate_perspective_interactive(cap)
            board = self._select_zone_interactive(
                cap,
                matrix,
                warp_w,
                warp_h,
                _WIN_BOARD_ZONE,
                "BOARD zone",
                "BOARD = your play area on the rectified table (not just the sheet).\n"
                "Drag the four corners, press ENTER when done",
            )
            hand = self._select_zone_interactive(
                cap,
                matrix,
                warp_w,
                warp_h,
                _WIN_HAND_ZONE,
                "HAND zone",
                "HAND = rack area on the rectified view. Drag corners, ENTER when done",
            )
        except KeyboardInterrupt:
            safe_destroy_all_windows()
            raise
        except Exception:
            safe_destroy_all_windows()
            raise

        return _CalibrationState(
            perspective_matrix=matrix,
            warp_w=warp_w,
            warp_h=warp_h,
            board_zone=board,
            hand_zone=hand,
            camera_index=cam_idx,
        )

    def _warp(self, frame: np.ndarray, matrix: np.ndarray, w: int, h: int) -> np.ndarray:
        return cv2.warpPerspective(
            frame,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _calibrate_perspective_interactive(self, cap: cv2.VideoCapture) -> tuple[np.ndarray, int, int]:
        print(
            "Place a sheet of A4 or Letter paper flat on your playing surface. "
            "Make sure the surface is darker than the paper. "
            "Press SPACE to capture (ESC to cancel)."
        )
        try:
            frame_idx = 0
            detection_failed = False
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera read failed during perspective calibration.")
                hint = frame.copy()
                _draw_perspective_calibration_overlay(
                    hint, detection_failed=detection_failed
                )
                cv2.imshow(_WIN_PERSPECTIVE, hint)
                key = cv2.waitKey(1) & 0xFF
                frame_idx += 1
                if frame_idx >= 5 and _calibration_window_closed(_WIN_PERSPECTIVE):
                    raise RuntimeError("Calibration cancelled.")
                if key == 27:
                    raise RuntimeError("Calibration cancelled.")
                if key == ord(" "):
                    quad = _find_largest_sheet_quad(frame)
                    if quad is None:
                        detection_failed = True
                        print(
                            "Sheet not detected. Check that the paper is on a dark surface "
                            "and fully in frame, then press SPACE to try again."
                        )
                        continue
                    m0, w0, h0 = _perspective_from_quad(quad)
                    m, w, h = _expand_warp_to_table_plane(
                        m0, w0, h0, frame.shape[1], frame.shape[0]
                    )
                    print(
                        "Plane + angle locked from sheet; rectified view spans full frame "
                        f"on the table ({w}×{h}). Draw board/hand zones over your real play area."
                    )
                    return m, w, h
        finally:
            safe_destroy_window(_WIN_PERSPECTIVE)

    def _select_zone_interactive(
        self,
        cap: cv2.VideoCapture,
        matrix: np.ndarray,
        warp_w: int,
        warp_h: int,
        window_name: str,
        title: str,
        instruction: str,
    ) -> list[tuple[float, float]]:
        margin_x = max(8, int(warp_w * 0.08))
        margin_y = max(8, int(warp_h * 0.08))
        corners: list[list[float]] = [
            [float(margin_x), float(margin_y)],
            [float(warp_w - margin_x), float(margin_y)],
            [float(warp_w - margin_x), float(warp_h - margin_y)],
            [float(margin_x), float(warp_h - margin_y)],
        ]
        cb = _zone_mouse_callback_factory(corners)
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, cb)
        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera read failed during zone selection.")
                warped = self._warp(frame, matrix, warp_w, warp_h)
                vis = _draw_zone_ui(warped, corners, title, instruction)
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(16) & 0xFF
                frame_idx += 1
                if frame_idx >= 5 and _calibration_window_closed(window_name):
                    raise RuntimeError("Calibration cancelled.")
                if key == 13:
                    break
                if key == 27:
                    raise RuntimeError("Calibration cancelled.")
        finally:
            safe_destroy_window(window_name)

        return [(float(c[0]), float(c[1])) for c in corners]


def _interactive_main() -> None:
    if not cv2_highgui_available():
        print(HIGHGUI_UNAVAILABLE_HINT, file=sys.stderr)
        raise SystemExit(1)
    cam = CameraManager()
    try:
        print("Running live preview (q to quit). Full FOV with board/hand quads.")
        while True:
            raw, _w = cam.read_raw_and_warped()
            if raw is None:
                print("No frame.")
                break
            preview = cam.draw_zones_on_source(raw)
            cv2.imshow("Bananagrams — preview", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()


CameraFeed = CameraManager


if __name__ == "__main__":
    _interactive_main()
