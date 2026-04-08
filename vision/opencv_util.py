"""
OpenCV HighGUI helpers for environments where GUI functions are stubs (e.g. headless wheels).
"""

from __future__ import annotations

import time

import cv2

_GUI_AVAILABLE: bool | None = None

# If any of these are missing, ``import cv2`` resolved to something other than opencv-python.
_REQUIRED_CV2_ATTRS: tuple[str, ...] = (
    "cvtColor",
    "COLOR_BGR2GRAY",
    "FONT_HERSHEY_SIMPLEX",
    "warpPerspective",
)


def ensure_opencv_cv2_runtime() -> None:
    """
    Fail fast when OpenCV is missing or shadowed (e.g. a local ``cv2.py`` on ``sys.path``).
    Call before importing modules that use ``cv2`` drawing or image ops.
    """
    missing = [name for name in _REQUIRED_CV2_ATTRS if not hasattr(cv2, name)]
    if not missing:
        return
    mod_file = getattr(cv2, "__file__", "(unknown)")
    raise ImportError(
        "The installed 'cv2' module is not OpenCV (or it failed to load its native extension). "
        f"Missing: {', '.join(missing)}. Loaded from: {mod_file}. "
        "Fix: remove any file or folder named cv2.py / cv2 in your project or PYTHONPATH, then reinstall:\n"
        "  pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y\n"
        "  pip install opencv-python"
    )

HIGHGUI_UNAVAILABLE_HINT = (
    "OpenCV GUI support is not available (common with opencv-python-headless). "
    "BananaBot needs windows for calibration and the live view. Install the full package:\n"
    "  pip uninstall opencv-python-headless -y\n"
    "  pip install opencv-python\n"
    "Or copy config/calibration.json from a machine where calibration already ran."
)


def cv2_highgui_available() -> bool:
    """Return True if ``namedWindow`` / ``imshow`` are functional in this build."""
    global _GUI_AVAILABLE
    if _GUI_AVAILABLE is not None:
        return _GUI_AVAILABLE
    try:
        probe = "__bananabot_gui_probe__"
        cv2.namedWindow(probe, cv2.WINDOW_NORMAL)
        try:
            cv2.destroyWindow(probe)
        except cv2.error:
            pass
        _GUI_AVAILABLE = True
    except cv2.error:
        _GUI_AVAILABLE = False
    return _GUI_AVAILABLE


def safe_destroy_window(name: str) -> None:
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def safe_destroy_all_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def safe_wait_key(delay_ms: int) -> int:
    """
    Like ``cv2.waitKey(delay_ms) & 0xFF``, but falls back to sleeping when waitKey fails.
    """
    try:
        return int(cv2.waitKey(delay_ms)) & 0xFF
    except cv2.error:
        time.sleep(max(0, delay_ms) / 1000.0)
        return 0xFF
