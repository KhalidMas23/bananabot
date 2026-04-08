"""Pytest: validate OpenCV before importing test modules that use ``cv2``."""

from vision.opencv_util import ensure_opencv_cv2_runtime

ensure_opencv_cv2_runtime()
