"""
Letter recognition on cropped tile regions; writes results into VisionTracker.

Requires the ``pytesseract`` package and a Tesseract OCR installation at runtime.
Tests mock ``pytesseract.image_to_data`` and do not need Tesseract.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .tracker import VisionTracker

TESS_CONFIG = (
    "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

_TILE_CROP_PADDING_PX = 4


@dataclass
class OCRResult:
    tile_id: str
    letter: str | None
    confidence: float
    needs_recheck: bool
    attempted_at: int


def _crop_tile_region(
    frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = _TILE_CROP_PADDING_PX
) -> np.ndarray | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    fh, fw = frame.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(fw, x + w + pad)
    y1 = min(fh, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1].copy()


def _preprocess_tile(roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _parse_best_letter_from_tesseract_data(data: dict[str, list]) -> tuple[str | None, float]:
    """Best single A–Z letter and normalized confidence [0, 1] from image_to_data dict."""
    texts = data.get("text", [])
    confs = data.get("conf", [])
    n = min(len(texts), len(confs))
    best_char: str | None = None
    best_conf = 0.0
    for i in range(n):
        raw = texts[i]
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            c_raw = confs[i]
            conf = float(c_raw) if c_raw not in ("", "-1", "nan") else -1.0
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        c = text[0].upper()
        if not ("A" <= c <= "Z"):
            continue
        conf01 = conf / 100.0
        if conf01 > best_conf:
            best_conf = conf01
            best_char = c
    return best_char, best_conf


def _ocr_single_orientation(prepared: np.ndarray) -> tuple[str | None, float]:
    data = pytesseract.image_to_data(
        prepared,
        config=TESS_CONFIG,
        output_type=Output.DICT,
    )
    return _parse_best_letter_from_tesseract_data(data)


def _ocr_best_over_rotations(prepared: np.ndarray) -> tuple[str | None, float]:
    best_letter: str | None = None
    best_conf = 0.0
    cur = prepared
    for _ in range(4):
        letter, conf = _ocr_single_orientation(cur)
        if letter is not None and conf > best_conf:
            best_conf = conf
            best_letter = letter
        cur = cv2.rotate(cur, cv2.ROTATE_90_CLOCKWISE)
    return best_letter, best_conf


def _prune_recheck_queue(tracker: VisionTracker, recheck: set[str]) -> None:
    """Drop recheck entries for tiles that no longer need OCR (dumped or letter committed)."""
    unlettered_ids = {t.tile_id for t in tracker.get_unlettered_tiles()}
    for tid in list(recheck):
        if tid not in unlettered_ids:
            recheck.discard(tid)


class TileOCR:
    """
    Runs Tesseract on unlettered tile crops and commits high-confidence reads.
    """

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self._confidence_threshold = confidence_threshold
        self._recheck: set[str] = set()

    def get_recheck_queue(self) -> list[str]:
        return sorted(self._recheck)

    def clear_recheck(self, tile_id: str) -> None:
        self._recheck.discard(tile_id)

    def process_frame(
        self,
        frame: np.ndarray,
        tracker: VisionTracker,
        *,
        frame_number: int = 0,
    ) -> dict[str, OCRResult]:
        _prune_recheck_queue(tracker, self._recheck)

        work = list(tracker.get_unlettered_tiles())
        results: dict[str, OCRResult] = {}

        for tile in work:
            roi = _crop_tile_region(frame, tile.bbox)
            if roi is None or roi.size == 0:
                res = OCRResult(
                    tile_id=tile.tile_id,
                    letter=None,
                    confidence=0.0,
                    needs_recheck=True,
                    attempted_at=frame_number,
                )
                results[tile.tile_id] = res
                self._recheck.add(tile.tile_id)
                continue

            prepared = _preprocess_tile(roi)
            letter, conf = _ocr_best_over_rotations(prepared)

            if letter is None or conf < self._confidence_threshold:
                res = OCRResult(
                    tile_id=tile.tile_id,
                    letter=None,
                    confidence=conf,
                    needs_recheck=True,
                    attempted_at=frame_number,
                )
                results[tile.tile_id] = res
                self._recheck.add(tile.tile_id)
                continue

            tracker.set_tile_letter(tile.tile_id, letter)
            self._recheck.discard(tile.tile_id)
            res = OCRResult(
                tile_id=tile.tile_id,
                letter=letter,
                confidence=conf,
                needs_recheck=False,
                attempted_at=frame_number,
            )
            results[tile.tile_id] = res

        return results


class OCRReader:
    """
    Integration wrapper: run :class:`TileOCR` and expose ``tile_id -> letter`` for
    :class:`vision.mapper.StateMapper`.
    """

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self._engine = TileOCR(confidence_threshold=confidence_threshold)
        self._frame_number = 0

    def read(self, frame: np.ndarray, tracker: VisionTracker) -> dict[str, str]:
        self._frame_number += 1
        results = self._engine.process_frame(
            frame, tracker, frame_number=self._frame_number
        )
        out: dict[str, str] = {}
        for tid, res in results.items():
            if res.letter is not None:
                out[tid] = res.letter
        return out
