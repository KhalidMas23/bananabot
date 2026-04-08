"""
Tile and hand tracking on perspective-corrected (rectified sheet) frames.

Bounding boxes and zone tests live in warped pixel space; the main UI reprojects
them onto the raw camera image via ``CameraManager.warp_to_source_matrix``.

Uses CameraManager only for zone classification. Does not import core/.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from enum import Enum, auto

import cv2
import numpy as np

from .camera import CameraManager

logger = logging.getLogger(__name__)


class Zone(Enum):
    BOARD = auto()
    HAND = auto()
    UNKNOWN = auto()


class TileState(Enum):
    ACTIVE = auto()
    OCCLUDED = auto()
    CONFIRMED_DUMPED = auto()


@dataclass
class TileRecord:
    """
    Per-tile snapshot in warped frame space (stable contract for OCR and mapper).

    tile_id:
        Unique ID assigned on first detection. Never reused.
    zone:
        Current zone classification (BOARD, HAND, UNKNOWN).
    bbox:
        Bounding box (x, y, w, h) in warped frame coordinates.
    is_occluded:
        True if tile is currently hidden by hand.
    state:
        ACTIVE, OCCLUDED, or CONFIRMED_DUMPED.
    is_dump_candidate:
        Set by mapper via flag_dump_candidates(). Cleared on confirm or cancel.
    letter:
        Set by OCR via set_tile_letter(). None until first successful read.
        Persists through occlusion. Cleared only if OCR explicitly passes None.
    grid_pos:
        Last known board cell ``(row, col)`` for overlay / mapper; maintained by
        ``StateMapper`` via ``set_tile_grid_pos``. None when not on the board.
    """

    tile_id: str
    zone: Zone
    bbox: tuple[int, int, int, int]
    is_occluded: bool
    state: TileState
    is_dump_candidate: bool = False
    letter: str | None = None
    grid_pos: tuple[int, int] | None = None


def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + 0.5 * w, y + 0.5 * h)


def _classify_zone(
    camera: CameraManager, cx: float, cy: float
) -> Zone:
    in_b = camera.point_in_board_zone(cx, cy)
    in_h = camera.point_in_hand_zone(cx, cy)
    if in_b:
        return Zone.BOARD
    if in_h:
        return Zone.HAND
    return Zone.UNKNOWN


def _detect_hand_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest skin-toned region in HSV; None if no plausible hand."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 60], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = frame.shape[:2]
    min_a = 0.002 * fw * fh
    best: tuple[int, int, int, int] | None = None
    best_a = 0.0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < min_a:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if a > best_a:
            best_a = a
            best = (int(x), int(y), int(w), int(h))
    return best


def _detect_tile_bboxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Light square-ish regions (cream / yellow-white tiles)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([12, 30, 100], dtype=np.uint8)
    upper = np.array([48, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = frame.shape[:2]
    frame_area = float(fh * fw)
    min_a = max(80.0, 0.00015 * frame_area)
    max_a = 0.035 * frame_area
    out: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < min_a or a > max_a:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / max(h, 1)
        if ar < 0.55 or ar > 1.85:
            continue
        out.append((int(x), int(y), int(w), int(h)))
    out.sort(key=lambda b: (b[2] * b[3]), reverse=True)
    return out


@dataclass
class _TrackedTile:
    record: TileRecord
    frames_missing: int = 0


class VisionTracker:
    """
    Tracks face-up tiles by centroid matching and a simple hand presence signal.
    """

    def __init__(
        self,
        camera: CameraManager,
        *,
        id_match_threshold_px: float = 40.0,
        occlusion_grace_frames: int = 5,
    ) -> None:
        self._camera = camera
        self._id_match_threshold_px = id_match_threshold_px
        self._occlusion_grace_frames = occlusion_grace_frames

        self._tracks: dict[str, _TrackedTile] = {}
        self._dumped: dict[str, TileRecord] = {}
        self._next_id = 1

        self._hand_bbox: tuple[int, int, int, int] | None = None
        self._hand_present: bool = False
        self._prev_hand_present: bool = False

    def update(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            return

        self._prev_hand_present = self._hand_present
        hb = _detect_hand_bbox(frame)
        self._hand_bbox = hb
        self._hand_present = hb is not None

        dets = _detect_tile_bboxes(frame)
        det_data: list[tuple[int, tuple[int, int, int, int], float, float]] = []
        for i, bb in enumerate(dets):
            cx, cy = _centroid(bb)
            det_data.append((i, bb, cx, cy))

        active_tracks = {
            tid: t
            for tid, t in self._tracks.items()
            if t.record.state in (TileState.ACTIVE, TileState.OCCLUDED)
        }

        pairs: list[tuple[float, str, int]] = []
        for tid, tr in active_tracks.items():
            tcx, tcy = _centroid(tr.record.bbox)
            for i, _bb, dcx, dcy in det_data:
                d = math.hypot(dcx - tcx, dcy - tcy)
                if d <= self._id_match_threshold_px:
                    pairs.append((d, tid, i))
        pairs.sort(key=lambda x: x[0])

        matched_tile: set[str] = set()
        matched_det: set[int] = set()
        tile_to_bbox: dict[str, tuple[int, int, int, int]] = {}
        for _d, tid, di in pairs:
            if tid in matched_tile or di in matched_det:
                continue
            matched_tile.add(tid)
            matched_det.add(di)
            tile_to_bbox[tid] = det_data[di][1]

        hand_now = self._hand_present

        for tid, bb in tile_to_bbox.items():
            tr = self._tracks[tid]
            cx, cy = _centroid(bb)
            zone = _classify_zone(self._camera, cx, cy)
            r = tr.record
            new_zone = zone
            if r.zone == Zone.HAND and zone == Zone.BOARD:
                new_zone = Zone.BOARD
            elif r.zone == Zone.BOARD and zone == Zone.HAND:
                new_zone = Zone.HAND
            elif zone == Zone.UNKNOWN and r.zone in (Zone.BOARD, Zone.HAND):
                new_zone = r.zone

            tr.frames_missing = 0
            tr.record = replace(
                r,
                bbox=bb,
                zone=new_zone,
                state=TileState.ACTIVE,
                is_occluded=False,
            )

        if self._prev_hand_present and not hand_now:
            for tid, tr in list(self._tracks.items()):
                if tr.record.state != TileState.OCCLUDED:
                    continue
                if tid in matched_tile:
                    continue
                self._finalize_dump(tid, tr.record)

        for tid, tr in list(self._tracks.items()):
            if tr.record.state == TileState.CONFIRMED_DUMPED:
                continue
            if tid in matched_tile:
                continue

            tr.frames_missing += 1
            r = tr.record

            if not hand_now:
                self._finalize_dump(tid, r)
                continue

            if tr.frames_missing >= self._occlusion_grace_frames:
                tr.record = replace(
                    r,
                    state=TileState.OCCLUDED,
                    is_occluded=True,
                )
            elif r.state == TileState.OCCLUDED:
                tr.record = replace(r, is_occluded=True)
            else:
                tr.record = replace(r, is_occluded=False)

        for i, bb, _cx, _cy in det_data:
            if i in matched_det:
                continue
            self._spawn_tile(bb)

    def _spawn_tile(self, bbox: tuple[int, int, int, int]) -> None:
        cx, cy = _centroid(bbox)
        zone = _classify_zone(self._camera, cx, cy)
        tid = f"tile_{self._next_id:03d}"
        self._next_id += 1
        self._tracks[tid] = _TrackedTile(
            record=TileRecord(
                tile_id=tid,
                zone=zone,
                bbox=bbox,
                is_occluded=False,
                state=TileState.ACTIVE,
                letter=None,
            ),
            frames_missing=0,
        )

    def _finalize_dump(self, tid: str, record: TileRecord) -> None:
        if record.state == TileState.CONFIRMED_DUMPED:
            return
        dumped = replace(
            record,
            state=TileState.CONFIRMED_DUMPED,
            is_occluded=False,
            is_dump_candidate=False,
        )
        self._dumped[tid] = dumped
        del self._tracks[tid]

    def get_active_tiles(self) -> list[TileRecord]:
        return [
            t.record
            for t in self._tracks.values()
            if t.record.state in (TileState.ACTIVE, TileState.OCCLUDED)
        ]

    def get_confirmed_dumped(self) -> list[TileRecord]:
        return list(self._dumped.values())

    def hand_present(self) -> bool:
        return self._hand_present

    def hand_bbox(self) -> tuple[int, int, int, int] | None:
        return self._hand_bbox

    def flag_dump_candidates(self, letter: str) -> None:
        ch = letter.strip()[:1].upper() if letter else ""
        if not ch:
            return
        for tr in self._tracks.values():
            r = tr.record
            if r.state == TileState.CONFIRMED_DUMPED:
                continue
            if r.letter == ch:
                tr.record = replace(r, is_dump_candidate=True)

    def confirm_dump(self, tile_id: str) -> None:
        if tile_id not in self._tracks:
            return
        tr = self._tracks[tile_id]
        r = tr.record
        if r.state == TileState.CONFIRMED_DUMPED:
            return
        letter = r.letter
        self._finalize_dump(tile_id, r)
        for _tid, t2 in list(self._tracks.items()):
            rec = t2.record
            if letter is not None and rec.letter == letter:
                t2.record = replace(rec, is_dump_candidate=False)

    def clear_dump_candidates(self) -> None:
        for tid, tr in self._tracks.items():
            r = tr.record
            if r.is_dump_candidate:
                tr.record = replace(r, is_dump_candidate=False)

    def set_tile_letter(self, tile_id: str, letter: str | None) -> bool:
        """
        OCR hook: set or clear the letter for an active track.

        ``None`` clears the letter (unreadable). Unknown ``tile_id`` returns
        False (e.g. tile just dumped in the same frame); never raises.
        """
        if tile_id not in self._tracks:
            return False
        tr = self._tracks[tile_id]
        r = tr.record
        if letter is None:
            stored: str | None = None
        else:
            stored = letter.strip().upper()
            if stored == "":
                stored = None
        tr.record = replace(r, letter=stored)
        return True

    def set_tile_grid_pos(
        self, tile_id: str, grid_pos: tuple[int, int] | None
    ) -> bool:
        """
        Mapper hook: annotate the last known board cell for a track.

        Does not change zone, occlusion, or dump state — only ``grid_pos``.

        If ``tile_id`` is missing (e.g. tile just dumped this frame), logs a warning
        and returns False; never raises.
        """
        if tile_id not in self._tracks:
            logger.warning(
                "set_tile_grid_pos: unknown tile_id=%r — no-op (tile may have been dumped)",
                tile_id,
            )
            return False
        tr = self._tracks[tile_id]
        r = tr.record
        tr.record = replace(r, grid_pos=grid_pos)
        return True

    def get_tile_letter(self, tile_id: str) -> str | None:
        """Stored letter for this track, or None if unknown id or letter not set."""
        if tile_id not in self._tracks:
            return None
        return self._tracks[tile_id].record.letter

    def get_unlettered_tiles(self) -> list[TileRecord]:
        """ACTIVE/OCCLUDED tiles with no letter yet — primary OCR work queue each frame."""
        return [
            t.record
            for t in self._tracks.values()
            if t.record.state in (TileState.ACTIVE, TileState.OCCLUDED)
            and t.record.letter is None
        ]
