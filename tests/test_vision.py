"""Vision layer tests (OCR + tracker hooks); Tesseract is always mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from vision.camera import _expand_warp_to_table_plane, _perspective_from_quad
from vision.ocr import OCRResult, TileOCR
from vision.tracker import (
    TileRecord,
    TileState,
    VisionTracker,
    Zone,
    _TrackedTile,
)


def _synthetic_frame() -> np.ndarray:
    """Light tile-like patch on dark background (no disk assets)."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[60:140, 60:140] = (235, 220, 180)
    return img


def _make_tesseract_dict(letter: str, conf_percent: float) -> dict[str, list]:
    return {
        "text": ["", letter, ""],
        "conf": ["-1", str(int(conf_percent)), "-1"],
    }


@pytest.fixture
def mock_tracker_tile() -> tuple[MagicMock, TileRecord]:
    tile = TileRecord(
        tile_id="tile_001",
        zone=Zone.BOARD,
        bbox=(60, 60, 80, 80),
        is_occluded=False,
        state=TileState.ACTIVE,
        letter=None,
        grid_pos=None,
    )
    tracker = MagicMock(spec=VisionTracker)
    tracker.get_unlettered_tiles.return_value = [tile]
    tracker.get_active_tiles.return_value = [tile]
    tracker.set_tile_letter = MagicMock(return_value=True)
    return tracker, tile


def test_high_confidence_commits_letter_and_no_recheck(mock_tracker_tile) -> None:
    tracker, tile = mock_tracker_tile
    ocr = TileOCR(confidence_threshold=0.75)
    frame = _synthetic_frame()

    with patch(
        "vision.ocr.pytesseract.image_to_data",
        return_value=_make_tesseract_dict("A", 92),
    ):
        out = ocr.process_frame(frame, tracker, frame_number=7)

    assert "tile_001" in out
    res = out["tile_001"]
    assert isinstance(res, OCRResult)
    assert res.letter == "A"
    assert res.confidence == pytest.approx(0.92)
    assert res.needs_recheck is False
    assert res.attempted_at == 7
    tracker.set_tile_letter.assert_called_once_with("tile_001", "A")
    assert ocr.get_recheck_queue() == []


def test_low_confidence_no_commit_and_recheck_queued(mock_tracker_tile) -> None:
    tracker, _tile = mock_tracker_tile
    ocr = TileOCR(confidence_threshold=0.75)
    frame = _synthetic_frame()

    with patch(
        "vision.ocr.pytesseract.image_to_data",
        return_value=_make_tesseract_dict("B", 50),
    ):
        out = ocr.process_frame(frame, tracker, frame_number=1)

    res = out["tile_001"]
    assert res.letter is None
    assert res.needs_recheck is True
    assert res.confidence == pytest.approx(0.5)
    tracker.set_tile_letter.assert_not_called()
    assert "tile_001" in ocr.get_recheck_queue()


def test_confident_read_on_retry_clears_recheck(mock_tracker_tile) -> None:
    tracker, _tile = mock_tracker_tile
    ocr = TileOCR(confidence_threshold=0.75)
    frame = _synthetic_frame()

    with patch(
        "vision.ocr.pytesseract.image_to_data",
        return_value=_make_tesseract_dict("C", 40),
    ):
        ocr.process_frame(frame, tracker, frame_number=0)
    assert "tile_001" in ocr.get_recheck_queue()

    with patch(
        "vision.ocr.pytesseract.image_to_data",
        return_value=_make_tesseract_dict("C", 96),
    ):
        ocr.process_frame(frame, tracker, frame_number=1)

    assert ocr.get_recheck_queue() == []
    assert tracker.set_tile_letter.call_count == 1
    tracker.set_tile_letter.assert_called_with("tile_001", "C")


def test_recheck_pruned_when_tile_not_in_tracker() -> None:
    tracker = MagicMock(spec=VisionTracker)
    tracker.get_unlettered_tiles.return_value = []
    tracker.get_active_tiles.return_value = []
    ocr = TileOCR()
    ocr._recheck.add("gone_tile")
    frame = _synthetic_frame()

    with patch("vision.ocr.pytesseract.image_to_data"):
        ocr.process_frame(frame, tracker, frame_number=0)

    assert ocr.get_recheck_queue() == []


def test_clear_recheck() -> None:
    ocr = TileOCR()
    ocr._recheck.add("x")
    ocr.clear_recheck("x")
    assert ocr.get_recheck_queue() == []


@pytest.fixture
def camera_mock() -> MagicMock:
    cam = MagicMock()
    cam.point_in_board_zone.return_value = True
    cam.point_in_hand_zone.return_value = False
    return cam


def test_flag_dump_candidates_only_matching_letter(camera_mock) -> None:
    tracker = VisionTracker(camera_mock)
    tracker._tracks["tile_001"] = _TrackedTile(
        record=TileRecord(
            tile_id="tile_001",
            zone=Zone.BOARD,
            bbox=(0, 0, 10, 10),
            is_occluded=False,
            state=TileState.ACTIVE,
            letter="E",
            grid_pos=None,
        ),
        frames_missing=0,
    )
    tracker._tracks["tile_002"] = _TrackedTile(
        record=TileRecord(
            tile_id="tile_002",
            zone=Zone.BOARD,
            bbox=(20, 0, 10, 10),
            is_occluded=False,
            state=TileState.ACTIVE,
            letter="A",
            grid_pos=None,
        ),
        frames_missing=0,
    )
    tracker.flag_dump_candidates("E")
    assert tracker._tracks["tile_001"].record.is_dump_candidate is True
    assert tracker._tracks["tile_002"].record.is_dump_candidate is False


def test_confirm_dump_clears_dump_candidates_same_letter(camera_mock) -> None:
    tracker = VisionTracker(camera_mock)
    tracker._tracks["tile_001"] = _TrackedTile(
        record=TileRecord(
            tile_id="tile_001",
            zone=Zone.BOARD,
            bbox=(0, 0, 10, 10),
            is_occluded=False,
            state=TileState.ACTIVE,
            letter="E",
            is_dump_candidate=True,
            grid_pos=None,
        ),
        frames_missing=0,
    )
    tracker._tracks["tile_002"] = _TrackedTile(
        record=TileRecord(
            tile_id="tile_002",
            zone=Zone.BOARD,
            bbox=(20, 0, 10, 10),
            is_occluded=False,
            state=TileState.ACTIVE,
            letter="E",
            is_dump_candidate=True,
            grid_pos=None,
        ),
        frames_missing=0,
    )
    tracker.confirm_dump("tile_001")
    assert "tile_001" not in tracker._tracks
    assert tracker._tracks["tile_002"].record.is_dump_candidate is False


def test_flagged_tile_occludes_under_hand_not_dumped(camera_mock) -> None:
    tracker = VisionTracker(camera_mock, occlusion_grace_frames=3)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    tile_bbox = (300, 300, 50, 50)

    with (
        patch(
            "vision.tracker._detect_tile_bboxes",
            side_effect=[
                [tile_bbox],
                [],
                [],
                [],
            ],
        ),
        patch(
            "vision.tracker._detect_hand_bbox",
            return_value=(0, 0, 120, 120),
        ),
    ):
        tracker.update(frame)
        assert tracker.set_tile_letter("tile_001", "E") is True
        tracker.flag_dump_candidates("E")
        for _ in range(3):
            tracker.update(frame)

    t = tracker._tracks["tile_001"]
    assert t.record.state == TileState.OCCLUDED
    assert t.record.is_dump_candidate is True
    assert t.record.letter == "E"


def test_expand_warp_covers_full_frame_not_only_paper() -> None:
    """Sheet-sized quad in a larger frame → rectified canvas grows past paper patch."""
    fw, fh = 640, 480
    src = np.array(
        [[200.0, 150.0], [440.0, 150.0], [440.0, 330.0], [200.0, 330.0]], dtype=np.float32
    )
    m0, w0, h0 = _perspective_from_quad(src)
    m, big_w, big_h = _expand_warp_to_table_plane(m0, w0, h0, fw, fh)
    assert big_w >= w0 and big_h >= h0
    assert big_w * big_h > w0 * h0 * 1.5
    corners = np.array([[[0.0, 0.0], [float(fw - 1), float(fh - 1)]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(corners, m)
    assert np.all(warped[0, :, 0] >= -0.5) and np.all(warped[0, :, 1] >= -0.5)
    assert np.all(warped[0, :, 0] < float(big_w)) and np.all(warped[0, :, 1] < float(big_h))
