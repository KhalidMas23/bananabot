"""
Stateless OpenCV drawing helpers for the live camera panel and digital grid.

All functions either return a new array or mutate a provided BGR ``uint8`` image.
Does not import the solver or hold display state.
"""

from __future__ import annotations

import math
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
from vision.opencv_util import ensure_opencv_cv2_runtime

ensure_opencv_cv2_runtime()

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from core.solver import SolverRecommendation

from vision.tracker import TileRecord

# --- Visual constants (BGR) ---
_COLOR_BG_PANEL: tuple[int, int, int] = (48, 48, 48)
_COLOR_CELL_EMPTY: tuple[int, int, int] = (64, 64, 64)
_COLOR_CELL_NORMAL: tuple[int, int, int] = (250, 250, 250)
_COLOR_CELL_OCCLUDED_GRID: tuple[int, int, int] = (110, 110, 110)
_COLOR_TEXT_DARK: tuple[int, int, int] = (30, 30, 30)
_COLOR_TEXT_OCCLUDED_GRID: tuple[int, int, int] = (160, 160, 160)
_COLOR_TEXT_LIGHT: tuple[int, int, int] = (200, 200, 200)
_COLOR_BORDER_NEUTRAL: tuple[int, int, int] = (160, 160, 160)
_COLOR_BORDER_OCCLUDED: tuple[int, int, int] = (100, 100, 180)
_COLOR_DUMP: tuple[int, int, int] = (0, 0, 255)
_REASONING_BG: tuple[int, int, int] = (20, 20, 20)
_FONT_MAP: dict[str, int] = {
    "SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
    "COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
}
_FONT: int = _FONT_MAP.get(
    os.getenv("BANANABOT_FONT", "DUPLEX").upper(),
    cv2.FONT_HERSHEY_DUPLEX,
)
_DUMP_ALPHA: float = 0.45
_MIN_CELL_PX: int = 24
_GRID_MARGIN: int = 8


def _ensure_bgr_uint8(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if frame.dtype != np.uint8:
        raise TypeError("frame must be uint8 BGR")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image (H, W, 3)")
    return frame


def _draw_dashed_rectangle(
    img: npt.NDArray[np.uint8],
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 6,
    gap_length: int = 4,
) -> None:
    """Draw a dashed axis-aligned rectangle (OpenCV has no native dashed rect)."""
    x2, y2 = x + w, y + h
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # top
    cx = x
    while cx < x2:
        nx = min(cx + dash_length, x2)
        segments.append(((cx, y), (nx, y)))
        cx = nx + gap_length
    # bottom
    cx = x
    while cx < x2:
        nx = min(cx + dash_length, x2)
        segments.append(((cx, y2 - 1), (nx, y2 - 1)))
        cx = nx + gap_length
    # left
    cy = y
    while cy < y2:
        ny = min(cy + dash_length, y2)
        segments.append(((x, cy), (x, ny)))
        cy = ny + gap_length
    # right
    cy = y
    while cy < y2:
        ny = min(cy + dash_length, y2)
        segments.append(((x2 - 1, cy), (x2 - 1, ny)))
        cy = ny + gap_length
    for p1, p2 in segments:
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def _warp_bbox_corners_to_source(
    warp_to_source_3x3: npt.NDArray[np.floating], bbox: tuple[int, int, int, int]
) -> npt.NDArray[np.int32]:
    """Map axis-aligned bbox corners from rectified space to the raw camera plane."""
    x, y, w, h = bbox
    corners = np.array(
        [
            [
                [float(x), float(y)],
                [float(x + w), float(y)],
                [float(x + w), float(y + h)],
                [float(x), float(y + h)],
            ]
        ],
        dtype=np.float32,
    )
    src = cv2.perspectiveTransform(corners, warp_to_source_3x3)
    return np.round(src).astype(np.int32).reshape(-1, 1, 2)


def _draw_dashed_line_segment(
    img: npt.NDArray[np.uint8],
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    draw_on = True
    while pos < dist:
        chunk = dash_length if draw_on else gap_length
        chunk = min(chunk, dist - pos)
        if draw_on:
            xa, ya = x0 + ux * pos, y0 + uy * pos
            xb, yb = x0 + ux * (pos + chunk), y0 + uy * (pos + chunk)
            cv2.line(
                img,
                (int(round(xa)), int(round(ya))),
                (int(round(xb)), int(round(yb))),
                color,
                thickness,
                cv2.LINE_AA,
            )
        pos += chunk
        draw_on = not draw_on


def _draw_dashed_quad(
    img: npt.NDArray[np.uint8],
    quad_pts: npt.NDArray[np.int32],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 6,
    gap_length: int = 4,
) -> None:
    pts2 = quad_pts.reshape(-1, 2)
    n = len(pts2)
    for i in range(n):
        p0 = (float(pts2[i][0]), float(pts2[i][1]))
        p1 = (float(pts2[(i + 1) % n][0]), float(pts2[(i + 1) % n][1]))
        _draw_dashed_line_segment(
            img, p0, p1, color, thickness, dash_length, gap_length
        )


def _put_text_bold(
    img: npt.NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Simulate bold using a 1px offset duplicate."""
    cv2.putText(
        img, text, org, _FONT, font_scale, color, thickness, lineType=cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (org[0] + 1, org[1]),
        _FONT,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _wrap_text(text: str, max_chars: int) -> list[str]:
    if max_chars < 8:
        max_chars = 8
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.rstrip()
        if not raw:
            lines.append("")
            continue
        words = raw.split()
        cur = words[0]
        for w in words[1:]:
            if len(cur) + 1 + len(w) <= max_chars:
                cur = f"{cur} {w}"
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def draw_camera_panel(
    frame: npt.NDArray[np.uint8],
    tracker_state: Mapping[str, TileRecord],
    *,
    warp_to_source_3x3: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Draw the live camera panel with per-tile borders (non–dump-candidate only).

    Normal tiles: thin neutral border. Occluded tiles: dashed, cooler/dimmed border.
    Dump candidates are left to :func:`draw_dump_candidates_camera`.

    When ``warp_to_source_3x3`` is set, ``frame`` is the raw camera image and each
    ``TileRecord.bbox`` (rectified coordinates) is projected to the source plane
    before drawing. When None, bboxes are drawn in the same pixel space as ``frame``.
    """
    _ensure_bgr_uint8(frame)
    out = frame.copy()
    for rec in tracker_state.values():
        if rec.is_dump_candidate:
            continue
        x, y, w, h = rec.bbox
        if w <= 0 or h <= 0:
            continue
        if warp_to_source_3x3 is not None:
            quad = _warp_bbox_corners_to_source(warp_to_source_3x3, rec.bbox)
            if rec.is_occluded:
                _draw_dashed_quad(
                    out, quad, _COLOR_BORDER_OCCLUDED, thickness=1
                )
            else:
                cv2.polylines(
                    out,
                    [quad],
                    isClosed=True,
                    color=_COLOR_BORDER_NEUTRAL,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
        elif rec.is_occluded:
            _draw_dashed_rectangle(
                out, x, y, w, h, _COLOR_BORDER_OCCLUDED, thickness=1
            )
        else:
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), _COLOR_BORDER_NEUTRAL, 1)
    return out


def draw_dump_candidates_camera(
    frame: npt.NDArray[np.uint8],
    tracker_state: Mapping[str, TileRecord],
    *,
    warp_to_source_3x3: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.uint8]:
    """Draw a solid red border around each tile with ``is_dump_candidate``."""
    _ensure_bgr_uint8(frame)
    out = frame.copy()
    for rec in tracker_state.values():
        if not rec.is_dump_candidate:
            continue
        x, y, w, h = rec.bbox
        if w <= 0 or h <= 0:
            continue
        if warp_to_source_3x3 is not None:
            quad = _warp_bbox_corners_to_source(warp_to_source_3x3, rec.bbox)
            cv2.polylines(
                out, [quad], isClosed=True, color=_COLOR_DUMP, thickness=2, lineType=cv2.LINE_AA
            )
        else:
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), _COLOR_DUMP, 2)
    return out


def _grid_dimensions(grid_state: Mapping[tuple[int, int], str]) -> tuple[int, int, int, int]:
    """Inclusive bounds; empty grid → single cell at (0,0)."""
    if not grid_state:
        return 0, 0, 0, 0
    rows = [r for r, _ in grid_state]
    cols = [c for _, c in grid_state]
    return min(rows), max(rows), min(cols), max(cols)


@dataclass(frozen=True)
class GridLayout:
    """Shared pixel layout for :func:`draw_grid_panel` and :func:`draw_dump_candidates_grid`."""

    cell_size: int
    origin_x: int
    origin_y: int
    min_row: int
    min_col: int
    nrows: int
    ncols: int


def _compute_grid_layout(
    grid_state: Mapping[tuple[int, int], str],
    panel_size: tuple[int, int],
) -> GridLayout:
    """Compute square cell size and origin so the grid fits and is centered in the panel."""
    pw, ph = panel_size
    if pw < 1 or ph < 1:
        raise ValueError("panel_size must be positive")

    if not grid_state:
        inner_w = max(1, pw - 2 * _GRID_MARGIN)
        inner_h = max(1, ph - 2 * _GRID_MARGIN)
        cell = min(_MIN_CELL_PX, inner_w, inner_h)
        cell = max(1, cell)
        grid_w = cell
        grid_h = cell
        ox = _GRID_MARGIN + (inner_w - grid_w) // 2
        oy = _GRID_MARGIN + (inner_h - grid_h) // 2
        return GridLayout(
            cell_size=cell,
            origin_x=ox,
            origin_y=oy,
            min_row=0,
            min_col=0,
            nrows=1,
            ncols=1,
        )

    r0, r1, c0, c1 = _grid_dimensions(grid_state)
    nrows = r1 - r0 + 1
    ncols = c1 - c0 + 1

    inner_w = max(1, pw - 2 * _GRID_MARGIN)
    inner_h = max(1, ph - 2 * _GRID_MARGIN)
    cell = min(inner_w // ncols, inner_h // nrows)
    if cell >= _MIN_CELL_PX:
        cell = max(_MIN_CELL_PX, cell)
        if cell * ncols > inner_w or cell * nrows > inner_h:
            cell = min(inner_w // ncols, inner_h // nrows)

    grid_w = cell * ncols
    grid_h = cell * nrows
    ox = _GRID_MARGIN + (inner_w - grid_w) // 2
    oy = _GRID_MARGIN + (inner_h - grid_h) // 2

    return GridLayout(
        cell_size=cell,
        origin_x=ox,
        origin_y=oy,
        min_row=r0,
        min_col=c0,
        nrows=nrows,
        ncols=ncols,
    )


def _cell_fill_and_text(
    letter: str | None,
    cell_occluded: bool,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return (fill_bgr, text_bgr) for empty, normal occupied, or occluded occupied cells."""
    if letter is None or letter == "":
        return _COLOR_CELL_EMPTY, _COLOR_TEXT_LIGHT
    if cell_occluded:
        return _COLOR_CELL_OCCLUDED_GRID, _COLOR_TEXT_OCCLUDED_GRID
    return _COLOR_CELL_NORMAL, _COLOR_TEXT_DARK


def draw_grid_panel(
    grid_state: dict[tuple[int, int], str],
    recommendation: SolverRecommendation | None,
    panel_size: tuple[int, int],
    occluded_cells: Collection[tuple[int, int]] | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Render the digital board into a new image of size ``panel_size`` (width, height).

    Square cells are sized to fit the occupied bounding box (plus empty cells inside
    that rectangle), centered in the panel. Minimum cell size is enforced when space
    allows; the layout shrinks if the board is too large for the panel.

    ``recommendation`` is reserved for API compatibility with the display layer; it is
    not interpreted here.

    ``occluded_cells`` optional set of ``(row, col)`` for board tiles that are occluded
    on the camera; those cells use a dimmed fill and lighter letter color.
    """
    _ = recommendation
    pw, ph = panel_size
    occ = frozenset(occluded_cells) if occluded_cells is not None else frozenset()

    layout = _compute_grid_layout(grid_state, (pw, ph))
    cell = layout.cell_size
    ox, oy = layout.origin_x, layout.origin_y
    r0, c0 = layout.min_row, layout.min_col
    nrows, ncols = layout.nrows, layout.ncols

    panel = np.full((ph, pw, 3), _COLOR_BG_PANEL, dtype=np.uint8)

    font_scale = max(0.35, min(1.2, cell / 48.0))

    for gi in range(nrows):
        for gj in range(ncols):
            r, c = r0 + gi, c0 + gj
            letter = grid_state.get((r, c))
            fill, text_base = _cell_fill_and_text(letter, (r, c) in occ)
            x1 = ox + gj * cell
            y1 = oy + gi * cell
            x2 = x1 + cell
            y2 = y1 + cell
            panel[y1:y2, x1:x2] = fill
            cv2.rectangle(panel, (x1, y1), (x2 - 1, y2 - 1), (80, 80, 80), 1)

            if letter:
                ch = letter.upper()[:1]
                (tw, th), _ = cv2.getTextSize(ch, _FONT, font_scale, 2)
                tx = x1 + (cell - tw) // 2
                ty = y1 + (cell + th) // 2
                _put_text_bold(panel, ch, (tx, ty), font_scale, text_base, thickness=1)

    return panel


def draw_dump_candidates_grid(
    grid_panel: npt.NDArray[np.uint8],
    grid_state: Mapping[tuple[int, int], str],
    dump_letters: Collection[str],
) -> npt.NDArray[np.uint8]:
    """
    Highlight grid cells whose letter is in ``dump_letters`` with a semi-transparent red
    overlay (alpha ~0.45), then redraw letters on top.

    Uses :func:`_compute_grid_layout` with the panel size of ``grid_panel`` so alignment
    matches :func:`draw_grid_panel`.

    If the same letter appears in multiple cells, all matching cells are highlighted.

    Only keys present in ``grid_state`` are considered (mapped board cells); hand tiles
    are not in ``grid_state`` and are never highlighted here.
    """
    _ensure_bgr_uint8(grid_panel)
    if not dump_letters:
        return grid_panel.copy()

    targets = {str(x).upper()[:1] for x in dump_letters if str(x)}
    if not targets:
        return grid_panel.copy()

    ph, pw = grid_panel.shape[:2]
    layout = _compute_grid_layout(dict(grid_state), (pw, ph))
    cell = layout.cell_size
    ox, oy = layout.origin_x, layout.origin_y
    r0, c0 = layout.min_row, layout.min_col
    nrows, ncols = layout.nrows, layout.ncols
    font_scale = max(0.35, min(1.2, cell / 48.0))

    out = grid_panel.copy()

    for (r, c), letter in grid_state.items():
        if not letter:
            continue
        ch = letter.upper()[:1]
        if ch not in targets:
            continue
        gi, gj = r - r0, c - c0
        if not (0 <= gi < nrows and 0 <= gj < ncols):
            continue
        x1 = ox + gj * cell
        y1 = oy + gi * cell
        x2 = x1 + cell
        y2 = y1 + cell
        roi = out[y1:y2, x1:x2]
        rpatch = np.full_like(roi, _COLOR_DUMP, dtype=np.uint8)
        blended = cv2.addWeighted(rpatch, _DUMP_ALPHA, roi, 1.0 - _DUMP_ALPHA, 0)
        out[y1:y2, x1:x2] = blended
        (tw, th), _ = cv2.getTextSize(ch, _FONT, font_scale, 2)
        tx = x1 + (cell - tw) // 2
        ty = y1 + (cell + th) // 2
        _put_text_bold(out, ch, (tx, ty), font_scale, _COLOR_TEXT_DARK, thickness=1)

    return out


def _truncate_with_ellipsis(text: str, max_chars: int) -> str:
    ell = "…"
    if max_chars < len(ell) + 1:
        return ell[:max_chars]
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ell)].rstrip() + ell


def draw_reasoning_panel(
    image: npt.NDArray[np.uint8],
    x: int,
    y: int,
    width: int,
    action_line: str,
    reasoning_body: str,
    expanded: bool,
    show_dump_hint: bool,
    bg_alpha: float = 0.62,
) -> None:
    """
    Mutate ``image`` in-place: draw a dark semi-transparent reasoning block.

    The action line uses a larger font; when ``expanded``, ``reasoning_body`` is
    word-wrapped and drawn below. If ``show_dump_hint``, appends the dump-resolution hint.
    Lines are clipped to the panel height; overflow ends with an ellipsis on the last
    visible line.
    """
    _ensure_bgr_uint8(image)
    h, w = image.shape[:2]
    font_action = 0.55
    font_body = 0.42
    thick_a = 2
    thick_b = 1
    margin = 8
    pad = 6

    max_chars = max(12, width // 7)
    lines: list[str] = [action_line]
    if expanded and reasoning_body.strip():
        lines.extend(_wrap_text(reasoning_body, max_chars))
    if show_dump_hint:
        lines.append("")
        lines.extend(_wrap_text("Any one highlighted tile satisfying the dump will resolve it.", max_chars))

    line_h_a = int(cv2.getTextSize("Ay", _FONT, font_action, thick_a)[0][1] + 6)
    line_h_b = int(cv2.getTextSize("Ay", _FONT, font_body, thick_b)[0][1] + 4)

    max_box_h = h - y
    box_w = min(width, w - x - 1)
    if max_box_h < 20 or box_w < 20:
        return

    inner_h = max_box_h - 2 * pad
    if inner_h < line_h_a:
        return

    n_after_action = max(0, (inner_h - line_h_a) // line_h_b)
    rest = lines[1:]
    if len(rest) <= n_after_action:
        draw_lines = lines
    else:
        draw_lines = [lines[0]] + rest[:n_after_action]
        if n_after_action > 0:
            draw_lines[-1] = _truncate_with_ellipsis(draw_lines[-1], max_chars)
        elif len(rest) > 0:
            draw_lines[0] = _truncate_with_ellipsis(draw_lines[0], max_chars)

    text_h = line_h_a + max(0, len(draw_lines) - 1) * line_h_b
    box_h = min(max_box_h, text_h + 2 * pad)

    y2 = min(h, y + box_h)
    x2 = min(w, x + box_w)
    roi = image[y:y2, x:x2]
    if roi.size == 0:
        return
    overlay = np.full_like(roi, _REASONING_BG, dtype=np.uint8)
    blended = cv2.addWeighted(overlay, bg_alpha, roi, 1.0 - bg_alpha, 0)
    image[y:y2, x:x2] = blended

    cy = y + pad + line_h_a - 4
    cv2.putText(
        image,
        _truncate_with_ellipsis(draw_lines[0], 200),
        (x + margin, cy),
        _FONT,
        font_action,
        (240, 240, 240),
        thick_a,
        lineType=cv2.LINE_AA,
    )
    cy += line_h_b
    max_chars_draw = max(8, max_chars + 20)
    for ln in draw_lines[1:]:
        if cy >= y2 - 4:
            break
        cv2.putText(
            image,
            _truncate_with_ellipsis(ln, max_chars_draw),
            (x + margin, cy),
            _FONT,
            font_body,
            (210, 210, 210),
            thick_b,
            lineType=cv2.LINE_AA,
        )
        cy += line_h_b


def draw_game_over_overlay(
    image: npt.NDArray[np.uint8],
    reasoning_text: str,
) -> None:
    """Mutate ``image``: full-frame dim overlay, centered GAME OVER, reasoning below."""
    _ensure_bgr_uint8(image)
    h, w = image.shape[:2]
    dim = image.copy()
    dim = (dim * 0.45).astype(np.uint8)
    image[:, :] = dim

    title = "GAME OVER"
    font_t = 1.35
    thick = 3
    (tw, th), _baseline = cv2.getTextSize(title, _FONT, font_t, thick)
    tx = (w - tw) // 2
    ty = (h + th) // 2
    _put_text_bold(image, title, (tx, ty), font_t, (255, 255, 255), thickness=thick)

    body_lines = _wrap_text(reasoning_text, max(16, w // 10))
    font_b = 0.5
    cy = ty + th + 20
    for ln in body_lines[:24]:
        if cy > h - 20:
            break
        (bw, bh), _ = cv2.getTextSize(ln, _FONT, font_b, 1)
        cv2.putText(
            image,
            ln,
            ((w - bw) // 2, cy),
            _FONT,
            font_b,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )
        cy += int(bh * 1.35)


def resize_to_height(
    frame: npt.NDArray[np.uint8],
    target_h: int,
) -> npt.NDArray[np.uint8]:
    """Resize ``frame`` proportionally to ``target_h`` (minimum 1 px wide/tall)."""
    _ensure_bgr_uint8(frame)
    if target_h < 1:
        raise ValueError("target_h must be positive")
    fh, fw = frame.shape[:2]
    if fh < 1:
        return frame.copy()
    scale = target_h / fh
    tw = max(1, int(round(fw * scale)))
    th = target_h
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def hstack_panels(left: npt.NDArray[np.uint8], right: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Concatenate two BGR images horizontally with exactly equal height.

    The shorter panel is extended with black rows **below** so both panels share the
    same top edge (no vertical centering drift).
    """
    _ensure_bgr_uint8(left)
    _ensure_bgr_uint8(right)
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    target_h = max(lh, rh)
    if lh < target_h:
        pad = np.zeros((target_h - lh, lw, 3), dtype=np.uint8)
        left = np.vstack([left, pad])
    if rh < target_h:
        pad = np.zeros((target_h - rh, rw, 3), dtype=np.uint8)
        right = np.vstack([right, pad])
    return np.hstack([left, right])
