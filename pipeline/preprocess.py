"""Подготовка: вычитание бумаги, поиск пятна и геометрия капли."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


DETECT_MAX_SIDE = 1400


@dataclass
class DropGeometry:
    center: tuple[float, float]
    radius: float
    mask: np.ndarray


@dataclass
class DropCandidate:
    index: int
    mask: np.ndarray
    geometry: DropGeometry
    score: float


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def resize_for_detection(rgb: np.ndarray, max_side: int = DETECT_MAX_SIDE) -> tuple[np.ndarray, float]:
    """Уменьшает изображение для быстрого поиска капель."""
    h, w = rgb.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rgb, scale


def detect_paper_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    chroma = np.sqrt(
        (lab[:, :, 1].astype(np.float32) - 128.0) ** 2
        + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2
    )

    image_area = rgb.shape[0] * rgb.shape[1]
    best_mask: np.ndarray | None = None
    best_score = -1.0

    for l_thresh in (185, 180, 175):
        paper = (l_channel > l_thresh) & (chroma < 32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        paper_u8 = (paper.astype(np.uint8) * 255)
        paper_u8 = cv2.morphologyEx(paper_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        paper_u8 = cv2.morphologyEx(paper_u8, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(paper_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.04:
                continue
            if area > image_area * 0.78:
                continue

            candidate = np.zeros_like(paper_u8, dtype=np.uint8)
            cv2.drawContours(candidate, [contour], -1, 255, thickness=cv2.FILLED)
            mask = candidate > 0

            mean_l = float(np.mean(l_channel[mask]))
            mean_chroma = float(np.mean(chroma[mask]))
            if mean_l < 170 or mean_chroma > 35:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            fill_ratio = area / max(bw * bh, 1)
            area_frac = area / image_area
            touches_border = x <= 2 or y <= 2
            touches_border |= x + bw >= rgb.shape[1] - 3
            touches_border |= y + bh >= rgb.shape[0] - 3

            score = mean_l * 0.55 + fill_ratio * 40.0 + area_frac * 25.0
            score -= mean_chroma * 0.35
            if touches_border and area_frac > 0.45:
                score -= 35.0

            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is not None:
            return best_mask

    paper = (l_channel > 180) & (chroma < 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    paper_u8 = (paper.astype(np.uint8) * 255)
    paper_u8 = cv2.morphologyEx(paper_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    paper_u8 = cv2.morphologyEx(paper_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(paper_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return paper

    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in ranked:
        if cv2.contourArea(contour) <= image_area * 0.78:
            paper_mask = np.zeros_like(paper_u8, dtype=np.uint8)
            cv2.drawContours(paper_mask, [contour], -1, 255, thickness=cv2.FILLED)
            return paper_mask > 0

    largest = ranked[0]
    paper_mask = np.zeros_like(paper_u8, dtype=np.uint8)
    cv2.drawContours(paper_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return paper_mask > 0


def estimate_paper_color(rgb: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    if np.any(paper_mask):
        pixels = rgb[paper_mask]
        if pixels.size >= 30:
            return np.median(pixels, axis=0)
    return np.median(rgb.reshape(-1, 3), axis=0)


def compute_stain_map(rgb: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]

    blur_l = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=28, sigmaY=28)
    local_dark = np.clip(blur_l - l_channel, 0, None)

    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    blur_a = cv2.GaussianBlur(a_channel, (0, 0), sigmaX=28, sigmaY=28)
    blur_b = cv2.GaussianBlur(b_channel, (0, 0), sigmaX=28, sigmaY=28)
    local_warm = np.clip((a_channel - blur_a) * 0.45 + (b_channel - blur_b) * 0.75, 0, None)
    local_stain = local_dark * 1.7 + local_warm * 1.1
    local_stain = cv2.GaussianBlur(local_stain, (5, 5), 0)

    paper_frac = float(np.count_nonzero(paper_mask)) / paper_mask.size
    if paper_frac < 0.75:
        paper = estimate_paper_color(rgb, paper_mask)
        paper_lab = cv2.cvtColor(np.uint8([[paper]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        delta_l = np.clip(paper_lab[0] - l_channel, 0, None)
        delta_a = np.clip(lab[:, :, 1] - paper_lab[1], 0, None)
        delta_b = np.clip(lab[:, :, 2] - paper_lab[2], 0, None)
        global_stain = delta_l * 1.4 + delta_a * 0.35 + delta_b * 0.55
        global_stain = cv2.GaussianBlur(global_stain, (5, 5), 0)
        stain = np.maximum(global_stain, local_stain)
    else:
        stain = local_stain

    stain[~paper_mask] = 0.0
    return stain.astype(np.float32)


def find_drop_geometry(stain: np.ndarray, mask: np.ndarray) -> DropGeometry | None:
    if not np.any(mask):
        return None

    mask_bool = mask > 0
    weights = stain * mask_bool
    if float(np.sum(weights)) > 0:
        ys, xs = np.indices(stain.shape)
        total = float(np.sum(weights))
        cx = float(np.sum(xs * weights) / total)
        cy = float(np.sum(ys * weights) / total)
    else:
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] <= 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

    ys, xs = np.where(mask_bool)
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radius = float(np.percentile(distances, 90))
    if radius < 8:
        return None

    tight = np.zeros(mask_bool.shape, dtype=np.uint8)
    cv2.circle(tight, (int(round(cx)), int(round(cy))), max(8, int(round(radius))), 255, -1)
    tight_mask = (tight > 0) & mask_bool

    return DropGeometry(center=(cx, cy), radius=radius, mask=tight_mask)


def _dilate_paper_mask(paper_mask: np.ndarray, margin_px: int = 14) -> np.ndarray:
    if margin_px <= 0:
        return paper_mask
    k = margin_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(paper_mask.astype(np.uint8), kernel, iterations=1)
    return dilated > 0


def _inner_paper_mask(paper_mask: np.ndarray, margin_frac: float = 0.08) -> np.ndarray:
    """Внутренняя зона бумаги — без краёв, где часто ложные пики окраски."""
    x0, y0, x1, y1 = _paper_bounds(paper_mask)
    paper_w = x1 - x0 + 1
    paper_h = y1 - y0 + 1
    mx = max(4, int(paper_w * margin_frac))
    my = max(4, int(paper_h * margin_frac))
    inner = np.zeros_like(paper_mask, dtype=bool)
    inner[y0 + my : y1 - my + 1, x0 + mx : x1 - mx + 1] = paper_mask[
        y0 + my : y1 - my + 1, x0 + mx : x1 - mx + 1
    ]
    if np.count_nonzero(inner) < 100:
        return paper_mask
    return inner


def _paper_bounds(paper_mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(paper_mask)
    if ys.size == 0:
        h, w = paper_mask.shape
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _max_drop_radius(paper_mask: np.ndarray) -> float:
    x0, y0, x1, y1 = _paper_bounds(paper_mask)
    paper_w = x1 - x0 + 1
    paper_h = y1 - y0 + 1
    return float(max(20, min(paper_w, paper_h) * 0.16))


def _min_drop_radius(paper_mask: np.ndarray) -> float:
    x0, y0, x1, y1 = _paper_bounds(paper_mask)
    short_side = min(x1 - x0 + 1, y1 - y0 + 1)
    return float(max(8, short_side * 0.018))


def _resolve_stain_point(
    cx: float,
    cy: float,
    stain: np.ndarray,
    paper_mask: np.ndarray,
    *,
    from_click: bool = False,
) -> tuple[float, float, float] | None:
    """Находит лучшую точку окраски рядом с кликом или пиком."""
    h, w = stain.shape
    active_paper = _dilate_paper_mask(paper_mask, 18 if from_click else 0)
    search_radius = 110 if from_click else 40
    min_stain = 0.08 if from_click else 0.18

    work = stain.copy()
    work[~active_paper] = 0.0
    if float(work.max()) < min_stain:
        return None

    cx_i = int(np.clip(round(cx), 0, w - 1))
    cy_i = int(np.clip(round(cy), 0, h - 1))

    x0 = max(0, cx_i - search_radius)
    x1 = min(w, cx_i + search_radius + 1)
    y0 = max(0, cy_i - search_radius)
    y1 = min(h, cy_i + search_radius + 1)

    window = work[y0:y1, x0:x1]
    valid = window >= min_stain
    if not np.any(valid):
        if not from_click:
            return None
        ys_all, xs_all = np.where(work >= min_stain)
        if ys_all.size == 0:
            return None
        dist_penalty = 0.0015
        scores = work[ys_all, xs_all] - dist_penalty * (
            (xs_all - cx) ** 2 + (ys_all - cy) ** 2
        )
        best = int(np.argmax(scores))
        px = float(xs_all[best])
        py = float(ys_all[best])
        return px, py, float(work[int(py), int(px)])

    ys, xs = np.where(valid)
    dist_penalty = 0.0025 if from_click else 0.001
    scores = window[ys, xs] - dist_penalty * (
        (xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2
    )
    best = int(np.argmax(scores))
    px = float(x0 + xs[best])
    py = float(y0 + ys[best])
    return px, py, float(work[int(py), int(px)])


def _mask_circularity(mask: np.ndarray) -> float:
    contour = None
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 5:
        return 0.0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1:
        return 0.0
    return float(4 * np.pi * area / (perimeter * perimeter))


def _radial_grow_mask(
    cx: float,
    cy: float,
    peak: float,
    stain: np.ndarray,
    paper_mask: np.ndarray,
    *,
    max_radius: float,
    stop_ratio: float = 0.30,
) -> np.ndarray | None:
    """Рост круга от центра, пока кромка остаётся окрашенной."""
    h, w = stain.shape
    cx_i = int(np.clip(round(cx), 0, w - 1))
    cy_i = int(np.clip(round(cy), 0, h - 1))
    if peak < 0.08:
        return None

    best_r = 0
    best_count = 0
    stop_level = peak * stop_ratio

    for radius in range(10, int(max_radius) + 1, 2):
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring, (cx_i, cy_i), radius, 255, 2)
        ring_region = (ring > 0) & paper_mask
        if not np.any(ring_region):
            break
        edge_stain = float(np.mean(stain[ring_region]))
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(filled, (cx_i, cy_i), radius, 255, -1)
        filled_region = (filled > 0) & paper_mask
        if edge_stain < stop_level and radius > 12:
            break
        best_r = radius
        best_count = int(np.count_nonzero(filled_region))

    if best_r < 10:
        best_r = int(min(max_radius, max(14, peak * 1.5 + 10)))

    out = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(out, (cx_i, cy_i), best_r, 255, -1)
    return (out > 0) & paper_mask


def _flood_stain_mask(
    cx: float,
    cy: float,
    peak: float,
    stain: np.ndarray,
    paper_mask: np.ndarray,
) -> np.ndarray | None:
    """Заливка связной области окраски от точки (OpenCV floodFill)."""
    h, w = stain.shape
    cx_i = int(np.clip(round(cx), 0, w - 1))
    cy_i = int(np.clip(round(cy), 0, h - 1))
    if peak < 0.08:
        return None

    paper_vals = stain[paper_mask]
    if paper_vals.size == 0:
        return None
    p95 = float(np.percentile(paper_vals, 95))
    scale = max(p95, peak, 1.0)

    norm = np.zeros((h, w), dtype=np.uint8)
    norm[paper_mask] = np.clip(stain[paper_mask] / scale * 255.0, 0, 255).astype(np.uint8)

    for tolerance in (28, 36, 44):
        work = norm.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        try:
            cv2.floodFill(work, ff_mask, (cx_i, cy_i), 255, tolerance, tolerance, flags)
        except cv2.error:
            continue
        blob = ff_mask[1:-1, 1:-1] > 0
        blob &= paper_mask
        if np.count_nonzero(blob) >= 60:
            return blob
    return None


def _stain_blob_mask(
    cx: float,
    cy: float,
    peak: float,
    stain: np.ndarray,
    paper_mask: np.ndarray,
    *,
    max_radius: float,
    from_click: bool = False,
) -> np.ndarray | None:
    """Выделяет одну каплю как связную область окраски вокруг точки."""
    h, w = stain.shape
    cx_i = int(np.clip(round(cx), 0, w - 1))
    cy_i = int(np.clip(round(cy), 0, h - 1))
    active_paper = _dilate_paper_mask(paper_mask, 16 if from_click else 0)

    if peak < (0.08 if from_click else 0.15):
        return None

    paper_pixels = max(int(np.count_nonzero(paper_mask)), 1)
    max_blob_area = int(paper_pixels * (0.14 if from_click else 0.10))
    min_blob_area = 50 if from_click else 120

    ratios = (0.40, 0.32, 0.26, 0.20, 0.16) if from_click else (0.42, 0.34, 0.28, 0.22)

    for ratio in ratios:
        threshold = peak * ratio
        binary = ((stain >= threshold) & active_paper).astype(np.uint8)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

        num_labels, labels = cv2.connectedComponents(binary)
        label = labels[cy_i, cx_i]
        if label == 0:
            continue

        blob = labels == label
        area = int(np.count_nonzero(blob))
        if area < min_blob_area or area > max_blob_area:
            continue

        geometry = find_drop_geometry(stain, blob)
        if geometry is None or geometry.radius > max_radius:
            continue
        return geometry.mask

    for extractor in (_flood_stain_mask, _radial_grow_mask):
        if extractor is _flood_stain_mask:
            blob = _flood_stain_mask(cx, cy, peak, stain, active_paper)
        else:
            blob = _radial_grow_mask(
                cx, cy, peak, stain, active_paper, max_radius=max_radius
            )
        if blob is None:
            continue
        geometry = find_drop_geometry(stain, blob)
        if geometry is None:
            continue
        if geometry.radius <= max_radius and np.count_nonzero(geometry.mask) >= min_blob_area:
            return geometry.mask

    return None


def _has_local_stain_peak(
    stain: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> bool:
    """Пятно должно быть заметно ярче (по окраске) окружения."""
    h, w = stain.shape
    cx_i = int(np.clip(round(center[0]), 0, w - 1))
    cy_i = int(np.clip(round(center[1]), 0, h - 1))
    center_val = float(stain[cy_i, cx_i])

    outer_r = int(max(radius * 2.2, radius + 18))
    inner_r = int(max(radius * 1.15, radius + 6))
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ring, (cx_i, cy_i), outer_r, 255, -1)
    cv2.circle(ring, (cx_i, cy_i), inner_r, 0, -1)
    ring_vals = stain[ring > 0]
    if ring_vals.size < 20:
        return center_val > 0.5
    return center_val >= float(np.median(ring_vals)) * 1.12


def _is_valid_drop_candidate(
    candidate: DropCandidate,
    stain: np.ndarray,
    paper_mask: np.ndarray,
    *,
    from_click: bool = False,
) -> bool:
    geometry = candidate.geometry
    if geometry.radius < _min_drop_radius(paper_mask):
        return False
    if geometry.radius > _max_drop_radius(paper_mask) * (1.15 if from_click else 1.0):
        return False

    cx_i = int(np.clip(round(geometry.center[0]), 0, stain.shape[1] - 1))
    cy_i = int(np.clip(round(geometry.center[1]), 0, stain.shape[0] - 1))
    center_stain = float(np.percentile(stain[candidate.mask], 92))

    paper_vals = stain[paper_mask]
    if paper_vals.size == 0:
        return False
    paper_max = float(np.max(paper_vals))
    paper_frac = float(np.count_nonzero(paper_mask)) / paper_mask.size
    min_center = paper_max * (
        0.12 if from_click else (0.18 if paper_frac > 0.72 else 0.30)
    )
    if center_stain < min_center:
        return False

    mean_stain = float(np.mean(stain[candidate.mask]))
    if mean_stain < paper_max * (0.10 if from_click else (0.15 if paper_frac > 0.72 else 0.22)):
        return False

    if _mask_circularity(candidate.mask) < (0.18 if from_click else 0.30):
        return False

    if not _has_local_stain_peak(stain, candidate.geometry.center, candidate.geometry.radius):
        return False

    if not from_click:
        x0, y0, x1, y1 = _paper_bounds(paper_mask)
        paper_w = x1 - x0 + 1
        paper_h = y1 - y0 + 1
        margin_x = paper_w * 0.05
        margin_y = paper_h * 0.05
        cx, cy = geometry.center
        if cx < x0 + margin_x or cx > x1 - margin_x:
            return False
        if cy < y0 + margin_y or cy > y1 - margin_y:
            return False
        paper_frac = float(np.count_nonzero(paper_mask)) / paper_mask.size
        if paper_frac < 0.72:
            margin_x = paper_w * 0.07
            margin_y = paper_h * 0.07
            if cx < x0 + margin_x or cx > x1 - margin_x:
                return False
            if cy < y0 + margin_y or cy > y1 - margin_y:
                return False

    return True


def _expand_small_drop(
    geometry: DropGeometry,
    stain: np.ndarray,
    paper_mask: np.ndarray,
    peak: float,
) -> DropGeometry:
    """Если пик окраски сильный, но маска крошечная — расширяем до типичного размера капли."""
    min_r = _min_drop_radius(paper_mask)
    if geometry.radius >= min_r:
        return geometry

    paper_vals = stain[paper_mask]
    paper_max = float(np.max(paper_vals)) if paper_vals.size else peak
    if peak < paper_max * 0.35:
        return geometry

    h, w = stain.shape
    cx_i = int(np.clip(round(geometry.center[0]), 0, w - 1))
    cy_i = int(np.clip(round(geometry.center[1]), 0, h - 1))
    target_r = int(min(_max_drop_radius(paper_mask), max(min_r, min_r * 1.4)))

    expanded = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(expanded, (cx_i, cy_i), target_r, 255, -1)
    expanded_mask = (expanded > 0) & paper_mask
    return find_drop_geometry(stain, expanded_mask) or geometry


def build_drop_at_point(
    cx: float,
    cy: float,
    stain: np.ndarray,
    paper_mask: np.ndarray,
    *,
    from_click: bool = False,
) -> DropCandidate | None:
    """Строит каплю вокруг точки (клик пользователя или пик окрашивания)."""
    resolved = _resolve_stain_point(cx, cy, stain, paper_mask, from_click=from_click)
    if resolved is None:
        return None
    cx, cy, peak = resolved

    max_radius = _max_drop_radius(paper_mask)
    blob_mask = _stain_blob_mask(
        cx, cy, peak, stain, paper_mask, max_radius=max_radius, from_click=from_click
    )
    if blob_mask is None:
        return None

    geometry = find_drop_geometry(stain, blob_mask)
    if geometry is None:
        return None
    geometry = _expand_small_drop(geometry, stain, paper_mask, peak)

    mean_stain = float(np.mean(stain[geometry.mask]))
    compactness = mean_stain / max(geometry.radius, 1.0)
    score = min(mean_stain, 30.0) * 2.5 + compactness * 18.0
    candidate = DropCandidate(index=0, mask=geometry.mask, geometry=geometry, score=score)
    if not _is_valid_drop_candidate(candidate, stain, paper_mask, from_click=from_click):
        return None
    return candidate


def _detection_mask(paper_mask: np.ndarray) -> np.ndarray:
    """Зона поиска капель. При «вся картинка = бумага» отрезаем только рамку кадра."""
    mask = paper_mask.copy()
    paper_frac = float(np.count_nonzero(paper_mask)) / paper_mask.size
    if paper_frac > 0.72:
        h, w = mask.shape
        mx = max(4, int(w * 0.025))
        my = max(4, int(h * 0.025))
        border = np.zeros_like(mask, dtype=bool)
        border[:my, :] = True
        border[-my:, :] = True
        border[:, :mx] = True
        border[:, -mx:] = True
        mask[border] = False
        return mask
    return _inner_paper_mask(mask, margin_frac=0.04)


def _find_stain_peak_positions(
    stain: np.ndarray,
    paper_mask: np.ndarray,
    *,
    min_distance: int = 36,
    min_stain: float | None = None,
    max_peaks: int = 6,
) -> list[tuple[float, float]]:
    """Локальные максимумы окраски во внутренней зоне бумаги."""
    paper_frac = float(np.count_nonzero(paper_mask)) / paper_mask.size
    search_mask = _detection_mask(paper_mask)
    work = stain.copy()
    work[~search_mask] = 0.0
    paper_max = float(work.max())
    if paper_max <= 0:
        return []

    stain_floor = paper_max * (0.50 if paper_frac > 0.72 else 0.42) if min_stain is None else min_stain
    stain_floor = max(stain_floor, 0.35)

    x0, y0, x1, y1 = _paper_bounds(search_mask)
    paper_w = x1 - x0 + 1
    paper_h = y1 - y0 + 1
    min_distance = max(min_distance, int(min(paper_w, paper_h) * 0.11))

    k = max(17, min_distance | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(work, kernel)
    peaks = (work >= dilated - 1e-4) & (work >= stain_floor)
    ys, xs = np.where(peaks)
    if ys.size == 0:
        return []

    order = np.argsort(work[ys, xs])[::-1]
    kept: list[tuple[float, float]] = []
    for idx in order:
        py = float(ys[idx])
        px = float(xs[idx])

        too_close = False
        for kx, ky in kept:
            if np.hypot(px - kx, py - ky) < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append((px, py))
        if len(kept) >= max_peaks:
            break
    return kept


def _dedupe_candidates(candidates: list[DropCandidate]) -> list[DropCandidate]:
    kept: list[DropCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        cx, cy = candidate.geometry.center
        duplicate = False
        for existing in kept:
            ex, ey = existing.geometry.center
            min_sep = max(candidate.geometry.radius, existing.geometry.radius) * 0.65
            if np.hypot(cx - ex, cy - ey) < min_sep:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def find_drop_candidates(
    stain: np.ndarray,
    rgb: np.ndarray,
    paper_mask: np.ndarray,
    *,
    max_candidates: int = 4,
) -> list[DropCandidate]:
    """Ищет капли по локальным максимумам окраски на бумаге."""
    del rgb  # совместимость сигнатуры

    peaks = _find_stain_peak_positions(stain, paper_mask)
    raw: list[DropCandidate] = []
    for cx, cy in peaks:
        candidate = build_drop_at_point(cx, cy, stain, paper_mask, from_click=False)
        if candidate is not None:
            raw.append(candidate)

    if not raw:
        inner = _inner_paper_mask(paper_mask)
        work = stain.copy()
        work[~inner] = 0.0
        if float(work.max()) >= 0.25:
            cy, cx = np.unravel_index(int(np.argmax(work)), work.shape)
            fallback = build_drop_at_point(float(cx), float(cy), stain, paper_mask, from_click=False)
            if fallback is not None:
                raw.append(fallback)

    kept = _dedupe_candidates(raw)
    kept.sort(key=lambda c: c.score, reverse=True)
    kept = kept[:max_candidates]
    for i, candidate in enumerate(kept):
        candidate.index = i
    return kept


def find_drop_candidates_full_image(rgb: np.ndarray) -> tuple[list[DropCandidate], float]:
    """Пики ищутся на уменьшенной копии, маска строится на полном разрешении."""
    detect_rgb, scale = resize_for_detection(rgb)
    paper_mask = detect_paper_mask(detect_rgb)
    stain = compute_stain_map(detect_rgb, paper_mask)

    x0, y0, x1, y1 = _paper_bounds(paper_mask)
    paper_short = min(x1 - x0 + 1, y1 - y0 + 1)
    min_distance = max(28, int(paper_short * 0.11 / max(scale, 1e-6)))
    peaks = _find_stain_peak_positions(stain, paper_mask, min_distance=min_distance)

    inv_scale = 1.0 / scale
    raw: list[DropCandidate] = []
    for cx, cy in peaks:
        candidate = build_drop_at_point_full(cx * inv_scale, cy * inv_scale, rgb, from_click=False)
        if candidate is not None:
            raw.append(candidate)

    if not raw:
        candidates = find_drop_candidates(stain, detect_rgb, paper_mask)
        for candidate in candidates:
            full = build_drop_at_point_full(
                candidate.geometry.center[0] * inv_scale,
                candidate.geometry.center[1] * inv_scale,
                rgb,
                from_click=False,
            )
            if full is not None:
                raw.append(full)

    kept = _dedupe_candidates(raw)
    kept.sort(key=lambda c: c.score, reverse=True)
    kept = kept[:4]
    for i, candidate in enumerate(kept):
        candidate.index = i
    return kept, scale


def _rasterize_mask(center: tuple[float, float], radius: float, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, thickness=cv2.FILLED)
    return mask > 0


def build_drop_at_point_full(
    cx: float,
    cy: float,
    rgb: np.ndarray,
    *,
    from_click: bool = True,
) -> DropCandidate | None:
    """Капля вокруг точки на полном изображении."""
    paper_mask = detect_paper_mask(rgb)
    stain = compute_stain_map(rgb, paper_mask)
    candidate = build_drop_at_point(cx, cy, stain, paper_mask, from_click=from_click)
    if candidate is None:
        return None
    candidate.index = 0
    return candidate


def build_drop_from_circle(
    cx: float,
    cy: float,
    radius: float,
    rgb: np.ndarray,
) -> DropCandidate:
    """Ручная круглая область — центр и радиус задаёт пользователь."""
    h, w = rgb.shape[:2]
    radius = float(np.clip(radius, 8.0, min(h, w) * 0.45))
    cx = float(np.clip(cx, 0.0, w - 1.0))
    cy = float(np.clip(cy, 0.0, h - 1.0))

    mask = _rasterize_mask((cx, cy), radius, (h, w))
    paper_mask = detect_paper_mask(rgb)
    stain = compute_stain_map(rgb, paper_mask)
    geometry = find_drop_geometry(stain, mask)
    if geometry is None:
        geometry = DropGeometry(center=(cx, cy), radius=radius, mask=mask)

    mean_stain = float(np.mean(stain[geometry.mask])) if np.any(geometry.mask) else 0.0
    score = mean_stain * 2.0 + geometry.radius
    return DropCandidate(index=0, mask=geometry.mask, geometry=geometry, score=score)


def crop_drop(
    rgb: np.ndarray,
    stain: np.ndarray,
    mask: np.ndarray,
    *,
    padding_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return rgb, stain, mask

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    pad_x = min(int(w * padding_ratio), max(12, w // 6))
    pad_y = min(int(h * padding_ratio), max(12, h // 6))

    height, width = stain.shape[:2]
    xa = max(0, x0 - pad_x)
    ya = max(0, y0 - pad_y)
    xb = min(width, x1 + pad_x + 1)
    yb = min(height, y1 + pad_y + 1)

    crop_mask = mask[ya:yb, xa:xb]
    return rgb[ya:yb, xa:xb], stain[ya:yb, xa:xb], crop_mask
