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

    paper = (l_channel > 180) & (chroma < 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    paper_u8 = (paper.astype(np.uint8) * 255)
    paper_u8 = cv2.morphologyEx(paper_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    paper_u8 = cv2.morphologyEx(paper_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(paper_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return paper

    largest = max(contours, key=cv2.contourArea)
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
    paper = estimate_paper_color(rgb, paper_mask)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    paper_lab = cv2.cvtColor(np.uint8([[paper]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

    delta_l = np.clip(paper_lab[0] - lab[:, :, 0], 0, None)
    delta_a = np.clip(lab[:, :, 1] - paper_lab[1], 0, None)
    delta_b = np.clip(lab[:, :, 2] - paper_lab[2], 0, None)

    stain = delta_l * 1.4 + delta_a * 0.35 + delta_b * 0.55
    stain = cv2.GaussianBlur(stain, (5, 5), 0)
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
    radius = float(np.percentile(distances, 97))
    if radius < 8:
        return None

    return DropGeometry(center=(cx, cy), radius=radius, mask=mask_bool)


def build_drop_at_point(
    cx: float,
    cy: float,
    stain: np.ndarray,
    paper_mask: np.ndarray,
) -> DropCandidate | None:
    """Строит каплю вокруг точки (клик пользователя или пик окрашивания)."""
    h, w = stain.shape
    cx_i = int(np.clip(cx, 0, w - 1))
    cy_i = int(np.clip(cy, 0, h - 1))
    if not paper_mask[cy_i, cx_i]:
        return None

    peak = float(stain[cy_i, cx_i])
    if peak < 0.2:
        return None

    max_r = min(h, w) // 2
    best_r = 0
    for radius in range(16, max_r, 6):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx_i, cy_i), radius, 255, thickness=cv2.FILLED)
        region = (mask > 0) & paper_mask
        if np.count_nonzero(region) == 0:
            break
        mean_stain = float(np.mean(stain[region]))
        if mean_stain < peak * 0.22:
            break
        best_r = radius

    if best_r < 16:
        best_r = 24
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx_i, cy_i), best_r, 255, thickness=cv2.FILLED)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx_i, cy_i), best_r, 255, thickness=cv2.FILLED)

    geometry = find_drop_geometry(stain, mask)
    if geometry is None:
        return None

    mean_stain = float(np.mean(stain[(mask > 0) & paper_mask]))
    score = min(mean_stain, 30.0) * 2.0 + geometry.radius * 0.8
    return DropCandidate(index=0, mask=(mask > 0), geometry=geometry, score=score)


def _scale_candidate(candidate: DropCandidate, inv_scale: float) -> DropCandidate:
    if inv_scale == 1.0:
        return candidate

    mask = cv2.resize(
        candidate.mask.astype(np.uint8) * 255,
        (
            int(candidate.mask.shape[1] * inv_scale),
            int(candidate.mask.shape[0] * inv_scale),
        ),
        interpolation=cv2.INTER_NEAREST,
    ) > 0

    cx, cy = candidate.geometry.center
    geometry = DropGeometry(
        center=(cx * inv_scale, cy * inv_scale),
        radius=candidate.geometry.radius * inv_scale,
        mask=mask,
    )
    return DropCandidate(index=candidate.index, mask=mask, geometry=geometry, score=candidate.score)


def find_drop_candidates(
    stain: np.ndarray,
    rgb: np.ndarray,
    paper_mask: np.ndarray,
    *,
    max_candidates: int = 4,
) -> list[DropCandidate]:
    """
    Быстрый поиск крупных капель: делит бумагу на полосы и ищет пик в каждой.
    Без Hough/watershed — не подвисает на больших фото.
    """
    ys, xs = np.where(paper_mask)
    if ys.size == 0:
        return []

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    paper_h = y1 - y0 + 1

    if paper_h > 500:
        bands = (
            (y0, y0 + paper_h // 3),
            (y0 + paper_h // 3, y0 + 2 * paper_h // 3),
            (y0 + 2 * paper_h // 3, y1 + 1),
        )
    else:
        mid = (y0 + y1) // 2
        bands = ((y0, mid), (mid, y1 + 1))

    raw: list[DropCandidate] = []
    image_area = stain.shape[0] * stain.shape[1]

    for ya, yb in bands:
        region = np.zeros_like(paper_mask, dtype=bool)
        region[ya:yb, x0 : x1 + 1] = paper_mask[ya:yb, x0 : x1 + 1]
        region_stain = stain.copy()
        region_stain[~region] = 0.0
        if float(region_stain.max()) < 0.25:
            continue

        cy, cx = np.unravel_index(int(np.argmax(region_stain)), region_stain.shape)
        candidate = build_drop_at_point(float(cx), float(cy), stain, paper_mask)
        if candidate is None:
            continue
        if np.count_nonzero(candidate.mask) < image_area * 0.008:
            continue
        raw.append(candidate)

    if not raw:
        cy, cx = np.unravel_index(int(np.argmax(stain * paper_mask)), stain.shape)
        fallback = build_drop_at_point(float(cx), float(cy), stain, paper_mask)
        if fallback is not None:
            raw.append(fallback)

    kept: list[DropCandidate] = []
    for candidate in sorted(raw, key=lambda c: c.score, reverse=True):
        cx, cy = candidate.geometry.center
        duplicate = False
        for existing in kept:
            ex, ey = existing.geometry.center
            if np.hypot(cx - ex, cy - ey) < max(candidate.geometry.radius, existing.geometry.radius) * 0.7:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=lambda c: (c.geometry.radius, c.score), reverse=True)
    kept = kept[:max_candidates]
    for i, candidate in enumerate(kept):
        candidate.index = i
    return kept


def find_drop_candidates_full_image(rgb: np.ndarray) -> tuple[list[DropCandidate], float]:
    """Поиск на уменьшенной копии, координаты возвращаются в полном масштабе."""
    detect_rgb, scale = resize_for_detection(rgb)
    inv_scale = 1.0 / scale

    paper_mask = detect_paper_mask(detect_rgb)
    stain = compute_stain_map(detect_rgb, paper_mask)
    candidates = find_drop_candidates(stain, detect_rgb, paper_mask)

    if inv_scale != 1.0:
        candidates = [_scale_candidate(c, inv_scale) for c in candidates]
        for c in candidates:
            c.mask = _rasterize_mask(c.geometry.center, c.geometry.radius, rgb.shape[:2])

    return candidates, scale


def _rasterize_mask(center: tuple[float, float], radius: float, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, thickness=cv2.FILLED)
    return mask > 0


def build_drop_at_point_full(
    cx: float,
    cy: float,
    rgb: np.ndarray,
) -> DropCandidate | None:
    """Капля вокруг точки клика на полном изображении."""
    paper_mask = detect_paper_mask(rgb)
    stain = compute_stain_map(rgb, paper_mask)
    candidate = build_drop_at_point(cx, cy, stain, paper_mask)
    if candidate is None:
        return None
    candidate.index = 0
    return candidate


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
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    height, width = stain.shape[:2]
    xa = max(0, x0 - pad_x)
    ya = max(0, y0 - pad_y)
    xb = min(width, x1 + pad_x + 1)
    yb = min(height, y1 + pad_y + 1)

    crop_mask = mask[ya:yb, xa:xb]
    return rgb[ya:yb, xa:xb], stain[ya:yb, xa:xb], crop_mask
