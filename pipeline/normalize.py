"""Нормализация контраста внутри пятна и инверсия."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_stain(
    stain: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "clahe",
    clip_limit: float = 2.5,
) -> np.ndarray:
    """Растягивает контраст только внутри капли, подавляет шум бумаги."""
    result = np.zeros_like(stain, dtype=np.float32)
    values = stain[mask > 0]
    if values.size == 0:
        return result.astype(np.uint8)

    p5 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    if p95 <= p5:
        p95 = p5 + 1.0

    stretched = np.clip((stain - p5) / (p95 - p5), 0, 1)
    stretched = (stretched * 255).astype(np.uint8)

    if method == "equalize":
        enhanced = cv2.equalizeHist(stretched)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(stretched)

    enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    result[mask > 0] = enhanced[mask > 0].astype(np.float32)
    return result.astype(np.uint8)


def invert_within_drop(enhanced: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Инверсия: тёмное ядро -> яркое; фон остаётся чёрным."""
    inverted = np.zeros_like(enhanced)
    inverted[mask > 0] = 255 - enhanced[mask > 0]
    return inverted
