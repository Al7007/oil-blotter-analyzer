"""Радиальная сегментация зон капельного теста (ASTM D7899 / blotter spot)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.preprocess import DropGeometry


@dataclass
class BlotterZones:
    """Зоны хроматограммы: C — центр, A — ауреола, D — диффузия, T — прозрачная кромка."""

    C: np.ndarray
    A: np.ndarray
    D: np.ndarray
    T: np.ndarray
    drop: np.ndarray
    geometry: DropGeometry
    radial_profile: np.ndarray
    core_radius_norm: float
    aureole_radius_norm: float
    diffusion_radius_norm: float


def _radial_distance(shape: tuple[int, int], center: tuple[float, float]) -> np.ndarray:
    h, w = shape
    ys, xs = np.indices((h, w))
    cx, cy = center
    return np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)


def _radial_profile(
    stain: np.ndarray,
    mask: np.ndarray,
    center: tuple[float, float],
    radius: float,
    bins: int = 64,
) -> np.ndarray:
    dist = _radial_distance(stain.shape, center)
    profile = np.zeros(bins, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.int32)

    norm_r = np.clip(dist / radius, 0, 1)
    valid = mask > 0
    bin_idx = np.clip((norm_r[valid] * (bins - 1)).astype(np.int32), 0, bins - 1)
    np.add.at(profile, bin_idx, stain[valid])
    np.add.at(counts, bin_idx, 1)

    counts = np.maximum(counts, 1)
    profile /= counts
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 9), 0).flatten()
    return profile


def _find_core_radius(profile: np.ndarray) -> float:
    """Граница центральной зоны C по спаду радиального профиля."""
    if profile.size < 8:
        return 0.18

    center_peak = float(np.max(profile[: max(4, len(profile) // 10)]))
    if center_peak <= 0:
        return 0.18

    threshold = center_peak * 0.62
    for i in range(1, min(20, len(profile))):
        if profile[i] < threshold:
            return (i + 1) / len(profile)

    return 0.22


def _find_aureole_radius(profile: np.ndarray, core_norm: float) -> float:
    """
    Внешняя граница жёлтого кольца A.
    Ищем локальный максимум/плато после центра (кольцо присадок).
    """
    start = max(2, int(core_norm * len(profile)))
    end = min(len(profile) - 2, int(0.55 * len(profile)))
    if start >= end:
        return min(core_norm + 0.12, 0.38)

    segment = profile[start:end]
    if segment.size == 0:
        return min(core_norm + 0.12, 0.38)

    peak_rel = int(np.argmax(segment))
    peak_idx = start + peak_rel
    aureole_end = min(peak_idx + 4, int(0.42 * len(profile)))
    return (aureole_end + 1) / len(profile)


def segment_blotter_zones(
    stain: np.ndarray,
    intensity: np.ndarray,
    geometry: DropGeometry,
) -> BlotterZones:
    """
    Делит каплю на концентрические зоны по радиусу и профилю окрашивания.
    """
    mask = geometry.mask > 0
    cx, cy = geometry.center
    radius = geometry.radius
    dist = _radial_distance(stain.shape, (cx, cy))
    norm_r = dist / radius

    profile = _radial_profile(stain, geometry.mask, (cx, cy), radius)
    core_norm = _find_core_radius(profile)
    aureole_norm = max(_find_aureole_radius(profile, core_norm), core_norm + 0.06)
    aureole_norm = min(aureole_norm, 0.45)
    diffusion_norm = 0.84

    stain_vals = stain[mask]
    dark_threshold = float(np.percentile(stain_vals, 68)) if stain_vals.size else 0.0

    C = mask & (norm_r <= core_norm) & (stain >= dark_threshold * 0.55)
    if not np.any(C):
        C = mask & (norm_r <= core_norm)

    A = mask & (norm_r > core_norm) & (norm_r <= aureole_norm)
    D = mask & (norm_r > aureole_norm) & (norm_r <= diffusion_norm)
    T = mask & (norm_r > diffusion_norm)

    return BlotterZones(
        C=C,
        A=A,
        D=D,
        T=T,
        drop=mask,
        geometry=geometry,
        radial_profile=profile,
        core_radius_norm=core_norm,
        aureole_radius_norm=aureole_norm,
        diffusion_radius_norm=diffusion_norm,
    )


def zones_to_legacy_dict(zones: BlotterZones) -> dict[str, np.ndarray]:
    """Совместимость с визуализацией: C=red, A=yellow, D=green, T=blue."""
    background = ~zones.drop
    return {
        "background": background,
        "red": zones.C,
        "yellow": zones.A,
        "green": zones.D,
        "blue": zones.T,
        "purple": zones.C & False,
        "drop": zones.drop,
        "C": zones.C,
        "A": zones.A,
        "D": zones.D,
        "T": zones.T,
    }
