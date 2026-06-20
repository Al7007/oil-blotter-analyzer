"""Спектральная визуализация по радиальным зонам ASTM."""

from __future__ import annotations

import numpy as np

from pipeline.radial import BlotterZones


def render_spectral(zones: BlotterZones, intensity: np.ndarray) -> np.ndarray:
    """
    Цвета по научным зонам, насыщенность — по локальной интенсивности пятна.
    C — красный/фиолетовый, A — жёлтый, D — зелёный, T — синий, фон — чёрный.
    """
    h, w = intensity.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    zone_defs: list[tuple[np.ndarray, tuple[int, int, int]]] = [
        (zones.T, (0, 0, 255)),
        (zones.D, (0, 255, 0)),
        (zones.A, (255, 255, 0)),
        (zones.C, (255, 0, 0)),
    ]

    for mask, color in zone_defs:
        if not np.any(mask):
            continue
        local = intensity[mask].astype(np.float32) / 255.0
        local = np.clip(local * 1.15 + 0.15, 0.2, 1.0)
        rgb = (np.array(color, dtype=np.float32) * local[:, None]).astype(np.uint8)
        out[mask] = rgb

    core_dark = zones.C & (intensity > 175)
    out[core_dark] = (160, 0, 200)

    sludge = zones.C & (intensity > 210)
    out[sludge] = (80, 0, 120)

    return out


def classify_zones(spectral_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Резервная сегментация по цвету (для совместимости)."""
    r = spectral_rgb[:, :, 0]
    g = spectral_rgb[:, :, 1]
    b = spectral_rgb[:, :, 2]

    background = (r < 20) & (g < 20) & (b < 20)
    blue = (b > 100) & (g < 120) & (r < 120) & ~background
    green = (g > 100) & (r < 120) & (b < 120) & ~background
    yellow = (r > 120) & (g > 120) & (b < 100) & ~background
    red = (r > 120) & (g < 80) & (b < 100) & ~background
    purple = (r > 70) & (b > 90) & (g < 60) & ~background

    return {
        "background": background,
        "blue": blue,
        "green": green,
        "yellow": yellow,
        "red": red,
        "purple": purple,
        "drop": blue | green | yellow | red | purple,
    }
