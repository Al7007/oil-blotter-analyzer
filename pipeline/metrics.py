"""Расчёт метрик по радиальным зонам ASTM и оверлей."""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.radial import BlotterZones
from pipeline.types import Diagnostics, DropConsistency


def _diameter_from_mask(mask: np.ndarray) -> float:
    area = float(np.count_nonzero(mask))
    if area <= 0:
        return 0.0
    return 2.0 * np.sqrt(area / np.pi)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    binary = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _ring_score(zones: BlotterZones) -> tuple[float, bool]:
    """Чёткость жёлтого кольца A вокруг ядра C."""
    if not np.any(zones.A) or not np.any(zones.C):
        return 0.0, False

    cx, cy = zones.geometry.center
    radius = zones.geometry.radius
    h, w = zones.C.shape
    ys, xs = np.indices((h, w))
    norm_r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / radius

    expected = (
        zones.drop
        & (norm_r > zones.core_radius_norm)
        & (norm_r <= zones.aureole_radius_norm)
    )
    coverage = np.count_nonzero(zones.A & expected) / max(np.count_nonzero(expected), 1)

    aureole_contour = _largest_contour(zones.A)
    if aureole_contour is None:
        return float(coverage * 60), False

    perimeter = cv2.arcLength(aureole_contour, True)
    area = cv2.contourArea(aureole_contour)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    continuous = coverage > 0.35 and circularity > 0.25
    score = float(np.clip(coverage * 50 + circularity * 50, 0, 100))
    return score, continuous


def _edge_roughness(mask: np.ndarray, border: int = 8) -> float:
    trimmed = mask.copy()
    h, w = trimmed.shape
    trimmed[:border, :] = False
    trimmed[-border:, :] = False
    trimmed[:, :border] = False
    trimmed[:, -border:] = False

    contour = _largest_contour(trimmed)
    if contour is None:
        return 0.0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 1:
        return 0.0
    circle_perimeter = 2 * np.pi * np.sqrt(area / np.pi)
    return float(max(0.0, perimeter / circle_perimeter - 1.0))


def _core_uniformity(zones: BlotterZones, intensity: np.ndarray) -> tuple[float, float]:
    if not np.any(zones.C):
        return 0.0, 0.0

    core_vals = intensity[zones.C]
    sludge = zones.C & (intensity > 200)
    sludge_ratio = np.count_nonzero(sludge) / max(np.count_nonzero(zones.C), 1)

    if core_vals.size == 0:
        return 0.0, float(sludge_ratio)

    std = float(np.std(core_vals))
    uniformity = float(np.clip(100 - std * 1.0 - sludge_ratio * 100, 0, 100))
    return uniformity, float(sludge_ratio)


def _circularity(mask: np.ndarray) -> float:
    contour = _largest_contour(mask)
    if contour is None:
        return 1.0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1:
        return 1.0
    return float(4 * np.pi * area / (perimeter * perimeter))


def _zone_area_fractions(zones: BlotterZones) -> tuple[float, float, float, float]:
    drop_pixels = max(np.count_nonzero(zones.drop), 1)
    return (
        float(np.count_nonzero(zones.C) / drop_pixels * 100),
        float(np.count_nonzero(zones.A) / drop_pixels * 100),
        float(np.count_nonzero(zones.D) / drop_pixels * 100),
        float(np.count_nonzero(zones.T) / drop_pixels * 100),
    )


def _zone_radius_fractions(zones: BlotterZones) -> tuple[float, float, float, float]:
    c = zones.core_radius_norm * 100
    a = max(0.0, (zones.aureole_radius_norm - zones.core_radius_norm) * 100)
    d = max(0.0, (zones.diffusion_radius_norm - zones.aureole_radius_norm) * 100)
    t = max(0.0, (1.0 - zones.diffusion_radius_norm) * 100)
    return float(c), float(a), float(d), float(t)


def _merit_of_dispersancy(
    dispersion_index: float,
    yellow_score: float,
    sludge_ratio: float,
    yellow_continuous: bool,
) -> float:
    """MD (Merit of Dispersancy) 0–100 по логике ASTM D7899."""
    md = dispersion_index * 100.0
    md += (yellow_score - 50.0) * 0.12
    md -= sludge_ratio * 35.0
    if yellow_continuous:
        md += 5.0
    return float(np.clip(md, 0.0, 100.0))


def _contamination_index(zones: BlotterZones, intensity: np.ndarray, sludge_ratio: float) -> float:
    """Индекс загрузки сажой/нагаром по зоне C."""
    drop_pixels = max(np.count_nonzero(zones.drop), 1)
    core_frac = np.count_nonzero(zones.C) / drop_pixels

    if np.any(zones.C):
        core_darkness = float(np.mean(intensity[zones.C]) / 255.0)
    else:
        core_darkness = 0.0

    ci = core_frac * 55.0 + sludge_ratio * 100.0 * 0.3 + core_darkness * 35.0
    return float(np.clip(ci, 0.0, 100.0))


def _oxidation_index(rgb: np.ndarray | None, zones: BlotterZones) -> float:
    """Окисление: потемнение/пожелтение зон A и D в Lab."""
    ad_mask = zones.A | zones.D
    if rgb is None or not np.any(ad_mask):
        return 0.0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_mask = zones.T if np.count_nonzero(zones.T) > 30 else zones.drop
    ref_a = float(np.mean(lab[:, :, 1][ref_mask]))
    ref_b = float(np.mean(lab[:, :, 2][ref_mask]))

    a_shift = float(np.mean(lab[:, :, 1][ad_mask]) - ref_a)
    b_shift = float(np.mean(lab[:, :, 2][ad_mask]) - ref_b)
    l_drop = float(np.mean(lab[:, :, 0][ad_mask]))

    oxidation = (a_shift * 0.45 + b_shift * 0.35) + max(0.0, (128.0 - l_drop) * 0.08)
    return float(np.clip(oxidation, 0.0, 100.0))


def _symmetry_index(zones: BlotterZones, edge_roughness: float) -> float:
    circularity = _circularity(zones.drop)
    symmetry = circularity * 100.0 - edge_roughness * 55.0
    return float(np.clip(symmetry, 0.0, 100.0))


def _blotter_code_digits(
    contamination_index: float,
    sludge_ratio: float,
    fuel_pct: float,
    water_pct: float,
    edge_roughness: float,
) -> tuple[int, int, int, str]:
    if contamination_index < 30.0 and sludge_ratio < 0.04:
        soot = 0
    elif contamination_index < 55.0 and sludge_ratio < 0.12:
        soot = 1
    else:
        soot = 2

    if fuel_pct < 2.5:
        fuel = 0
    elif fuel_pct < 6.0:
        fuel = 1
    else:
        fuel = 2

    if water_pct < 0.8 and edge_roughness < 0.10:
        water = 0
    elif water_pct < 2.5:
        water = 1
    else:
        water = 2

    return soot, fuel, water, f"{soot}{fuel}{water}"


def _estimate_fuel_pct(zones: BlotterZones, t_thickness_norm: float) -> float:
    """
    Полевая оценка % топлива по зоне T (ASTM blotter spot).

    Опора: узкая прозрачная кромка (~10–12% радиуса) — норма;
    расширение зоны T коррелирует с разбавлением дизельным топливом
    (Machinery Lubrication, Practical Sailor). Не заменяет GC/FTIR.
    """
    baseline_thickness = 0.11
    baseline_area = 1.0 - (1.0 - baseline_thickness) ** 2

    excess_thickness = max(0.0, t_thickness_norm - baseline_thickness)
    from_thickness = excess_thickness / 0.16 * 10.0

    drop_area = max(np.count_nonzero(zones.drop), 1)
    t_area_frac = np.count_nonzero(zones.T) / drop_area
    excess_area = max(0.0, t_area_frac - baseline_area)
    from_area = excess_area / 0.18 * 8.0

    estimate = 0.55 * from_thickness + 0.45 * from_area
    return float(np.clip(estimate, 0.0, 20.0))


def _estimate_water_pct(zones: BlotterZones, edge_roughness: float) -> float:
    """
    Полевая оценка % воды/антифриза по форме внешнего ореола.

    Опора: вода и гликоль дают рваный, зигзагообразный край зоны T
    и нарушают круглость пятна (полевые наблюдения blotter test).
    """
    circularity = _circularity(zones.drop)
    roughness_score = max(0.0, (edge_roughness - 0.08) / 0.25)
    shape_score = max(0.0, (0.82 - circularity) / 0.22)

    estimate = roughness_score * 4.5 + shape_score * 3.5
    return float(np.clip(estimate, 0.0, 12.0))


def _confidence_label(
    symmetry_index: float,
    edge_roughness: float,
    *,
    metric: str,
    estimate_pct: float,
    t_thickness_norm: float,
) -> str:
    if metric == "fuel":
        if symmetry_index >= 70.0 and 0.06 <= t_thickness_norm <= 0.22:
            return "высокая"
        if symmetry_index < 50.0 or t_thickness_norm < 0.04:
            return "низкая"
        return "средняя"

    if symmetry_index >= 72.0 and edge_roughness < 0.14 and estimate_pct < 4.0:
        return "высокая"
    if symmetry_index < 50.0 or edge_roughness > 0.28:
        return "низкая"
    return "средняя"


def _fuel_status_from_pct(pct: float) -> str:
    if pct < 2.5:
        return "в норме"
    if pct < 6.0:
        return "слегка разжижено"
    return "избыток топлива"


def _water_status_from_pct(pct: float) -> str:
    if pct < 0.8:
        return "в норме"
    if pct < 2.5:
        return "подозрение на воду"
    return "возможна вода/антифриз"


def _md_status(md: float) -> str:
    if md >= 80.0:
        return "отличная дисперсия"
    if md >= 60.0:
        return "хорошая дисперсия"
    if md >= 40.0:
        return "удовлетворительная"
    return "низкая дисперсия"


def analyze(
    zones: BlotterZones,
    intensity: np.ndarray,
    rgb: np.ndarray | None = None,
) -> Diagnostics:
    """Метрики A–D по зонам C/A/D/T (не по шумной LUT)."""
    d = _diameter_from_mask(zones.C)
    D = _diameter_from_mask(zones.D | zones.A | zones.C)

    if D > 0 and d > 0:
        dispersion_index = 1.0 - (d * d) / (D * D)
    else:
        dispersion_index = 0.0
    dispersion_index = float(np.clip(dispersion_index, 0.0, 1.0))

    yellow_score, yellow_continuous = _ring_score(zones)
    core_uniformity, sludge_ratio = _core_uniformity(zones, intensity)

    t_thickness_norm = max(0.0, 1.0 - zones.diffusion_radius_norm)
    blue_aura_ratio = t_thickness_norm

    blue_roughness = _edge_roughness(zones.T)

    fuel_estimate_pct = _estimate_fuel_pct(zones, t_thickness_norm)
    water_estimate_pct = _estimate_water_pct(zones, blue_roughness)
    fuel_status = _fuel_status_from_pct(fuel_estimate_pct)
    water_status = _water_status_from_pct(water_estimate_pct)

    merit_of_dispersancy = _merit_of_dispersancy(
        dispersion_index, yellow_score, sludge_ratio, yellow_continuous
    )
    contamination_index = _contamination_index(zones, intensity, sludge_ratio)
    oxidation_index = _oxidation_index(rgb, zones)
    symmetry_index = _symmetry_index(zones, blue_roughness)

    soot_digit, fuel_digit, water_digit, blotter_code = _blotter_code_digits(
        contamination_index,
        sludge_ratio,
        fuel_estimate_pct,
        water_estimate_pct,
        blue_roughness,
    )

    zone_area_c, zone_area_a, zone_area_d, zone_area_t = _zone_area_fractions(zones)
    zone_radius_c, zone_radius_a, zone_radius_d, zone_radius_t = _zone_radius_fractions(zones)

    fuel_confidence = _confidence_label(
        symmetry_index,
        blue_roughness,
        metric="fuel",
        estimate_pct=fuel_estimate_pct,
        t_thickness_norm=t_thickness_norm,
    )
    water_confidence = _confidence_label(
        symmetry_index,
        blue_roughness,
        metric="water",
        estimate_pct=water_estimate_pct,
        t_thickness_norm=t_thickness_norm,
    )

    additive_resource_pct = float(
        np.clip(yellow_score * 0.4 + dispersion_index * 100 * 0.45 + core_uniformity * 0.15, 0, 100)
    )

    if sludge_ratio > 0.15:
        core_status = "критическое скопление шлама"
    elif sludge_ratio > 0.06:
        core_status = "есть тёмные вкрапления"
    elif core_uniformity > 60:
        core_status = "ядро однородное"
    else:
        core_status = "ядро неоднородное"

    if merit_of_dispersancy >= 70.0 and additive_resource_pct >= 65:
        recommendation = "эксплуатация разрешена"
    elif merit_of_dispersancy >= 40.0:
        recommendation = "плановый контроль, возможна близкая замена"
    else:
        recommendation = "рекомендуется замена масла"

    md_label = _md_status(merit_of_dispersancy)
    details = [
        f"Коэффициент дисперсии DS = {dispersion_index:.2f} "
        f"(d={d:.0f}px, D={D:.0f}px; ASTM: >0.7 отлично, <0.3 критично)",
        f"MD (Merit of Dispersancy) = {merit_of_dispersancy:.0f}/100 — {md_label}",
        f"Индекс загрузки сажой CI = {contamination_index:.0f}/100 "
        f"(шлам {sludge_ratio * 100:.1f}%)",
        f"Blotter-код {blotter_code} (сажа/топливо/вода: {soot_digit}/{fuel_digit}/{water_digit}; "
        f"0=норма, 1=умеренно, 2=высоко)",
        f"Жёлтое кольцо A: {yellow_score:.0f}% "
        f"({'замкнутое' if yellow_continuous else 'размытое/рваное'})",
        f"Зоны по радиусу: C {zone_radius_c:.0f}% | A {zone_radius_a:.0f}% | "
        f"D {zone_radius_d:.0f}% | T {zone_radius_t:.0f}%",
        f"Зоны по площади: C {zone_area_c:.1f}% | A {zone_area_a:.1f}% | "
        f"D {zone_area_d:.1f}% | T {zone_area_t:.1f}%",
        f"Окисление (A+D): {oxidation_index:.0f}/100, симметрия пятна: {symmetry_index:.0f}/100",
        f"Зона C (ядро): однородность {core_uniformity:.0f}%, {core_status}",
        f"Топливо ~{fuel_estimate_pct:.1f}% ({fuel_status}), "
        f"достоверность оценки: {fuel_confidence}",
        f"Вода/антифриз ~{water_estimate_pct:.1f}% ({water_status}), "
        f"достоверность оценки: {water_confidence} (полевой скрининг, не GC/KF)",
    ]

    verdict = (
        f"MD {merit_of_dispersancy:.0f}/100 ({md_label}). "
        f"Blotter-код {blotter_code}. "
        f"Ресурс присадок: {additive_resource_pct:.0f}%. "
        f"Топливо: ~{fuel_estimate_pct:.1f}% ({fuel_status}, {fuel_confidence} достоверность). "
        f"Вода: ~{water_estimate_pct:.1f}% ({water_status}, {water_confidence} достоверность). "
        f"Ядро: {core_status}. Рекомендация: {recommendation}."
    )

    return Diagnostics(
        dispersion_index=dispersion_index,
        red_diameter_px=d,
        green_diameter_px=D,
        yellow_ring_score=yellow_score,
        yellow_ring_continuous=yellow_continuous,
        core_uniformity=core_uniformity,
        sludge_ratio=sludge_ratio,
        blue_aura_ratio=blue_aura_ratio,
        blue_edge_roughness=blue_roughness,
        fuel_estimate_pct=fuel_estimate_pct,
        water_estimate_pct=water_estimate_pct,
        additive_resource_pct=additive_resource_pct,
        merit_of_dispersancy=merit_of_dispersancy,
        contamination_index=contamination_index,
        blotter_code=blotter_code,
        blotter_code_soot=soot_digit,
        blotter_code_fuel=fuel_digit,
        blotter_code_water=water_digit,
        zone_radius_c_pct=zone_radius_c,
        zone_radius_a_pct=zone_radius_a,
        zone_radius_d_pct=zone_radius_d,
        zone_radius_t_pct=zone_radius_t,
        zone_area_c_pct=zone_area_c,
        zone_area_a_pct=zone_area_a,
        zone_area_d_pct=zone_area_d,
        zone_area_t_pct=zone_area_t,
        oxidation_index=oxidation_index,
        symmetry_index=symmetry_index,
        fuel_confidence=fuel_confidence,
        water_confidence=water_confidence,
        fuel_status=fuel_status,
        water_status=water_status,
        core_status=core_status,
        recommendation=recommendation,
        verdict=verdict,
        details=details,
    )


def compute_drop_consistency(diagnostics_list: list[Diagnostics]) -> DropConsistency:
    """Сравнение метрик между каплями на одном снимке."""
    if len(diagnostics_list) < 2:
        return DropConsistency(available=False, drop_count=len(diagnostics_list))

    ds_vals = [d.dispersion_index for d in diagnostics_list]
    md_vals = [d.merit_of_dispersancy for d in diagnostics_list]
    fuel_vals = [d.fuel_estimate_pct for d in diagnostics_list]
    water_vals = [d.water_estimate_pct for d in diagnostics_list]

    ds_spread = float(max(ds_vals) - min(ds_vals))
    md_spread = float(max(md_vals) - min(md_vals))
    fuel_spread = float(max(fuel_vals) - min(fuel_vals))
    water_spread = float(max(water_vals) - min(water_vals))

    penalties = (
        ds_spread * 80.0
        + md_spread * 0.6
        + fuel_spread * 4.0
        + water_spread * 6.0
    )
    overall = float(np.clip(100.0 - penalties, 0.0, 100.0))

    if overall >= 85.0:
        summary = "высокая согласованность — выводы надёжны"
    elif overall >= 65.0:
        summary = "умеренная согласованность — доверяйте DS/MD, вода/топливо усредняйте"
    else:
        summary = "низкая согласованность — ориентируйтесь на DS/MD, воду/топливо не доверяйте"

    if water_spread > 2.0 and ds_spread < 0.15:
        summary += "; расхождение по воде — артефакт фото, не разница масла"

    return DropConsistency(
        available=True,
        drop_count=len(diagnostics_list),
        ds_spread=ds_spread,
        md_spread=md_spread,
        fuel_spread=fuel_spread,
        water_spread=water_spread,
        overall_score=overall,
        summary=summary,
    )


def draw_overlay(
    spectral_rgb: np.ndarray,
    diagnostics: Diagnostics,
    zones: BlotterZones,
) -> np.ndarray:
    """Концентрические окружности зон C/A/D/T и метрики."""
    overlay = spectral_rgb.copy()
    cx, cy = int(zones.geometry.center[0]), int(zones.geometry.center[1])
    R = zones.geometry.radius

    ring_specs = (
        (zones.core_radius_norm * R, (255, 80, 80), "C"),
        (zones.aureole_radius_norm * R, (255, 255, 255), "A"),
        (zones.diffusion_radius_norm * R, (80, 255, 80), "D"),
        (R, (80, 160, 255), "T"),
    )
    for radius_px, color, _label in ring_specs:
        if radius_px > 2:
            cv2.circle(overlay, (cx, cy), int(radius_px), color, 1, cv2.LINE_AA)

    status_color = (0, 220, 0) if diagnostics.merit_of_dispersancy >= 70 else (0, 180, 255)
    if diagnostics.merit_of_dispersancy < 40:
        status_color = (255, 60, 60)

    cv2.putText(
        overlay,
        f"DS={diagnostics.dispersion_index:.2f} MD={diagnostics.merit_of_dispersancy:.0f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Code {diagnostics.blotter_code} CI={diagnostics.contamination_index:.0f}",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Fuel~{diagnostics.fuel_estimate_pct:.1f}%({diagnostics.fuel_confidence[:3]}) "
        f"Water~{diagnostics.water_estimate_pct:.1f}%({diagnostics.water_confidence[:3]})",
        (8, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay
