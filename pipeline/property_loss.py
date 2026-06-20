"""Оценка остатка и потери свойств масла относительно свежего (полевой расчёт)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.types import Diagnostics


@dataclass(frozen=True)
class PropertyEstimate:
    """Одно свойство: сколько сохранилось от исходного (100% = свежее масло)."""

    key: str
    title: str
    retention_pct: float
    loss_pct: float
    basis: str


@dataclass(frozen=True)
class PropertyLossReport:
    properties: tuple[PropertyEstimate, ...]
    overall_retention_pct: float
    overall_loss_pct: float
    summary: str


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _confidence_weight(confidence: str) -> float:
    return {"высокая": 1.0, "средняя": 0.7, "низкая": 0.35}.get(confidence, 0.5)


def compute_property_loss(d: Diagnostics) -> PropertyLossReport:
    """
    Приблизительный остаток свойств масла относительно свежего.

    Не заменяет лабораторию (вязкость, TBN, FTIR). Основано на метриках капельного теста.
    """
    sludge_pct = d.sludge_ratio * 100.0
    fuel_w = _confidence_weight(d.fuel_confidence)
    water_w = _confidence_weight(d.water_confidence)
    if d.water_confidence == "низкая" and d.dispersion_index > 0.7 and d.contamination_index < 30:
        water_w *= 0.35

    dispersancy = _clip(d.merit_of_dispersancy)

    detergent = _clip(
        0.50 * d.yellow_ring_score
        + 0.30 * d.core_uniformity
        + 0.20 * max(0.0, 100.0 - sludge_pct * 2.0)
        - (0.0 if d.yellow_ring_continuous else 8.0)
    )

    additive_package = _clip(d.additive_resource_pct)

    tbn_proxy = _clip(0.55 * d.additive_resource_pct + 0.45 * d.yellow_ring_score)

    oxidation_resistance = _clip(100.0 - d.oxidation_index)

    soot_capacity = _clip(0.60 * d.dispersion_index * 100.0 + 0.40 * (100.0 - d.contamination_index))

    fuel_visc_loss = min(d.fuel_estimate_pct * 2.6 * fuel_w, 28.0)
    ox_visc_loss = d.oxidation_index * 0.45
    water_visc_loss = min(d.water_estimate_pct * 1.8 * water_w, 12.0)
    viscosity_retention = _clip(100.0 - fuel_visc_loss - ox_visc_loss - water_visc_loss)

    antiwear = _clip(0.65 * d.additive_resource_pct + 0.35 * d.core_uniformity - sludge_pct * 0.8)

    anticorrosion = _clip(
        0.70 * tbn_proxy + 0.30 * max(0.0, 100.0 - d.water_estimate_pct * 5.0 * water_w)
    )

    thermal_stability = _clip(
        0.50 * oxidation_resistance
        + 0.30 * dispersancy
        + 0.20 * max(0.0, 100.0 - sludge_pct * 1.5)
    )

    seal_compatibility = _clip(
        0.55 * oxidation_resistance + 0.25 * viscosity_retention + 0.20 * additive_package
    )

    properties = (
        PropertyEstimate(
            "dispersancy",
            "Диспергирующая способность",
            dispersancy,
            _clip(100.0 - dispersancy),
            "MD, DS, кольцо A, шлам в ядре",
        ),
        PropertyEstimate(
            "detergent",
            "Моющие (детергентные) свойства",
            detergent,
            _clip(100.0 - detergent),
            "Кольцо A, однородность ядра C, шлам",
        ),
        PropertyEstimate(
            "additive",
            "Пакет присадок (общий)",
            additive_package,
            _clip(100.0 - additive_package),
            "Кольцо A, DS, однородность ядра",
        ),
        PropertyEstimate(
            "tbn",
            "Щёлочной резерв (TBN, оценка)",
            tbn_proxy,
            _clip(100.0 - tbn_proxy),
            "Ресурс присадок, кольцо A (не лабораторный TBN)",
        ),
        PropertyEstimate(
            "oxidation",
            "Стойкость к окислению",
            oxidation_resistance,
            _clip(100.0 - oxidation_resistance),
            "Индекс окисления зон A+D",
        ),
        PropertyEstimate(
            "soot",
            "Саже- и грязеёмкость",
            soot_capacity,
            _clip(100.0 - soot_capacity),
            "DS, индекс загрузки CI",
        ),
        PropertyEstimate(
            "viscosity",
            "Удержание вязкости (оценка)",
            viscosity_retention,
            _clip(100.0 - viscosity_retention),
            "Разбавление топливом, окисление, вода (не кинематическая вязкость)",
        ),
        PropertyEstimate(
            "antiwear",
            "Противоизносная защита (оценка)",
            antiwear,
            _clip(100.0 - antiwear),
            "Ресурс присадок, чистота ядра",
        ),
        PropertyEstimate(
            "anticorrosion",
            "Антикоррозионные свойства (оценка)",
            anticorrosion,
            _clip(100.0 - anticorrosion),
            "TBN-оценка, содержание воды",
        ),
        PropertyEstimate(
            "thermal",
            "Термостойкость / стабильность (оценка)",
            thermal_stability,
            _clip(100.0 - thermal_stability),
            "Окисление, дисперсия, шлам",
        ),
        PropertyEstimate(
            "seals",
            "Совместимость с уплотнениями (оценка)",
            seal_compatibility,
            _clip(100.0 - seal_compatibility),
            "Окисление, вязкость, пакет присадок",
        ),
    )

    weights = {
        "dispersancy": 0.17,
        "detergent": 0.13,
        "additive": 0.14,
        "tbn": 0.09,
        "oxidation": 0.11,
        "soot": 0.14,
        "viscosity": 0.10,
        "antiwear": 0.05,
        "anticorrosion": 0.04,
        "thermal": 0.02,
        "seals": 0.01,
    }
    overall = sum(p.retention_pct * weights[p.key] for p in properties)
    overall = _clip(overall)
    overall_loss = _clip(100.0 - overall)

    if overall >= 80:
        summary = (
            "Масло сохранило большую часть ключевых свойств — близко к свежему "
            "по данным капельного теста."
        )
    elif overall >= 65:
        summary = (
            "Заметная, но некритичная деградация свойств — масло ещё работоспособно, "
            "рекомендуется контроль."
        )
    elif overall >= 45:
        summary = (
            "Существенная потеря свойств — планируйте замену и лабораторную проверку "
            "при сомнениях."
        )
    else:
        summary = (
            "Критическая деградация свойств — масло, вероятно, исчерпало ресурс "
            "по полевым признакам."
        )

    return PropertyLossReport(
        properties=properties,
        overall_retention_pct=overall,
        overall_loss_pct=overall_loss,
        summary=summary,
    )
