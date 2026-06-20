"""Пороги и цветовые теги для показателей диагностики."""

from __future__ import annotations

from typing import Literal

from pipeline.types import Diagnostics

ColorTag = Literal["good", "warn", "bad"]


def _high_good(value: float, good_min: float, warn_min: float) -> ColorTag:
    if value >= good_min:
        return "good"
    if value >= warn_min:
        return "warn"
    return "bad"


def _low_good(value: float, good_max: float, warn_max: float) -> ColorTag:
    if value <= good_max:
        return "good"
    if value <= warn_max:
        return "warn"
    return "bad"


def tag_ds(ds: float) -> ColorTag:
    return _high_good(ds, 0.7, 0.35)


def tag_md(md: float) -> ColorTag:
    return _high_good(md, 80.0, 40.0)


def tag_ci(ci: float) -> ColorTag:
    return _low_good(ci, 30.0, 55.0)


def tag_additive(pct: float) -> ColorTag:
    return _high_good(pct, 65.0, 40.0)


def tag_ring_score(score: float, continuous: bool) -> ColorTag:
    tag = _high_good(score, 70.0, 40.0)
    if not continuous and tag == "good":
        return "warn"
    return tag


def tag_oxidation(index: float) -> ColorTag:
    return _low_good(index, 25.0, 50.0)


def tag_symmetry(index: float) -> ColorTag:
    return _high_good(index, 72.0, 50.0)


def tag_blotter_digit(digit: int) -> ColorTag:
    if digit <= 0:
        return "good"
    if digit == 1:
        return "warn"
    return "bad"


def tag_blotter_code_worst(d: Diagnostics) -> ColorTag:
    digits = (d.blotter_code_soot, d.blotter_code_fuel, d.blotter_code_water)
    if any(x >= 2 for x in digits):
        return "bad"
    if any(x >= 1 for x in digits):
        return "warn"
    return "good"


def tag_fuel(pct: float, confidence: str) -> ColorTag:
    if confidence == "низкая" and pct < 6.0:
        return "warn"
    return _low_good(pct, 2.5, 6.0)


def tag_water(pct: float, confidence: str, ds: float, ci: float) -> ColorTag:
    if confidence == "низкая" and ds > 0.7 and ci < 30:
        return "warn"
    if confidence == "низкая" and pct < 2.5:
        return "warn"
    return _low_good(pct, 0.8, 2.5)


def tag_consistency(score: float) -> ColorTag:
    return _high_good(score, 85.0, 65.0)


def tag_core_status(status: str) -> ColorTag:
    if "однородное" in status and "неоднород" not in status:
        return "good"
    if "вкрапления" in status or "неоднород" in status:
        return "warn"
    if "критическ" in status or "шлам" in status:
        return "bad"
    return "warn"


def tag_recommendation(recommendation: str) -> ColorTag:
    if "разрешена" in recommendation:
        return "good"
    if "рекомендуется замена" in recommendation:
        return "bad"
    return "warn"


def tag_overall_grade(grade: str) -> ColorTag:
    if grade == "хорошее":
        return "good"
    if grade in ("удовлетворительное", "на грани"):
        return "warn"
    return "bad"


def tag_spread_ds(spread: float) -> ColorTag:
    return _low_good(spread, 0.08, 0.2)


def tag_spread_md(spread: float) -> ColorTag:
    return _low_good(spread, 8.0, 20.0)


def tag_spread_fuel(spread: float) -> ColorTag:
    return _low_good(spread, 1.5, 4.0)


def tag_spread_water(spread: float) -> ColorTag:
    return _low_good(spread, 1.0, 3.0)


def tag_retention(pct: float) -> ColorTag:
    """Остаток свойства: высокий — хорошо."""
    return _high_good(pct, 75.0, 50.0)


def tag_loss(pct: float) -> ColorTag:
    """Потеря свойства: низкая — хорошо."""
    return _low_good(pct, 25.0, 50.0)
