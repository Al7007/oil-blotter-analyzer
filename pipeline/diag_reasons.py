"""Тексты всплывающих пояснений для красных (опасных) показателей."""

from __future__ import annotations

from pipeline.diag_colors import ColorTag
from pipeline.types import Diagnostics, DropConsistency


def reason_for(
    metric: str,
    tag: ColorTag,
    d: Diagnostics,
    dc: DropConsistency | None = None,
    *,
    blotter_digit: int | None = None,
    property_title: str | None = None,
    retention_pct: float | None = None,
    loss_pct: float | None = None,
) -> str | None:
    """Возвращает текст для всплывающего окна, если показатель красный."""
    if tag != "bad":
        return None

    if metric == "ds":
        return (
            f"DS = {d.dispersion_index:.2f} — ниже 0.35 (критично).\n\n"
            "Сажа плохо диспергируется и склонна скапливаться в центре пятна. "
            "Масло, вероятно, исчерпало диспергирующую способность.\n\n"
            "Рекомендация: замена масла, при необходимости — лабораторный анализ."
        )

    if metric == "md":
        return (
            f"MD = {d.merit_of_dispersancy:.0f}/100 — ниже 40.\n\n"
            "Низкая диспергирующая способность: присадки не удерживают продукты "
            "горения в растворе. Возможны отложения и ускоренный износ.\n\n"
            "Рекомендация: замена масла."
        )

    if metric == "ci":
        return (
            f"CI = {d.contamination_index:.0f}/100 — высокая загрузка сажой.\n\n"
            f"Шлам в ядре: {d.sludge_ratio * 100:.1f}%. "
            "В масле много твёрдых продуктов горения.\n\n"
            "Рекомендация: замена масла и проверка двигателя на износ."
        )

    if metric == "additive":
        return (
            f"Ресурс присадок {d.additive_resource_pct:.0f}% — ниже 40%.\n\n"
            "Присадочный пакет почти исчерпан: кольцо A, DS или однородность ядра "
            "в критической зоне.\n\n"
            "Рекомендация: ближайшая замена масла."
        )

    if metric == "ring":
        return (
            f"Кольцо A: {d.yellow_ring_score:.0f}% — размытое или неполное.\n\n"
            "Жёлтое кольцо присадок (зона A) прерывистое — дисперганты и "
            "антиокислители работают слабо.\n\n"
            "Рекомендация: замена масла или срочный контроль."
        )

    if metric == "core":
        return (
            f"Ядро: {d.core_status}.\n\n"
            "В центре пятна (зона C) критическое скопление шлама или нагара. "
            "Масло не справляется с удержанием загрязнений.\n\n"
            "Рекомендация: замена масла."
        )

    if metric == "oxidation":
        return (
            f"Окисление {d.oxidation_index:.0f}/100 — выраженное.\n\n"
            "Масло сильно окислено (длительная эксплуатация или перегрев). "
            "Падает вязкость и ресурс присадок.\n\n"
            "Рекомендация: замена масла."
        )

    if metric == "symmetry":
        return (
            f"Симметрия {d.symmetry_index:.0f}/100 — низкая.\n\n"
            "Пятно сильно отклоняется от круга. Оценки топлива и воды ненадёжны. "
            "Проверьте нанесение капли, складки бумаги, блики и кроп."
        )

    if metric == "fuel":
        return (
            f"Топливо ~{d.fuel_estimate_pct:.1f}% — избыток (>6%).\n\n"
            "Широкая зона T указывает на разбавление дизельным топливом. "
            "Подтвердите хроматографией (GC) в лаборатории."
        )

    if metric == "water":
        return (
            f"Вода ~{d.water_estimate_pct:.1f}% — высокий риск.\n\n"
            "Рваный край зоны T или сильная некруглость пятна. "
            "Подтвердите методом Karl Fischer.\n\n"
            f"Достоверность оценки: {d.water_confidence}."
        )

    if metric == "blotter_digit" and blotter_digit is not None:
        labels = ("сажа/нагар", "топливо", "вода/антифриз")
        idx = blotter_digit
        if 0 <= idx < 3:
            return (
                f"Blotter-код, цифра {idx + 1} ({labels[idx]}) = 2 — выраженный признак.\n\n"
                + _blotter_digit_detail(idx, d)
            )
        return None

    if metric == "blotter_code":
        return (
            f"Blotter-код {d.blotter_code} "
            f"({d.blotter_code_soot}/{d.blotter_code_fuel}/{d.blotter_code_water}).\n\n"
            "Хотя бы один фактор в зоне «2» (высоко). "
            "Если DS и CI хорошие, а третья цифра = 2 — часто артефакт края пятна на фото."
        )

    if metric == "recommendation":
        return (
            f"{d.recommendation.capitalize()}.\n\n"
            "Показатели дисперсии и/или присадок в критической зоне. "
            "Дальнейшая эксплуатация на этом масле не рекомендуется."
        )

    if metric == "consistency" and dc is not None:
        return (
            f"Согласованность {dc.overall_score:.0f}/100 — низкая.\n\n"
            f"Разброс DS: {dc.ds_spread:.2f}, MD: {dc.md_spread:.1f}, "
            f"топлива: {dc.fuel_spread:.1f}%, воды: {dc.water_spread:.1f}%.\n\n"
            f"{dc.summary}\n\n"
            "Чаще всего: разное освещение, край кропа, ручка у капли — не разница масла. "
            "Доверяйте DS, MD и CI; воду и топливо между каплями не сравнивайте."
        )

    if metric == "spread":
        return (
            "Большой разброс DS/MD между каплями на одном снимке.\n\n"
            "Капли проанализированы несогласованно — проверьте клик по центру, "
            "одинаковое время подсыхания и отсутствие артефактов у края."
        )

    if metric == "spread_fw" and dc is not None:
        return (
            f"Разброс топлива {dc.fuel_spread:.1f}%, воды {dc.water_spread:.1f}%.\n\n"
            "Эти показатели чувствительны к форме края и фото. "
            "Расхождение не означает разное масло — чаще артефакт съёмки."
        )

    if metric == "overall_grade":
        return (
            "Масло в критическом состоянии по совокупности показателей.\n\n"
            "Диспергирование, присадки или загрузка сажой в опасной зоне. "
            "Замена масла необходима."
        )

    if metric == "retention" and property_title and retention_pct is not None:
        return (
            f"{property_title}: осталось {retention_pct:.0f}%.\n\n"
            "Сильная потеря этого свойства относительно свежего масла "
            "(полевой расчёт по капельному тесту).\n\n"
            "Для точных значений — лаборатория (вязкость, TBN, FTIR)."
        )

    if metric == "loss" and property_title and loss_pct is not None:
        return (
            f"{property_title}: потеря {loss_pct:.0f}%.\n\n"
            "Более половины этого свойства утрачено относительно свежего масла."
        )

    if metric == "overall_retention" and retention_pct is not None:
        return (
            f"Сохранено ~{retention_pct:.0f}% свойств — критически мало.\n\n"
            "Масло, вероятно, исчерпало ресурс по полевым признакам. "
            "Рекомендуется замена."
        )

    return (
        "Показатель в опасной (красной) зоне.\n\n"
        "Откройте «Подробно» для деталей или «Описать результат» для общего вывода."
    )


def _blotter_digit_detail(idx: int, d: Diagnostics) -> str:
    if idx == 0:
        return (
            f"CI = {d.contamination_index:.0f}/100, шлам {d.sludge_ratio * 100:.1f}%. "
            "Много сажи и нагара в зоне C."
        )
    if idx == 1:
        return f"Оценка топлива ~{d.fuel_estimate_pct:.1f}%. Сильное разбавление дизелем."
    return (
        f"Оценка воды ~{d.water_estimate_pct:.1f}%, достоверность {d.water_confidence}. "
        "При DS > 0.7 и низком CI часто это артефакт края T на фото, а не вода."
    )
