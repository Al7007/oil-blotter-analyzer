"""Формирование краткого, подробного и сводного текста диагностики для GUI."""

from __future__ import annotations

from pipeline import diag_colors as c
from pipeline.diag_reasons import reason_for
from pipeline.diag_segments import DiagSegment, seg
from pipeline.types import PipelineResult


def _val(
    text: str,
    tag: str,
    metric: str,
    d,
    dc=None,
    **reason_kw,
) -> DiagSegment:
    return seg(text, f"val_{tag}", alert=reason_for(metric, tag, d, dc, **reason_kw))


def _sub(*lines: str) -> str:
    return "\n".join(f"    {line}" for line in lines)


def _append_metric(
    segments: list[DiagSegment],
    title: str,
    value: str,
    tag: str,
    *hint_lines: str,
    metric: str | None = None,
    d=None,
    dc=None,
    **reason_kw,
) -> None:
    """title — название; value — текущее число/текст (цвет); подсказки — серые."""
    alert = reason_for(metric, tag, d, dc, **reason_kw) if metric and d is not None else None
    segments.append(seg(f"• {title}: "))
    segments.append(seg(f"{value}\n", f"val_{tag}", alert=alert))
    for line in hint_lines:
        if line.startswith("▸") or line.startswith("Ваш результат"):
            cur_alert = alert if tag == "bad" else None
            segments.append(seg(f"    {line}\n", f"cur_{tag}", alert=cur_alert))
        else:
            segments.append(seg(f"    {line}\n", "hint"))


def _core_match_key(status: str) -> str:
    s = status.lower()
    if "критическ" in s:
        return "критическ"
    if "вкрапления" in s:
        return "вкрапления"
    if "неоднород" in s:
        return "неоднородное"
    return "однородное"


def _append_choice_scale(
    segments: list[DiagSegment],
    title: str,
    value_summary: str,
    tag: str,
    options: list[tuple[str, str]],
    active_key: str,
    intro: str,
    *,
    metric: str | None = None,
    d=None,
    dc=None,
) -> None:
    """Шкала качественных состояний: ▶ — текущее, ○ — остальные (серые)."""
    alert = reason_for(metric, tag, d, dc) if metric and d is not None else None
    segments.append(seg(f"• {title}: "))
    segments.append(seg(f"{value_summary}\n", f"val_{tag}", alert=alert))
    segments.append(seg(f"    {intro}\n", "hint"))
    segments.append(seg("    Варианты (▶ — ваш результат сейчас):\n", "hint"))
    for match_key, label in options:
        is_active = match_key == active_key
        prefix = "    ▶ " if is_active else "      ○ "
        if is_active:
            segments.append(seg(f"{prefix}{label}\n", f"cur_{tag}", alert=alert if tag == "bad" else None))
        else:
            segments.append(seg(f"{prefix}{label}\n", "dim"))


def _ds_comment(ds: float) -> str:
    if ds > 0.7:
        return "Ваш результат: отличная дисперсия."
    if ds > 0.35:
        return "Ваш результат: приемлемо, стоит наблюдать."
    return "Ваш результат: низкая дисперсия — сажа склонна скапливаться в ядре."


def _md_comment(md: float) -> str:
    if md >= 80:
        return "Ваш результат: отличная диспергирующая способность."
    if md >= 60:
        return "Ваш результат: хорошая, но ресурс присадок снижается."
    if md >= 40:
        return "Ваш результат: удовлетворительная — планируйте замену."
    return "Ваш результат: низкая — масло плохо удерживает загрязнения."


def _ci_comment(ci: float) -> str:
    if ci < 30:
        return "Ваш результат: низкая загрузка сажой."
    if ci < 55:
        return "Ваш результат: умеренная загрузка."
    return "Ваш результат: высокая загрузка сажой/нагаром."


def _append_blotter_code(segments: list[DiagSegment], d) -> None:
    segments.append(seg("• Blotter-код: "))
    for i, char in enumerate(d.blotter_code):
        digit_tag = c.tag_blotter_digit(int(char))
        segments.append(
            seg(
                char,
                digit_tag,
                alert=reason_for("blotter_digit", digit_tag, d, blotter_digit=i),
            )
        )
    segments.append(
        seg(
            f"  (сажа/топливо/вода: "
            f"{d.blotter_code_soot}/{d.blotter_code_fuel}/{d.blotter_code_water})\n"
        )
    )


def format_brief_diagnostics(result: PipelineResult) -> list[DiagSegment]:
    d = result.diagnostics
    dc = result.drop_consistency
    ring = "чёткое" if d.yellow_ring_continuous else "размытое"
    segments: list[DiagSegment] = [
        seg("КРАТКИЙ ИТОГ\n"),
        _val(f"{d.recommendation}\n", c.tag_recommendation(d.recommendation), "recommendation", d),
        seg("\n"),
        seg(f"Капля {result.selected_candidate + 1} из {len(result.candidates)}\n\n"),
        seg("• MD: "),
        _val(f"{d.merit_of_dispersancy:.0f}/100", c.tag_md(d.merit_of_dispersancy), "md", d),
        seg("  |  DS: "),
        _val(f"{d.dispersion_index:.2f}\n", c.tag_ds(d.dispersion_index), "ds", d),
    ]
    _append_blotter_code(segments, d)
    segments.append(seg("  |  Ресурс присадок: "))
    segments.append(_val(f"{d.additive_resource_pct:.0f}%\n", c.tag_additive(d.additive_resource_pct), "additive", d))
    segments.append(seg("• Сажа (CI): "))
    segments.append(_val(f"{d.contamination_index:.0f}/100", c.tag_ci(d.contamination_index), "ci", d))
    segments.append(seg("  |  Ядро: "))
    segments.append(_val(f"{d.core_status}\n", c.tag_core_status(d.core_status), "core", d))
    segments.append(seg("• Кольцо A: "))
    segments.append(
        _val(
            f"{d.yellow_ring_score:.0f}% ({ring})\n",
            c.tag_ring_score(d.yellow_ring_score, d.yellow_ring_continuous),
            "ring",
            d,
        )
    )
    segments.extend(
        [
            seg("\nТопливо и вода (ориентир):\n"),
            seg("• Топливо ~"),
            _val(
                f"{d.fuel_estimate_pct:.1f}% — {d.fuel_status} ({d.fuel_confidence})\n",
                c.tag_fuel(d.fuel_estimate_pct, d.fuel_confidence),
                "fuel",
                d,
            ),
            seg("• Вода ~"),
            _val(
                f"{d.water_estimate_pct:.1f}% — {d.water_status} ({d.water_confidence})\n",
                c.tag_water(d.water_estimate_pct, d.water_confidence, d.dispersion_index, d.contamination_index),
                "water",
                d,
            ),
        ]
    )
    if dc.available:
        segments.extend(
            [
                seg("\nСогласованность "),
                seg(f"{dc.drop_count} капель: ", None),
                _val(
                    f"{dc.overall_score:.0f}/100\n",
                    c.tag_consistency(dc.overall_score),
                    "consistency",
                    d,
                    dc,
                ),
                seg(f"{dc.summary}\n"),
            ]
        )
    segments.append(
        seg(
            "\nЗелёный — норма | Жёлтый — погранично | Красный — опасно.\n"
            "Клик по красному значению — причина и рекомендация.\n"
            "«Подробно» — детали | «Описать результат» — вывод | «Потеря свойств» — деградация.\n"
        )
    )
    return segments


def _overall_grade(d) -> tuple[str, str]:
    if d.merit_of_dispersancy >= 80 and d.dispersion_index > 0.7 and d.contamination_index < 30:
        return "хорошее", "Масло в хорошем состоянии"
    if d.merit_of_dispersancy >= 60 and d.dispersion_index > 0.35:
        return "удовлетворительное", "Масло в удовлетворительном состоянии"
    if d.merit_of_dispersancy >= 40:
        return "на грани", "Масло близко к пределу ресурса"
    return "критическое", "Масло в критическом состоянии"


def _dispersion_paragraph(d) -> tuple[str, str]:
    tag = c.tag_md(d.merit_of_dispersancy)
    if d.dispersion_index > 0.7:
        text = (
            f"Диспергирующая способность высокая (DS = {d.dispersion_index:.2f}, "
            f"MD = {d.merit_of_dispersancy:.0f}/100): сажа хорошо удерживается в масле "
            "и равномерно распределяется по пятну, а не скапливается в центре."
        )
    elif d.dispersion_index > 0.35:
        text = (
            f"Диспергирующая способность умеренная (DS = {d.dispersion_index:.2f}, "
            f"MD = {d.merit_of_dispersancy:.0f}/100): масло ещё справляется, "
            "но запас по диспергированию уменьшается — имеет смысл усилить контроль."
        )
        tag = "warn"
    else:
        text = (
            f"Диспергирующая способность низкая (DS = {d.dispersion_index:.2f}, "
            f"MD = {d.merit_of_dispersancy:.0f}/100): сажа плохо удерживается, "
            "возможны отложения и ускоренный износ."
        )
        tag = "bad"
    ring = "чёткое и замкнутое" if d.yellow_ring_continuous else "размытое или неполное"
    text += (
        f" Жёлтое кольцо присадок (зона A) — {d.yellow_ring_score:.0f}%, {ring}. "
        f"Остаточный ресурс присадок оценивается в {d.additive_resource_pct:.0f}%."
    )
    return text, tag


def _contamination_paragraph(d) -> tuple[str, str]:
    tag = c.tag_ci(d.contamination_index)
    if d.contamination_index < 30:
        level, detail = "низкая", "загрузка сажой минимальна, ядро пятна однородное"
    elif d.contamination_index < 55:
        level, detail = "умеренная", "в масле накопилась заметная, но ещё контролируемая сажа"
    else:
        level, detail = "высокая", "масло сильно загружено твёрдыми продуктами горения"
    text = (
        f"Загрузка сажой и нагаром — {level} (CI = {d.contamination_index:.0f}/100): "
        f"{detail}. Состояние ядра: {d.core_status}."
    )
    return text, tag


def _fuel_water_paragraph(d) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    fuel_tag = c.tag_fuel(d.fuel_estimate_pct, d.fuel_confidence)
    if d.fuel_confidence == "низкая":
        parts.append(
            (
                f"Признаки разбавления топливом (~{d.fuel_estimate_pct:.1f}%, {d.fuel_status}) "
                f"оценены с низкой достоверностью — ориентируйтесь на лабораторию при подозрении.",
                fuel_tag,
            )
        )
    elif d.fuel_estimate_pct < 2.5:
        parts.append(
            (f"Разбавление топливом не выявлено (~{d.fuel_estimate_pct:.1f}%, {d.fuel_status}).", fuel_tag)
        )
    else:
        parts.append(
            (
                f"Есть признаки разбавления топливом (~{d.fuel_estimate_pct:.1f}%, {d.fuel_status}, "
                f"достоверность {d.fuel_confidence}).",
                fuel_tag,
            )
        )

    water_tag = c.tag_water(
        d.water_estimate_pct, d.water_confidence, d.dispersion_index, d.contamination_index
    )
    water_suspicious = (
        d.blotter_code_water >= 2 and d.dispersion_index > 0.7 and d.contamination_index < 30
    )
    if d.water_confidence == "низкая" or water_suspicious:
        suffix = (
            ": при отличных DS и CI это скорее артефакт края пятна или фото, "
            "чем реальное содержание воды."
            if water_suspicious
            else " — не используйте для решений без лабораторной проверки."
        )
        parts.append(
            (
                f"Оценка воды/антифриза (~{d.water_estimate_pct:.1f}%, {d.water_status}) "
                f"ненадёжна ({d.water_confidence} достоверность){suffix}",
                water_tag,
            )
        )
    elif d.water_estimate_pct < 0.8:
        parts.append((f"Признаков воды или антифриза не обнаружено (~{d.water_estimate_pct:.1f}%).", water_tag))
    else:
        parts.append(
            (
                f"Есть признаки воды или антифриза (~{d.water_estimate_pct:.1f}%, {d.water_status}, "
                f"достоверность {d.water_confidence}) — рекомендуется проверка методом Karl Fischer.",
                water_tag,
            )
        )
    return parts


def format_result_summary(result: PipelineResult) -> list[DiagSegment]:
    d = result.diagnostics
    dc = result.drop_consistency
    grade, title = _overall_grade(d)
    grade_tag = c.tag_overall_grade(grade)

    segments: list[DiagSegment] = [
        seg("ОБЩИЙ ВЫВОД\n\n"),
        seg(
            title + ".\n\n",
            f"val_{grade_tag}",
            alert=reason_for("overall_grade", grade_tag, d),
        ),
    ]

    disp_text, disp_tag = _dispersion_paragraph(d)
    segments.append(
        seg(disp_text + "\n\n", f"val_{disp_tag}", alert=reason_for("md", disp_tag, d))
    )

    cont_text, cont_tag = _contamination_paragraph(d)
    segments.append(
        seg(cont_text + "\n\n", f"val_{cont_tag}", alert=reason_for("ci", cont_tag, d))
    )

    if d.oxidation_index >= 50:
        ox_text = (
            f"Индекс окисления повышен ({d.oxidation_index:.0f}/100) — масло длительно "
            "в эксплуатации или подвергалось перегреву."
        )
    elif d.oxidation_index >= 25:
        ox_text = (
            f"Признаки окисления умеренные ({d.oxidation_index:.0f}/100) — типично для "
            "масла со средним пробегом."
        )
    else:
        ox_text = (
            f"Окисление минимально ({d.oxidation_index:.0f}/100) — характерно для свежего "
            "или мало отработанного масла."
        )
    ox_tag = c.tag_oxidation(d.oxidation_index)
    segments.append(
        seg(ox_text + "\n\n", f"val_{ox_tag}", alert=reason_for("oxidation", ox_tag, d))
    )

    for text, tag in _fuel_water_paragraph(d):
        metric = "water" if "вод" in text.lower() or "антифриз" in text.lower() else "fuel"
        segments.append(
            seg(text + " ", f"val_{tag}", alert=reason_for(metric, tag, d))
        )
    segments.append(seg("\n\n"))

    segments.append(seg("Blotter-код "))
    for i, char in enumerate(d.blotter_code):
        digit_tag = c.tag_blotter_digit(int(char))
        segments.append(
            seg(
                char,
                digit_tag,
                alert=reason_for("blotter_digit", digit_tag, d, blotter_digit=i),
            )
        )
    segments.append(
        seg(
            f" (сажа/топливо/вода: {d.blotter_code_soot}/{d.blotter_code_fuel}/"
            f"{d.blotter_code_water}): первая цифра — сажа, вторая — топливо, "
            "третья — вода/антифриз (0 — норма, 1 — умеренно, 2 — выраженно).\n"
        )
    )

    if dc.available:
        segments.extend(
            [
                seg("\n"),
                seg(
                    f"На снимке проанализировано {dc.drop_count} капли: согласованность метрик "
                    f"{dc.overall_score:.0f}/100. {dc.summary}\n",
                    f"val_{c.tag_consistency(dc.overall_score)}",
                    alert=reason_for(
                        "consistency",
                        c.tag_consistency(dc.overall_score),
                        d,
                        dc,
                    ),
                ),
            ]
        )

    rec_tag = c.tag_recommendation(d.recommendation)
    segments.extend(
        [
            seg("\nИТОГОВАЯ РЕКОМЕНДАЦИЯ\n\n"),
            seg(
                f"{d.recommendation.capitalize()}.\n",
                f"val_{rec_tag}",
                alert=reason_for("recommendation", rec_tag, d),
            ),
        ]
    )

    if grade == "хорошее":
        closing = (
            "По основным надёжным показателям (DS, MD, CI, кольцо присадок) масло "
            "можно считать пригодным к дальнейшей эксплуатации в рамках полевого скрининга."
        )
    elif grade == "удовлетворительное":
        closing = (
            "Масло пока работоспособно, но ресурс присадок и диспергирование снижаются — "
            "рекомендуется плановый контроль и подготовка к замене."
        )
    elif grade == "на грани":
        closing = (
            "Показатели близки к предельным: целесообразна замена масла в ближайшее "
            "техобслуживание и лабораторная проверка при сомнениях."
        )
    else:
        closing = (
            "Показатели критические: диспергирование и присадки не справляются — "
            "замена масла необходима."
        )
    segments.extend(
        [
            seg(closing + "\n\n", f"val_{grade_tag}", alert=reason_for("overall_grade", grade_tag, d)),
            seg(
                f"Анализ выполнен для капли {result.selected_candidate + 1} "
                f"из {len(result.candidates)}.\n\n"
            ),
            seg("«Кратко» / «Подробно» — вернуться к числовым метрикам.\n"),
        ]
    )
    return segments


def format_detailed_diagnostics(result: PipelineResult) -> list[DiagSegment]:
    d = result.diagnostics
    dc = result.drop_consistency
    ring = "чёткое" if d.yellow_ring_continuous else "размытое"
    segments: list[DiagSegment] = [
        seg("ПОДРОБНЫЙ ОТЧЁТ\n\n"),
        seg("Рекомендация: "),
        _val(f"{d.recommendation}\n", c.tag_recommendation(d.recommendation), "recommendation", d),
        seg(
            _sub(
                "Итоговое решение по капельному тесту: можно ли продолжать эксплуатацию,",
                "нужен ли контроль или замена масла.",
            )
            + "\n",
            "hint",
        ),
        seg(f"Капля {result.selected_candidate + 1} из {len(result.candidates)}\n\n"),
        seg("ОСНОВНЫЕ МЕТРИКИ\n\n"),
    ]

    _append_metric(
        segments,
        "DS (дисперсия)",
        f"{d.dispersion_index:.2f}  (d={d.red_diameter_px:.0f}px, D={d.green_diameter_px:.0f}px)",
        c.tag_ds(d.dispersion_index),
        "Коэффициент диспергирующей способности по ASTM: DS = 1 − (d/D)².",
        "d — диаметр тёмного ядра (зона C), D — диаметр всей окрашенной области.",
        "Чем выше DS, тем лучше сажа размазана по пятну, а не скоплена в центре.",
        ">0.7 — отлично; 0.35–0.7 — наблюдение; <0.35 — критично.",
        _ds_comment(d.dispersion_index),
        metric="ds",
        d=d,
    )
    segments.append(seg("\n"))
    _append_metric(
        segments,
        "MD (Merit of Dispersancy)",
        f"{d.merit_of_dispersancy:.0f}/100",
        c.tag_md(d.merit_of_dispersancy),
        "Индекс качества диспергирования (0–100) по логике ASTM D7899.",
        "Учитывает DS, жёлтое кольцо присадок и шлам в ядре.",
        "≥80 — отлично; 60–79 — хорошо; 40–59 — удовлетворительно; <40 — низко.",
        _md_comment(d.merit_of_dispersancy),
        metric="md",
        d=d,
    )
    segments.append(seg("\n"))
    _append_metric(
        segments,
        "CI (загрузка сажой)",
        f"{d.contamination_index:.0f}/100  (шлам {d.sludge_ratio * 100:.1f}%)",
        c.tag_ci(d.contamination_index),
        "Contamination Index — насколько масло загружено сажой и нагаром.",
        "Считается по площади и темноте зоны C (центр) и доле шлама.",
        "<30 — мало; 30–55 — умеренно; >55 — много сажи.",
        _ci_comment(d.contamination_index),
        metric="ci",
        d=d,
    )
    segments.append(seg("\n"))
    _append_blotter_code(segments, d)
    segments.append(
        seg(
            _sub(
                "Трёхзначный качественный код полевого blotter-теста.",
                "1-я цифра — сажа; 2-я — топливо; 3-я — вода/антифриз.",
                "0 = норма, 1 = умеренно, 2 = выраженно.",
                "Если DS и CI хорошие, а 3-я цифра = 2 — часто это артефакт фото, не вода.",
            )
            + "\n",
            "hint",
        )
    )
    segments.append(seg("\n"))
    _append_metric(
        segments,
        "Ресурс присадок",
        f"{d.additive_resource_pct:.0f}%",
        c.tag_additive(d.additive_resource_pct),
        "Сводный индекс остаточного ресурса присадочного пакета (не лабораторный TBN).",
        "~40% — кольцо A, ~45% — DS, ~15% — однородность ядра.",
        "Показывает, насколько присадки ещё выполняют защитную функцию.",
        metric="additive",
        d=d,
    )
    segments.append(seg("\n"))
    ring_tag = c.tag_ring_score(d.yellow_ring_score, d.yellow_ring_continuous)
    _append_choice_scale(
        segments,
        "Жёлтое кольцо A",
        f"{d.yellow_ring_score:.0f}% — {ring}",
        ring_tag,
        [
            ("чёткое", "чёткое — присадки работают"),
            ("размытое", "размытое — присадки ослабевают"),
        ],
        ring,
        "Зона A (ауреола) — кольцо вокруг ядра. Процент — замкнутость и ровность кольца.",
        metric="ring",
        d=d,
    )
    segments.append(seg("\n"))
    core_tag = c.tag_core_status(d.core_status)
    _append_choice_scale(
        segments,
        "Ядро (зона C)",
        d.core_status,
        core_tag,
        [
            ("однородное", "ядро однородное — хорошо, без скоплений"),
            ("неоднородное", "ядро неоднородное — неравномерность"),
            ("вкрапления", "есть тёмные вкрапления — начало шлама"),
            ("критическ", "критическое скопление шлама"),
        ],
        _core_match_key(d.core_status),
        "Центр пятна, где скапливается нерастворимая сажа.",
        metric="core",
        d=d,
    )
    segments.append(seg("\n"))
    _append_metric(
        segments,
        "Окисление",
        f"{d.oxidation_index:.0f}/100",
        c.tag_oxidation(d.oxidation_index),
        "Индекс окисления масла по пожелтению/потемнению зон A и D (цвет Lab).",
        "Высокое значение — масло дольше в эксплуатации или перегрето.",
        "Низкое — типично для свежего масла.",
        metric="oxidation",
        d=d,
    )
    segments.append(seg("\n"))
    _append_metric(
        segments,
        "Симметрия пятна",
        f"{d.symmetry_index:.0f}/100",
        c.tag_symmetry(d.symmetry_index),
        "Насколько пятно близко к идеальному кругу.",
        "Низкая симметрия часто означает кривое нанесение капли, складку бумаги",
        "или блик — тогда % воды и топлива менее достоверны.",
        metric="symmetry",
        d=d,
    )
    segments.extend([seg("\n"), seg("ЗОНЫ (радиус / площадь)\n\n")])
    zone_specs = (
        ("C", d.zone_radius_c_pct, d.zone_area_c_pct, "Центр — сажа, нагар, нерастворимые частицы."),
        ("A", d.zone_radius_a_pct, d.zone_area_a_pct, "Ауреола — зона работы диспергирующих и антиокислительных присадок."),
        ("D", d.zone_radius_d_pct, d.zone_area_d_pct, "Диффузия — область, куда диспергированы мелкие частицы сажи."),
        (
            "T",
            d.zone_radius_t_pct,
            d.zone_area_t_pct,
            "Прозрачная кромка — растворитель, лёгкие фракции, признак разбавления топливом. "
            "Узкая кромка (~10–12% радиуса) — норма.",
        ),
    )
    for name, r_pct, a_pct, hint in zone_specs:
        segments.append(seg(f"• {name}: {r_pct:.0f}% / {a_pct:.1f}%\n"))
        segments.append(seg(_sub(hint) + "\n\n", "hint"))

    segments.append(seg("ТОПЛИВО И ВОДА (полевой скрининг)\n\n"))
    _append_metric(
        segments,
        "Топливо",
        f"~{d.fuel_estimate_pct:.1f}% — {d.fuel_status}, достоверность: {d.fuel_confidence}",
        c.tag_fuel(d.fuel_estimate_pct, d.fuel_confidence),
        "Оценка разбавления дизельным топливом по ширине зоны T.",
        "Не заменяет хроматографию (GC) в лаборатории.",
        "<2.5% — норма; 2.5–6% — слегка разжижено; >6% — избыток топлива.",
        metric="fuel",
        d=d,
    )
    _append_metric(
        segments,
        "Вода",
        f"~{d.water_estimate_pct:.1f}% — {d.water_status}, достоверность: {d.water_confidence}",
        c.tag_water(d.water_estimate_pct, d.water_confidence, d.dispersion_index, d.contamination_index),
        "Оценка по рваности края зоны T и отклонению пятна от круга.",
        "Самый ненадёжный показатель: чувствителен к ручке, складкам, бликам.",
        "Не заменяет метод Karl Fischer. При низкой достоверности не принимайте решений.",
        metric="water",
        d=d,
    )
    if dc.available:
        segments.extend([seg("\n"), seg(f"СОГЛАСОВАННОСТЬ КАПЕЛЬ ({dc.drop_count} шт.)\n\n")])
        _append_metric(
            segments,
            "Общая оценка",
            f"{dc.overall_score:.0f}/100",
            c.tag_consistency(dc.overall_score),
            "Насколько метрики совпадают между каплями на одном снимке.",
            "≥85 — выводы надёжны; 65–84 — доверяйте DS/MD; <65 — только DS, MD, CI.",
            metric="consistency",
            d=d,
            dc=dc,
        )
        segments.append(seg("\n"))
        if dc.ds_spread > 0.2 or dc.md_spread > 20:
            spread_tag = "bad"
        elif dc.ds_spread > 0.08 or dc.md_spread > 8:
            spread_tag = "warn"
        else:
            spread_tag = "good"
        _append_metric(
            segments,
            "Разброс DS / MD",
            f"DS {dc.ds_spread:.2f}, MD {dc.md_spread:.1f}",
            spread_tag,
            "Малый разброс — стабильный результат. Большой — проверьте кроп и освещение.",
            metric="spread",
            d=d,
            dc=dc,
        )
        if dc.fuel_spread > 4 or dc.water_spread > 3:
            fw_tag = "bad"
        elif dc.fuel_spread > 1.5 or dc.water_spread > 1:
            fw_tag = "warn"
        else:
            fw_tag = "good"
        _append_metric(
            segments,
            "Разброс топлива / воды",
            f"топливо {dc.fuel_spread:.1f}%, вода {dc.water_spread:.1f}%",
            fw_tag,
            dc.summary,
            metric="spread_fw",
            d=d,
            dc=dc,
        )

    segments.extend(
        [
            seg("\n\nСПРАВКА\n"),
            seg(
                _sub(
                    "Надёжные для решений: DS, MD, CI, кольцо A, ресурс присадок.",
                    "Ориентировочные: % топлива и % воды (смотрите достоверность).",
                    "Для точных значений — лаборатория: GC (топливо), Karl Fischer (вода), ICP (металлы).",
                    "Цвет значения: зелёный — норма, жёлтый — погранично, красный — опасно.",
                    "Серый текст — справка; ▶ — ваш текущий вариант.",
                    "Клик по красному значению — причина и рекомендация.",
                )
                + "\n",
                "hint",
            ),
        ]
    )
    return segments


def format_property_loss_report(result: PipelineResult) -> list[DiagSegment]:
    from pipeline.property_loss import compute_property_loss

    d = result.diagnostics
    report = compute_property_loss(d)
    segments: list[DiagSegment] = [
        seg("ПОТЕРЯ СВОЙСТВ МАСЛА\n\n"),
        seg("Оценка относительно свежего масла (100% = исходные свойства).\n"),
        seg("Полевой расчёт по капельному тесту — не заменяет лабораторию.\n\n"),
        seg("Капля "),
        seg(f"{result.selected_candidate + 1} из {len(result.candidates)}\n\n"),
        seg("Сводка: "),
        _val(
            f"сохранено ~{report.overall_retention_pct:.0f}%, ",
            c.tag_retention(report.overall_retention_pct),
            "overall_retention",
            d,
            retention_pct=report.overall_retention_pct,
        ),
        _val(
            f"потеря ~{report.overall_loss_pct:.0f}%\n",
            c.tag_loss(report.overall_loss_pct),
            "loss",
            d,
            property_title="Сводка свойств",
            loss_pct=report.overall_loss_pct,
        ),
        seg(
            f"{report.summary}\n\n",
            f"val_{c.tag_retention(report.overall_retention_pct)}",
            alert=reason_for(
                "overall_retention",
                c.tag_retention(report.overall_retention_pct),
                d,
                retention_pct=report.overall_retention_pct,
            ),
        ),
        seg("ПО СВОЙСТВАМ\n\n"),
    ]

    for prop in report.properties:
        ret_tag = c.tag_retention(prop.retention_pct)
        loss_tag = c.tag_loss(prop.loss_pct)
        segments.append(seg(f"• {prop.title}\n"))
        segments.append(seg("    Осталось: "))
        segments.append(
            _val(
                f"{prop.retention_pct:.0f}%",
                ret_tag,
                "retention",
                d,
                property_title=prop.title,
                retention_pct=prop.retention_pct,
            )
        )
        segments.append(seg("  |  Потеря: "))
        segments.append(
            _val(
                f"{prop.loss_pct:.0f}%\n",
                loss_tag,
                "loss",
                d,
                property_title=prop.title,
                loss_pct=prop.loss_pct,
            )
        )
        segments.append(seg(f"    Основа расчёта: {prop.basis}.\n\n", "hint"))

    segments.extend(
        [
            seg("ИНТЕРПРЕТАЦИЯ\n\n"),
            seg(
                _sub(
                    "Диспергирование и моющие свойства — по зонам C/A/D и MD/DS.",
                    "TBN и вязкость — оценочные, не заменяют ASTM D2896 и D445.",
                    "Вязкость снижается при разбавлении топливом и окислении.",
                    "Зелёный — мало потерь (<25%), жёлтый — умеренно, красный — сильная деградация.",
                )
                + "\n\n",
                "hint",
            ),
            seg("«Кратко» / «Подробно» — вернуться к метрикам.\n"),
        ]
    )
    return segments
