"""Сегменты текста диагностики с цветовыми тегами."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagSegment:
    text: str
    tag: str | None = None
    alert: str | None = None  # текст всплывающего окна при клике (красные значения)


def seg(text: str, tag: str | None = None, alert: str | None = None) -> DiagSegment:
    return DiagSegment(text, tag, alert)


def join_segments(segments: list[DiagSegment]) -> str:
    return "".join(s.text for s in segments)
