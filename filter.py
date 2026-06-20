"""Обратная совместимость: старый API перенаправлен в pipeline."""

from __future__ import annotations

from PIL import Image

from pipeline import run_pipeline


def apply_oil_drop_filter(image: Image.Image, **_kwargs) -> tuple[Image.Image, Image.Image]:
    result = run_pipeline(image)
    intensity = Image.fromarray(result.inverted, mode="L")
    return result.spectral, intensity
