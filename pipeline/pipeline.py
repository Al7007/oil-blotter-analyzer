"""Оркестратор конвейера обработки капельного теста."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from pipeline.colormap import render_spectral
from pipeline.metrics import analyze, draw_overlay
from pipeline.normalize import enhance_stain, invert_within_drop
from pipeline.preprocess import (
    build_drop_at_point_full,
    build_drop_from_circle,
    compute_stain_map,
    crop_drop,
    detect_paper_mask,
    find_drop_candidates_full_image,
    find_drop_geometry,
)
from pipeline.radial import segment_blotter_zones, zones_to_legacy_dict
from pipeline.types import Diagnostics, DropCandidateInfo, DropConsistency, PipelineResult

ManualRegion = tuple[float, float, float]


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _rgb_to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb, mode="RGB")


def draw_candidate_markers(
    rgb: np.ndarray,
    candidates: list,
    selected_index: int,
) -> np.ndarray:
    marked = rgb.copy()
    for candidate in candidates:
        cx, cy = int(candidate.geometry.center[0]), int(candidate.geometry.center[1])
        radius = max(12, int(candidate.geometry.radius))
        color = (40, 220, 40) if candidate.index == selected_index else (255, 200, 0)
        cv2.circle(marked, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.putText(
            marked,
            str(candidate.index + 1),
            (cx - 8, cy - radius - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return marked


def _merge_candidates(candidates: list, selected) -> tuple[list, int]:
    """Добавляет каплю в список, если рядом ещё нет другой."""
    for i, c in enumerate(candidates):
        if (
            np.hypot(
                c.geometry.center[0] - selected.geometry.center[0],
                c.geometry.center[1] - selected.geometry.center[1],
            )
            < max(c.geometry.radius, selected.geometry.radius) * 0.6
        ):
            return candidates, i

    selected.index = len(candidates)
    merged = list(candidates) + [selected]
    return merged, len(merged) - 1


def _candidates_from_manual_regions(
    rgb: np.ndarray,
    manual_regions: list[ManualRegion],
) -> list:
    return [build_drop_from_circle(cx, cy, radius, rgb) for cx, cy, radius in manual_regions]


def run_pipeline(
    image: Image.Image,
    *,
    contrast_method: str = "clahe",
    clahe_clip: float = 2.5,
    padding_ratio: float = 0.15,
    candidate_index: int = 0,
    click_point: tuple[float, float] | None = None,
    manual_regions: list[ManualRegion] | None = None,
) -> PipelineResult:
    original_rgb = _pil_to_rgb(image)

    if manual_regions:
        candidates = _candidates_from_manual_regions(original_rgb, manual_regions)
        if not candidates:
            raise ValueError("Не задано ни одной области для анализа.")
        idx = min(max(candidate_index, 0), len(candidates) - 1)
    else:
        candidates, _ = find_drop_candidates_full_image(original_rgb)

        if click_point is not None:
            selected = build_drop_at_point_full(click_point[0], click_point[1], original_rgb)
            if selected is None:
                raise ValueError(
                    "Не удалось выделить каплю в этой точке. "
                    "Зажмите мышь на центре пятна и потяните, чтобы задать круг."
                )
            candidates, idx = _merge_candidates(candidates, selected)
        elif candidates:
            idx = min(max(candidate_index, 0), len(candidates) - 1)
        else:
            raise ValueError(
                "Не удалось найти каплю. Зажмите мышь на центре пятна и потяните, "
                "чтобы выделить круглую область."
            )

    for i, c in enumerate(candidates):
        c.index = i
    idx = min(max(idx, 0), len(candidates) - 1)
    selected = candidates[idx]

    paper_mask = detect_paper_mask(original_rgb)
    stain_full = compute_stain_map(original_rgb, paper_mask)

    cropped_rgb, stain, mask = crop_drop(
        original_rgb, stain_full, selected.mask, padding_ratio=padding_ratio
    )

    geometry = find_drop_geometry(stain, mask) or selected.geometry

    enhanced = enhance_stain(stain, geometry.mask, method=contrast_method, clip_limit=clahe_clip)
    inverted = invert_within_drop(enhanced, geometry.mask)

    blotter_zones = segment_blotter_zones(stain, inverted, geometry)
    spectral_rgb = render_spectral(blotter_zones, inverted)

    diagnostics = analyze(blotter_zones, inverted, rgb=cropped_rgb)
    zone_dict = zones_to_legacy_dict(blotter_zones)
    overlay_rgb = draw_overlay(spectral_rgb, diagnostics, blotter_zones)
    marked_rgb = draw_candidate_markers(original_rgb, candidates, idx)

    candidate_info = [
        DropCandidateInfo(
            index=c.index,
            center=c.geometry.center,
            radius=c.geometry.radius,
            score=c.score,
        )
        for c in candidates
    ]

    return PipelineResult(
        original=_rgb_to_pil(marked_rgb),
        cropped=_rgb_to_pil(cropped_rgb),
        grayscale=enhanced,
        normalized=enhanced,
        inverted=inverted,
        spectral=_rgb_to_pil(spectral_rgb),
        overlay=_rgb_to_pil(overlay_rgb),
        diagnostics=diagnostics,
        zone_masks=zone_dict,
        candidates=candidate_info,
        selected_candidate=idx,
        drop_consistency=DropConsistency(available=False),
    )
