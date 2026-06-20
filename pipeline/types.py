"""Типы данных конвейера обработки капельного теста."""



from __future__ import annotations



from dataclasses import dataclass, field



import numpy as np

from PIL import Image





@dataclass

class DropCandidateInfo:

    index: int

    center: tuple[float, float]

    radius: float

    score: float





@dataclass

class DropConsistency:

    """Согласованность метрик между каплями на одном снимке."""



    available: bool = False

    drop_count: int = 0

    ds_spread: float = 0.0

    md_spread: float = 0.0

    fuel_spread: float = 0.0

    water_spread: float = 0.0

    overall_score: float = 0.0

    summary: str = "—"





@dataclass

class Diagnostics:

    """Метрики диагностики и текстовый вердикт."""



    dispersion_index: float = 0.0

    red_diameter_px: float = 0.0

    green_diameter_px: float = 0.0

    yellow_ring_score: float = 0.0

    yellow_ring_continuous: bool = False

    core_uniformity: float = 0.0

    sludge_ratio: float = 0.0

    blue_aura_ratio: float = 0.0

    blue_edge_roughness: float = 0.0

    fuel_estimate_pct: float = 0.0

    water_estimate_pct: float = 0.0

    additive_resource_pct: float = 0.0

    merit_of_dispersancy: float = 0.0

    contamination_index: float = 0.0

    blotter_code: str = "000"

    blotter_code_soot: int = 0

    blotter_code_fuel: int = 0

    blotter_code_water: int = 0

    zone_radius_c_pct: float = 0.0

    zone_radius_a_pct: float = 0.0

    zone_radius_d_pct: float = 0.0

    zone_radius_t_pct: float = 0.0

    zone_area_c_pct: float = 0.0

    zone_area_a_pct: float = 0.0

    zone_area_d_pct: float = 0.0

    zone_area_t_pct: float = 0.0

    oxidation_index: float = 0.0

    symmetry_index: float = 0.0

    fuel_confidence: str = "—"

    water_confidence: str = "—"

    fuel_status: str = "—"

    water_status: str = "—"

    core_status: str = "—"

    recommendation: str = "—"

    verdict: str = "—"

    details: list[str] = field(default_factory=list)





@dataclass

class PipelineResult:

    """Результат полного конвейера."""



    original: Image.Image

    cropped: Image.Image

    grayscale: np.ndarray

    normalized: np.ndarray

    inverted: np.ndarray

    spectral: Image.Image

    overlay: Image.Image

    diagnostics: Diagnostics

    zone_masks: dict[str, np.ndarray] = field(default_factory=dict)

    candidates: list[DropCandidateInfo] = field(default_factory=list)

    selected_candidate: int = 0

    drop_consistency: DropConsistency = field(default_factory=DropConsistency)


