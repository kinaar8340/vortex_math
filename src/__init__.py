"""Vortex Math Unit Circle Visualization.

Maps unit-circle positions stepped by arc lengths of 9/π radians
onto vortex-math concepts (digital roots, doubling sequence 1-2-4-8-7-5,
and the special role of 3-6-9).
"""

from .analysis import (
    compute_all_metrics,
    metrics_for_experiment,
    metrics_sweep,
    symmetry_score,
)
from .core import (
    CORE_MODULI,
    DEFAULT_LABEL_MODULUS,
    DEFAULT_STEP_RADIANS,
    EXTENDED_MODULI,
    FAMILY_37,
    FAMILY_111,
    SUGGESTED_MODULI,
    TRINITY_DIGITS,
    VORTEX_CYCLE,
    circle_positions,
    digital_root,
    doubling_orbit,
    family_orbit_report,
    labels_for_orbit,
    modular_label,
    orbit_stats,
    paired_label,
    position_to_label,
    position_to_vortex_digit,
    resonance_scan,
    step_radians_for,
    vortex_doubling_sequence,
)

__all__ = [
    "CORE_MODULI",
    "DEFAULT_LABEL_MODULUS",
    "DEFAULT_STEP_RADIANS",
    "EXTENDED_MODULI",
    "FAMILY_37",
    "FAMILY_111",
    "SUGGESTED_MODULI",
    "TRINITY_DIGITS",
    "VORTEX_CYCLE",
    "circle_positions",
    "compute_all_metrics",
    "digital_root",
    "doubling_orbit",
    "family_orbit_report",
    "labels_for_orbit",
    "metrics_for_experiment",
    "metrics_sweep",
    "modular_label",
    "orbit_stats",
    "paired_label",
    "position_to_label",
    "position_to_vortex_digit",
    "resonance_scan",
    "step_radians_for",
    "symmetry_score",
    "vortex_doubling_sequence",
]

__version__ = "0.1.0"
