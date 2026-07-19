"""Vortex Math Unit Circle Visualization.

Maps unit-circle positions stepped by arc lengths of 9/π radians
onto vortex-math concepts (digital roots, doubling sequence 1-2-4-8-7-5,
and the special role of 3-6-9).
"""

from .core import (
    digital_root,
    vortex_doubling_sequence,
    circle_positions,
    position_to_vortex_digit,
    VORTEX_CYCLE,
    TRINITY_DIGITS,
    DEFAULT_STEP_RADIANS,
)

__all__ = [
    "digital_root",
    "vortex_doubling_sequence",
    "circle_positions",
    "position_to_vortex_digit",
    "VORTEX_CYCLE",
    "TRINITY_DIGITS",
    "DEFAULT_STEP_RADIANS",
]

__version__ = "0.1.0"
