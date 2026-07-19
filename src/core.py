"""
Pure math for vortex-math unit-circle mapping.

Vortex math (popularized by Marko Rodin / Tesla-inspired numerology) focuses on:
  - Digital roots (repeated digit sum → single digit 1–9)
  - The doubling circuit: 1 → 2 → 4 → 8 → 7 → 5 → 1  (never hits 3, 6, 9)
  - The 3-6-9 "axis" / control numbers (Trinity)

This module maps positions on the unit circle obtained by stepping
arc length ``step_radians`` (default ``9/π``) to those concepts.
Because ``9/π`` is an irrational multiple of ``2π`` (in the usual sense of
dense rotations on the circle), long orbits fill the circumference densely.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Classic vortex doubling circuit (digital roots of powers of 2).
VORTEX_CYCLE: tuple[int, ...] = (1, 2, 4, 8, 7, 5)

# 3-6-9 "control" / Trinity digits — outside the doubling circuit.
TRINITY_DIGITS: frozenset[int] = frozenset({3, 6, 9})

# Default arc step: 9/π radians per hop on the unit circle.
# (9 related to digital-root base; π ties the circle's geometry.)
DEFAULT_STEP_RADIANS: float = 9.0 / np.pi

# Full circle
TWO_PI: float = 2.0 * np.pi

MappingMethod = Literal[
    "step_index",
    "angle_bin",
    "sin_dr",
    "cos_dr",
    "doubling_cycle",
]


# ---------------------------------------------------------------------------
# Digital root & sequences
# ---------------------------------------------------------------------------


def digital_root(n: int) -> int:
    """Return the single-digit digital root of ``n`` (result in 1–9, or 0 for 0).

    Digital root is the iterative sum of decimal digits until a single digit
    remains. Equivalently for positive integers: ``n % 9``, with multiples of
    9 mapping to 9 (not 0), except ``n == 0`` which maps to 0.

    Parameters
    ----------
    n :
        Integer (negative values use ``abs(n)`` for the digit process).

    Returns
    -------
    int
        Digital root in ``{0, 1, …, 9}``.

    Examples
    --------
    >>> digital_root(1)
    1
    >>> digital_root(18)
    9
    >>> digital_root(247)
    4
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"digital_root expects int, got {type(n).__name__}")
    n = int(abs(n))
    if n == 0:
        return 0
    r = n % 9
    return 9 if r == 0 else r


def vortex_doubling_sequence(length: int = 100) -> list[int]:
    """Generate the repeating vortex doubling cycle ``1-2-4-8-7-5``.

    This is the sequence of digital roots of successive powers of 2:
    ``dr(2^0)=1, dr(2^1)=2, dr(2^2)=4, …``. Digits 3, 6, 9 never appear
    in pure doubling; they form the complementary "axis" of vortex math.

    Parameters
    ----------
    length :
        Number of terms to return (non-negative).

    Returns
    -------
    list[int]
        Sequence of length ``length`` cycling through :data:`VORTEX_CYCLE`.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return []
    cycle = list(VORTEX_CYCLE)
    return [cycle[i % len(cycle)] for i in range(length)]


def trinity_related_sequence(length: int = 100) -> list[int]:
    """Optional companion sequence highlighting 3-6-9 via multiples of 3.

    Digital roots of successive multiples of 3 cycle as ``3-6-9-3-6-9-…``.
    Useful for marking the Trinity axis beside the doubling circuit.

    Parameters
    ----------
    length :
        Number of terms.

    Returns
    -------
    list[int]
        Sequence of digital roots of ``3, 6, 9, 12, …``.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    return [digital_root(3 * (i + 1)) for i in range(length)]


# ---------------------------------------------------------------------------
# Unit-circle positions
# ---------------------------------------------------------------------------


def circle_positions(
    num_steps: int,
    step_radians: float = DEFAULT_STEP_RADIANS,
    start_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` on the unit circle after successive arc steps.

    Position ``k`` is at angle ``start_angle + k * step_radians`` (mod ``2π``):

        x_k = cos(θ_k),  y_k = sin(θ_k)

    With the default ``step_radians = 9/π``, the rotation number relative to
    ``2π`` is irrational, so the orbit is dense on the circle as
    ``num_steps → ∞``.

    Parameters
    ----------
    num_steps :
        Number of points (including the starting point as step 0).
    step_radians :
        Arc length advanced between consecutive points.
    start_angle :
        Initial angle in radians (default 0 = positive x-axis).

    Returns
    -------
    x, y : np.ndarray
        Shape ``(num_steps,)`` coordinates on the unit circle.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if num_steps == 0:
        return np.array([]), np.array([])

    k = np.arange(num_steps, dtype=float)
    theta = np.mod(start_angle + k * step_radians, TWO_PI)
    return np.cos(theta), np.sin(theta)


def circle_angles(
    num_steps: int,
    step_radians: float = DEFAULT_STEP_RADIANS,
    start_angle: float = 0.0,
) -> np.ndarray:
    """Return angles (mod ``2π``) for each step. Same convention as :func:`circle_positions`."""
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if num_steps == 0:
        return np.array([])
    k = np.arange(num_steps, dtype=float)
    return np.mod(start_angle + k * step_radians, TWO_PI)


# ---------------------------------------------------------------------------
# Angle / index → vortex digit mapping
# ---------------------------------------------------------------------------


def _map_step_index(theta: float, step_index: int | None, **_kwargs: object) -> int:
    """Default: digital root of the step index (1-based for non-zero steps)."""
    if step_index is None:
        raise ValueError("step_index required for method 'step_index'")
    # Step 0 → digit 9 (often associated with completion / center);
    # subsequent steps use digital_root of the 1-based index.
    if step_index == 0:
        return 9
    return digital_root(step_index)


def _map_angle_bin(theta: float, step_index: int | None = None, **_kwargs: object) -> int:
    """Partition the circle into 9 equal sectors → digits 1–9."""
    # Normalize to [0, 2π)
    t = float(np.mod(theta, TWO_PI))
    bin_idx = int(np.floor(t / TWO_PI * 9.0))  # 0..8
    return bin_idx + 1  # 1..9


def _map_sin_dr(theta: float, step_index: int | None = None, **_kwargs: object) -> int:
    """Digital root of a scaled |sin(θ)| (exploratory mapping)."""
    # Map |sin| ∈ [0,1] → integer 1..999 then digital root
    s = abs(np.sin(theta))
    n = max(1, int(round(s * 999)))
    return digital_root(n)


def _map_cos_dr(theta: float, step_index: int | None = None, **_kwargs: object) -> int:
    """Digital root of a scaled |cos(θ)| (exploratory mapping)."""
    c = abs(np.cos(theta))
    n = max(1, int(round(c * 999)))
    return digital_root(n)


def _map_doubling_cycle(
    theta: float, step_index: int | None = None, **_kwargs: object
) -> int:
    """Assign digit from the vortex doubling cycle by step index."""
    if step_index is None:
        raise ValueError("step_index required for method 'doubling_cycle'")
    return VORTEX_CYCLE[step_index % len(VORTEX_CYCLE)]


_MAPPING_REGISTRY: dict[str, Callable[..., int]] = {
    "step_index": _map_step_index,
    "angle_bin": _map_angle_bin,
    "sin_dr": _map_sin_dr,
    "cos_dr": _map_cos_dr,
    "doubling_cycle": _map_doubling_cycle,
}


def register_mapping(name: str, fn: Callable[..., int]) -> None:
    """Register a custom angle→digit mapping for :func:`position_to_vortex_digit`."""
    _MAPPING_REGISTRY[name] = fn


def position_to_vortex_digit(
    theta: float,
    method: str = "step_index",
    step_index: int | None = None,
) -> int:
    """Map an angle (and optional step index) to a vortex digit 1–9.

    Parameters
    ----------
    theta :
        Angle in radians.
    method :
        Mapping strategy:

        - ``"step_index"`` (default): digital root of the step index
          (step 0 → 9). Links the orbit order to digital roots.
        - ``"angle_bin"``: ``floor((θ / 2π) * 9) + 1`` — nine equal arcs.
        - ``"sin_dr"`` / ``"cos_dr"``: digital root of scaled |sin| / |cos|.
        - ``"doubling_cycle"``: cycle through 1-2-4-8-7-5 by step index.

        Custom methods may be added via :func:`register_mapping`.
    step_index :
        Discrete step number along the orbit (required for some methods).

    Returns
    -------
    int
        Digit in 1–9.
    """
    if method not in _MAPPING_REGISTRY:
        known = ", ".join(sorted(_MAPPING_REGISTRY))
        raise ValueError(f"Unknown method {method!r}. Known: {known}")
    digit = _MAPPING_REGISTRY[method](theta, step_index=step_index)
    if digit < 1 or digit > 9:
        raise ValueError(f"Mapping {method!r} returned out-of-range digit {digit}")
    return int(digit)


def digits_for_orbit(
    num_steps: int,
    step_radians: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    start_angle: float = 0.0,
) -> np.ndarray:
    """Vector of vortex digits for each point of an orbit.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(num_steps,)`` with values in 1–9.
    """
    angles = circle_angles(num_steps, step_radians, start_angle)
    return np.array(
        [
            position_to_vortex_digit(float(th), method=method, step_index=i)
            for i, th in enumerate(angles)
        ],
        dtype=int,
    )


def vortex_number_circle_coords(
    radius: float = 1.0,
    include_center_nine: bool = True,
) -> dict[int, tuple[float, float]]:
    """Coordinates for the classic 1–9 vortex number layout.

    Digits 1–8 are placed at equal 45° (π/4) spacing starting with 1 at
    the top (standard "clock-like" vortex diagram variants often use
    40° for 1–9 around the rim; here we use equal spacing of the eight
    circuit-adjacent positions and put 9 at the origin when requested).

    A common pedagogical layout places numbers 1–9 equally around the circle
    at 40° intervals. We support that via ``equal_nine=True`` in callers;
    this function returns the equal-40° rim placement for 1–9, with optional
    duplicate center marker for 9.

    Parameters
    ----------
    radius :
        Circle radius for rim digits.
    include_center_nine :
        If True, digit 9 is also available at the origin (callers choose).

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping digit → (x, y). Digits 1–9 on the rim at 40° steps,
        starting with 1 at angle 90° (top) going clockwise or
        counter-clockwise. Convention: counter-clockwise from +x would be
        standard math; for a readable "number circle" we place 1 at top
        and proceed clockwise (matching many vortex diagrams).
    """
    # Place 1 at top (π/2), then every 40° clockwise for digits 1..9.
    coords: dict[int, tuple[float, float]] = {}
    for i, digit in enumerate(range(1, 10)):
        # Clockwise from top: angle = π/2 - i * (40°)
        angle = np.pi / 2.0 - i * (40.0 * np.pi / 180.0)
        coords[digit] = (radius * float(np.cos(angle)), radius * float(np.sin(angle)))
    if include_center_nine:
        # Callers may draw 9 both on rim and as a center special marker.
        pass
    return coords


def doubling_edges() -> list[tuple[int, int]]:
    """Edges of the vortex doubling star: 1→2→4→8→7→5→1."""
    cycle = list(VORTEX_CYCLE)
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def default_config() -> dict:
    """Small config dict for step size and mapping (easy to extend)."""
    return {
        "step_radians": DEFAULT_STEP_RADIANS,
        "mapping_method": "step_index",
        "start_angle": 0.0,
        "style": "dark",  # or "light"
    }
