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

from typing import Callable, Literal, Sequence

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

# Labeling modulus (independent of geometric step size by default).
# Step stays 9/π unless step_mode couples geometry to m (m/π).
DEFAULT_LABEL_MODULUS: int = 9

# High-signal moduli for sweeps (digit sum, R_3 factor, prime mover, CRT product).
CORE_MODULI: tuple[int, ...] = (9, 37, 111, 333)
SUGGESTED_MODULI: tuple[int, ...] = CORE_MODULI  # alias for backward compatibility

# Extra candidates: 142857 (7), small primes, 3^3, repunit-related 41.
EXTENDED_MODULI: tuple[int, ...] = (7, 13, 27, 41)

# Full circle
TWO_PI: float = 2.0 * np.pi

StepMode = Literal["nine_over_pi", "m_over_pi"]

MappingMethod = Literal[
    "step_index",
    "angle_bin",
    "sin_dr",
    "cos_dr",
    "doubling_cycle",
    "mod",
    "paired",
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


def modular_label(k: int, modulus: int = DEFAULT_LABEL_MODULUS) -> int:
    """Plain modular residue of ``k`` — the flexible labeling primitive.

    Returns ``k % modulus`` in ``{0, 1, …, modulus - 1}``. This is the
    natural label when the digit-sum story of mod 9 no longer applies.

    Parameters
    ----------
    k :
        Integer to reduce (step index, scaled value, …).
    modulus :
        Positive modulus ``m``.

    Returns
    -------
    int
        Residue in ``0 .. m-1``.
    """
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"modular_label expects int, got {type(k).__name__}")
    return int(k) % int(modulus)


def doubling_orbit(
    start: int = 1,
    modulus: int = DEFAULT_LABEL_MODULUS,
    max_steps: int | None = None,
) -> tuple[list[int], int]:
    """Generalized vortex doubling circuit: ``x → 2x (mod m)``.

    Walks from ``start`` until a residue repeats (cycle detected) or
    ``max_steps`` is hit. Classic vortex is ``modulus=9, start=1`` →
    ``[1, 2, 4, 8, 7, 5]`` with cycle length 6. For ``modulus=37`` and
    start 1, 2 is a primitive root so the orbit has length 36.

    Parameters
    ----------
    start :
        Initial residue.
    modulus :
        Positive modulus ``m``.
    max_steps :
        Optional hard cap (default ``m + 1``, enough to detect any cycle).

    Returns
    -------
    orbit, cycle_length :
        List of distinct residues until first repeat, and ``len(orbit)``
        when a cycle closes (or the truncated length if capped).
    """
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    if max_steps is None:
        max_steps = int(modulus) + 1
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    seen: dict[int, int] = {}
    steps: list[int] = []
    x = modular_label(start, modulus)
    for i in range(max_steps):
        if x in seen:
            break
        seen[x] = i
        steps.append(x)
        x = (2 * x) % modulus
    return steps, len(steps)


def doubling_cycle_structure(modulus: int = DEFAULT_LABEL_MODULUS) -> dict:
    """Full cycle decomposition of multiplication-by-2 on ``Z/mZ``.

    Useful when sweeping moduli: reports every orbit, its length, and
    whether residue 0 is a fixed point (always ``0 → 0``).

    Parameters
    ----------
    modulus :
        Positive modulus ``m``.

    Returns
    -------
    dict
        Keys: ``modulus``, ``cycles`` (list of residue lists, each a
        full cycle), ``cycle_lengths``, ``num_cycles``, ``orbit_from_1``
        (transient+cycle walk from 1), ``length_from_1``.
    """
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")

    visited: set[int] = set()
    cycles: list[list[int]] = []

    for seed in range(modulus):
        if seed in visited:
            continue
        # Follow until we hit a known residue or close a loop.
        path: list[int] = []
        path_index: dict[int, int] = {}
        x = seed
        while x not in visited and x not in path_index:
            path_index[x] = len(path)
            path.append(x)
            x = (2 * x) % modulus

        if x in path_index:
            # New cycle starts at first repeat within this path.
            cyc = path[path_index[x] :]
            cycles.append(cyc)
            visited.update(path)
        else:
            # Merged into a previously visited component (tail into known cycle).
            visited.update(path)

    orbit_1, len_1 = doubling_orbit(1, modulus)
    lengths = [len(c) for c in cycles]
    return {
        "modulus": int(modulus),
        "cycles": cycles,
        "cycle_lengths": lengths,
        "num_cycles": len(cycles),
        "orbit_from_1": orbit_1,
        "length_from_1": len_1,
    }


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


def modulus_sweep_report(
    moduli: Sequence[int] | None = None,
) -> list[dict]:
    """Summarize doubling structure for a list of moduli (default suggested set).

    Returns one :func:`doubling_cycle_structure` dict per modulus, sorted
    by the given order (default :data:`SUGGESTED_MODULI`).
    """
    if moduli is None:
        moduli = SUGGESTED_MODULI
    return [doubling_cycle_structure(int(m)) for m in moduli]


def resolve_moduli(
    extended: bool = False,
    moduli: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Choose modulus list for sweeps.

    Parameters
    ----------
    extended :
        If True and ``moduli`` is None, include :data:`EXTENDED_MODULI`
        (7, 13, 27, 41) after the core set.
    moduli :
        Explicit list overrides ``extended``.
    """
    if moduli is not None:
        return tuple(int(m) for m in moduli)
    if extended:
        # Preserve order; de-dupe if lists ever overlap.
        seen: set[int] = set()
        out: list[int] = []
        for m in (*CORE_MODULI, *EXTENDED_MODULI):
            if m not in seen:
                seen.add(m)
                out.append(m)
        return tuple(out)
    return tuple(CORE_MODULI)


def step_radians_for(
    modulus: int = DEFAULT_LABEL_MODULUS,
    step_mode: StepMode | str = "nine_over_pi",
    explicit: float | None = None,
) -> float:
    """Resolve geometric arc step from a step mode.

    Parameters
    ----------
    modulus :
        Used when ``step_mode == "m_over_pi"`` → ``m/π``.
    step_mode :
        - ``"nine_over_pi"`` (default): classic ``9/π`` (geometry decoupled from m).
        - ``"m_over_pi"``: couple winding rate to the modulus → ``m/π``.
    explicit :
        If set, overrides the mode and returns this value (radians).

    Returns
    -------
    float
        Arc step in radians.
    """
    if explicit is not None:
        return float(explicit)
    mode = str(step_mode).strip().lower().replace("-", "_")
    # Accept CLI aliases
    aliases = {
        "nine_over_pi": "nine_over_pi",
        "9_over_pi": "nine_over_pi",
        "default": "nine_over_pi",
        "m_over_pi": "m_over_pi",
        "mod_over_pi": "m_over_pi",
    }
    mode = aliases.get(mode, mode)
    if mode == "nine_over_pi":
        return float(DEFAULT_STEP_RADIANS)
    if mode == "m_over_pi":
        if modulus <= 0:
            raise ValueError(f"modulus must be positive for m/π step, got {modulus}")
        return float(modulus) / float(np.pi)
    raise ValueError(
        f"Unknown step_mode {step_mode!r}. Use 'nine_over_pi' or 'm_over_pi'."
    )


def paired_label(
    k: int,
    modulus: int = 37,
) -> tuple[int, int, int]:
    """CRT-style paired label: classic digital root + residue mod ``m``.

    Returns ``(digital_root, k % m, packed)`` where packed encodes both for
    a single colormap channel::

        packed = (dr - 1) * m + r    # dr ∈ 1..9, r ∈ 0..m-1
        # → packed ∈ 0 .. 9m - 1

    Step 0 uses digital root 9 (completion / center convention).

    Parameters
    ----------
    k :
        Step index (or other integer).
    modulus :
        Second component modulus (default 37 — your repunit prime).

    Returns
    -------
    dr, residue, packed : tuple[int, int, int]
    """
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    kk = int(k)
    if kk == 0:
        dr = 9
    else:
        dr = digital_root(kk)
        if dr == 0:
            dr = 9
    residue = modular_label(kk, modulus)
    packed = (dr - 1) * int(modulus) + residue
    return dr, residue, packed


def unpack_paired_label(
    packed: int,
    modulus: int = 37,
) -> tuple[int, int]:
    """Inverse of :func:`paired_label` packing → ``(digital_root, residue)``."""
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    p = int(packed)
    dr = (p // int(modulus)) + 1
    residue = p % int(modulus)
    return dr, residue


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


def _map_step_index(
    theta: float,
    step_index: int | None,
    modulus: int = DEFAULT_LABEL_MODULUS,
    **_kwargs: object,
) -> int:
    """Label by step index.

    For ``modulus == 9`` (classic vortex): digital root, with step 0 → 9.
    For other moduli: plain ``step_index % modulus``.
    """
    if step_index is None:
        raise ValueError("step_index required for method 'step_index'")
    if modulus == 9:
        # Step 0 → digit 9 (completion / center); else digital root.
        if step_index == 0:
            return 9
        return digital_root(step_index)
    return modular_label(step_index, modulus)


def _map_angle_bin(
    theta: float,
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
    **_kwargs: object,
) -> int:
    """Partition the circle into ``modulus`` equal sectors.

    Classic vortex (``modulus == 9``): labels 1–9.
    General ``m``: labels 0 … m-1.
    """
    t = float(np.mod(theta, TWO_PI))
    bin_idx = int(np.floor(t / TWO_PI * float(modulus))) % modulus
    if modulus == 9:
        return bin_idx + 1  # 1..9
    return bin_idx  # 0..m-1


def _map_sin_dr(
    theta: float,
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
    **_kwargs: object,
) -> int:
    """Map scaled |sin(θ)| via digital root (m=9) or modular residue."""
    s = abs(np.sin(theta))
    n = max(1, int(round(s * 999)))
    if modulus == 9:
        return digital_root(n)
    return modular_label(n, modulus)


def _map_cos_dr(
    theta: float,
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
    **_kwargs: object,
) -> int:
    """Map scaled |cos(θ)| via digital root (m=9) or modular residue."""
    c = abs(np.cos(theta))
    n = max(1, int(round(c * 999)))
    if modulus == 9:
        return digital_root(n)
    return modular_label(n, modulus)


def _map_doubling_cycle(
    theta: float,
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
    **_kwargs: object,
) -> int:
    """Assign label from the ×2 orbit starting at 1, by step index."""
    if step_index is None:
        raise ValueError("step_index required for method 'doubling_cycle'")
    if modulus == 9:
        return VORTEX_CYCLE[step_index % len(VORTEX_CYCLE)]
    orbit, _ = doubling_orbit(1, modulus)
    if not orbit:
        return 0
    return orbit[step_index % len(orbit)]


def _map_mod(
    theta: float,
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
    **_kwargs: object,
) -> int:
    """Always plain modular label of the step index (ignores digital-root special case)."""
    if step_index is None:
        raise ValueError("step_index required for method 'mod'")
    return modular_label(step_index, modulus)


def _map_paired(
    theta: float,
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
    **_kwargs: object,
) -> int:
    """Packed (digital_root, k % m) for dual-structure colormaps."""
    if step_index is None:
        raise ValueError("step_index required for method 'paired'")
    _dr, _r, packed = paired_label(step_index, modulus)
    return packed


_MAPPING_REGISTRY: dict[str, Callable[..., int]] = {
    "step_index": _map_step_index,
    "angle_bin": _map_angle_bin,
    "sin_dr": _map_sin_dr,
    "cos_dr": _map_cos_dr,
    "doubling_cycle": _map_doubling_cycle,
    "mod": _map_mod,
    "paired": _map_paired,
}


def register_mapping(name: str, fn: Callable[..., int]) -> None:
    """Register a custom angle→label mapping for :func:`position_to_label`."""
    _MAPPING_REGISTRY[name] = fn


def position_to_label(
    theta: float,
    method: str = "step_index",
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
) -> int:
    """Map an angle (and optional step index) to a modular / vortex label.

    Geometric orbit is independent of ``modulus`` — only the discrete label
    changes. Default geometric step remains ``9/π``; labeling modulus is the
    free parameter for mechanism discovery (9, 37, 111, 333, …).

    Parameters
    ----------
    theta :
        Angle in radians.
    method :
        Mapping strategy:

        - ``"step_index"`` (default): digital root when ``modulus==9``
          (step 0 → 9); else ``step_index % modulus``.
        - ``"mod"``: always ``step_index % modulus`` (no digital-root case).
        - ``"paired"``: packed ``(digital_root(k), k % m)`` CRT-style dual label.
        - ``"angle_bin"``: ``modulus`` equal arcs (1–9 when m=9, else 0..m-1).
        - ``"sin_dr"`` / ``"cos_dr"``: digital root or modular map of scaled trig.
        - ``"doubling_cycle"``: walk the ×2 orbit mod ``m`` by step index.

        Custom methods may be added via :func:`register_mapping`.
    step_index :
        Discrete step number along the orbit (required for some methods).
    modulus :
        Labeling modulus. Geometric step is chosen separately via
        :func:`step_radians_for` (default still ``9/π``).

    Returns
    -------
    int
        Label: classic vortex digit in 1–9 when ``modulus==9`` and method is
        digital-root based; residue in ``0 .. modulus-1``; or packed paired
        code for ``"paired"``.
    """
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    if method not in _MAPPING_REGISTRY:
        known = ", ".join(sorted(_MAPPING_REGISTRY))
        raise ValueError(f"Unknown method {method!r}. Known: {known}")
    label = _MAPPING_REGISTRY[method](
        theta, step_index=step_index, modulus=int(modulus)
    )
    return int(label)


def position_to_vortex_digit(
    theta: float,
    method: str = "step_index",
    step_index: int | None = None,
    modulus: int = DEFAULT_LABEL_MODULUS,
) -> int:
    """Map an angle to a label (alias of :func:`position_to_label`).

    When ``modulus == 9`` and method is a classic vortex map, the result is a
    digit in 1–9. For other moduli, returns the modular label (no 1–9 clamp).
    """
    return position_to_label(
        theta, method=method, step_index=step_index, modulus=modulus
    )


def labels_for_orbit(
    num_steps: int,
    step_radians: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    start_angle: float = 0.0,
    modulus: int = DEFAULT_LABEL_MODULUS,
) -> np.ndarray:
    """Vector of modular / vortex labels for each point of an orbit.

    Geometry uses ``step_radians`` (default ``9/π``). Labeling uses
    ``modulus`` independently.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(num_steps,)``.
    """
    angles = circle_angles(num_steps, step_radians, start_angle)
    return np.array(
        [
            position_to_label(
                float(th), method=method, step_index=i, modulus=modulus
            )
            for i, th in enumerate(angles)
        ],
        dtype=int,
    )


def digits_for_orbit(
    num_steps: int,
    step_radians: float = DEFAULT_STEP_RADIANS,
    method: str = "step_index",
    start_angle: float = 0.0,
    modulus: int = DEFAULT_LABEL_MODULUS,
) -> np.ndarray:
    """Alias of :func:`labels_for_orbit` (historical name for vortex digits)."""
    return labels_for_orbit(
        num_steps,
        step_radians=step_radians,
        method=method,
        start_angle=start_angle,
        modulus=modulus,
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
        "step_radians": DEFAULT_STEP_RADIANS,  # geometry: 9/π unless step_mode changes it
        "step_mode": "nine_over_pi",  # or "m_over_pi" to couple step to modulus
        "label_modulus": DEFAULT_LABEL_MODULUS,
        "mapping_method": "step_index",
        "start_angle": 0.0,
        "style": "dark",  # or "light"
    }
