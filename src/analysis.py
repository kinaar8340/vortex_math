"""
Lightweight quantitative metrics for vortex-math orbits.

Designed for (step_mode, modulus, n_steps) experiments on the unit circle:
irrational (or coupled m/π) rotation + modular / vortex labeling.

Core scores (all in ~[0, 1] where higher is "cleaner" for the intended sense):

- **angular_uniformity** — even fill across angular sectors
- **label_progression** — how orderly labels wind when sorted by angle
- **sector_label_entropy** — within-sector label purity (high = mixed/segregated
  scale: normalized mean entropy; see function docs)
- **symmetry_score** — weighted composite of uniformity + progression
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import (
    TWO_PI,
    circle_angles,
    labels_for_orbit,
    step_radians_for,
)

# Default composite weights (tunable)
UNIFORMITY_WEIGHT = 0.6
PROGRESSION_WEIGHT = 0.4


def angular_uniformity_score(
    angles: np.ndarray | Sequence[float],
    n_sectors: int = 12,
) -> tuple[float, np.ndarray]:
    """How evenly points are distributed across equal angular sectors.

    Parameters
    ----------
    angles :
        Sample of angles in radians.
    n_sectors :
        Number of equal bins on ``[0, 2π)``.

    Returns
    -------
    uniformity, counts :
        Score in ``(0, 1]`` (1.0 ≈ perfectly flat occupancy) and the bin counts.
        Uses coefficient of variation: ``uniformity = 1 / (1 + cv)``.
    """
    if n_sectors < 2:
        raise ValueError("n_sectors must be >= 2")
    ang = np.mod(np.asarray(angles, dtype=float), TWO_PI)
    if ang.size == 0:
        raise ValueError("angles must be non-empty")

    counts, _ = np.histogram(ang, bins=n_sectors, range=(0.0, TWO_PI))
    expected = float(ang.size) / float(n_sectors)
    if expected <= 0:
        return 0.0, counts
    cv = float(np.std(counts.astype(float)) / expected)
    uniformity = 1.0 / (1.0 + cv)
    return float(uniformity), counts


def label_progression_smoothness(
    labels: np.ndarray | Sequence[int],
    angles: np.ndarray | Sequence[float],
) -> float:
    """How consistently labels increase (or decrease) when walking by angle.

    Sort points by angle, then look at successive label differences. Score is
    the fraction of nonzero steps that share the dominant direction::

        max(n_increase, n_decrease) / (n_increase + n_decrease)

    High score → labels wind around the circle in an orderly monochromatic
    gradient (the "orderly winding" often visible for m=37). Circular wrap
    at the cut angle is ignored for the first/last pair (open chain).
    """
    labs = np.asarray(labels)
    ang = np.mod(np.asarray(angles, dtype=float), TWO_PI)
    if labs.size != ang.size or labs.size < 2:
        raise ValueError("labels and angles must have same length >= 2")

    order = np.argsort(ang, kind="mergesort")
    sorted_labels = labs[order].astype(float)
    diffs = np.diff(sorted_labels)
    # Also include wrap-around edge (last → first) for a closed circle walk
    wrap = sorted_labels[0] - sorted_labels[-1]
    diffs = np.concatenate([diffs, np.array([wrap])])

    increases = int(np.sum(diffs > 0))
    decreases = int(np.sum(diffs < 0))
    total = increases + decreases
    if total == 0:
        return 1.0
    return float(max(increases, decreases) / total)


def sector_label_dispersion(
    labels: np.ndarray | Sequence[int],
    angles: np.ndarray | Sequence[float],
    n_sectors: int = 12,
) -> dict:
    """How mixed vs pure labels are inside each angular sector.

    For each sector, compute Shannon entropy of the label distribution
    (normalized by ``log(#distinct labels in that sector or globally)``).

    Returns
    -------
    dict
        ``mean_normalized_entropy`` ∈ [0, 1]:
            0 → each occupied sector is label-pure (segregated colors);
            1 → labels fully mixed within sectors.
        ``purity`` = ``1 - mean_normalized_entropy`` (higher = more segregated).
        ``per_sector_entropy``: list of raw entropies.
    """
    labs = np.asarray(labels)
    ang = np.mod(np.asarray(angles, dtype=float), TWO_PI)
    if labs.size != ang.size or labs.size == 0:
        raise ValueError("labels and angles must be same non-zero length")
    if n_sectors < 2:
        raise ValueError("n_sectors must be >= 2")

    sector = np.floor(ang / TWO_PI * n_sectors).astype(int)
    sector = np.clip(sector, 0, n_sectors - 1)
    n_global = max(int(np.unique(labs).size), 1)
    max_h = float(np.log(n_global))

    raw: list[float] = []
    norms: list[float] = []
    for s in range(n_sectors):
        mask = sector == s
        if not np.any(mask):
            continue
        _, counts = np.unique(labs[mask], return_counts=True)
        p = counts.astype(float) / float(counts.sum())
        h = float(-np.sum(p * np.log(p)))
        raw.append(h)
        norms.append(h / max_h if max_h > 0 else 0.0)

    mean_norm = float(np.mean(norms)) if norms else 0.0
    return {
        "mean_normalized_entropy": mean_norm,
        "purity": 1.0 - mean_norm,
        "per_sector_entropy": raw,
        "n_sectors_occupied": len(raw),
    }


def symmetry_score(
    angles: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
    n_sectors: int = 12,
    w_uniformity: float = UNIFORMITY_WEIGHT,
    w_progression: float = PROGRESSION_WEIGHT,
) -> float:
    """Composite score: weighted uniformity + label progression.

    Default weights: 0.6 uniformity + 0.4 progression (tunable).
    """
    w_sum = w_uniformity + w_progression
    if w_sum <= 0:
        raise ValueError("weights must sum to a positive value")
    u, _ = angular_uniformity_score(angles, n_sectors=n_sectors)
    p = label_progression_smoothness(labels, angles)
    return float((w_uniformity * u + w_progression * p) / w_sum)


def compute_all_metrics(
    angles: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
    modulus: int,
    n_sectors: int = 12,
    step_mode: str | None = None,
    method: str | None = None,
    num_steps: int | None = None,
) -> dict:
    """Compute the practical metric suite for one orbit sample.

    Returns
    -------
    dict
        ``modulus``, optional metadata, and:

        - ``angular_uniformity``
        - ``label_progression``
        - ``sector_label_entropy`` (mean normalized entropy; low = pure sectors)
        - ``sector_purity`` (1 − entropy)
        - ``symmetry_score``
        - ``sector_counts`` (list)
    """
    u, counts = angular_uniformity_score(angles, n_sectors=n_sectors)
    prog = label_progression_smoothness(labels, angles)
    disp = sector_label_dispersion(labels, angles, n_sectors=n_sectors)
    sym = symmetry_score(angles, labels, n_sectors=n_sectors)

    out: dict = {
        "modulus": int(modulus),
        "angular_uniformity": u,
        "label_progression": prog,
        "sector_label_entropy": disp["mean_normalized_entropy"],
        "sector_purity": disp["purity"],
        "symmetry_score": sym,
        "n_sectors": int(n_sectors),
        "sector_counts": counts.tolist(),
    }
    if step_mode is not None:
        out["step_mode"] = step_mode
    if method is not None:
        out["method"] = method
    if num_steps is not None:
        out["num_steps"] = int(num_steps)
    return out


def metrics_for_experiment(
    modulus: int,
    step_mode: str = "nine_over_pi",
    num_steps: int = 500,
    method: str = "step_index",
    n_sectors: int = 12,
    explicit_step: float | None = None,
) -> dict:
    """End-to-end: build orbit for (m, step_mode) and compute all metrics."""
    step = step_radians_for(modulus, step_mode=step_mode, explicit=explicit_step)
    angles = circle_angles(num_steps, step)
    labels = labels_for_orbit(
        num_steps, step_radians=step, method=method, modulus=modulus
    )
    result = compute_all_metrics(
        angles,
        labels,
        modulus=modulus,
        n_sectors=n_sectors,
        step_mode=step_mode,
        method=method,
        num_steps=num_steps,
    )
    result["step_radians"] = float(step)
    return result


def metrics_sweep(
    moduli: Sequence[int],
    step_modes: Sequence[str] = ("nine_over_pi", "m_over_pi"),
    num_steps: int = 500,
    method: str = "step_index",
    n_sectors: int = 12,
) -> list[dict]:
    """Run :func:`metrics_for_experiment` over moduli × step_modes."""
    rows: list[dict] = []
    for mode in step_modes:
        for m in moduli:
            rows.append(
                metrics_for_experiment(
                    modulus=int(m),
                    step_mode=mode,
                    num_steps=num_steps,
                    method=method,
                    n_sectors=n_sectors,
                )
            )
    return rows


def format_metrics_table(rows: Sequence[dict]) -> str:
    """Pretty text table for CLI output."""
    lines = [
        f"  {'m':>5}  {'mode':>12}  {'unif':>6}  {'prog':>6}  "
        f"{'purity':>6}  {'sym':>6}",
        "  " + "-" * 52,
    ]
    for r in rows:
        mode = str(r.get("step_mode", "—"))
        lines.append(
            f"  {int(r['modulus']):>5}  {mode:>12}  "
            f"{r['angular_uniformity']:>6.3f}  "
            f"{r['label_progression']:>6.3f}  "
            f"{r['sector_purity']:>6.3f}  "
            f"{r['symmetry_score']:>6.3f}"
        )
    lines.append(
        "  legend: unif=angular uniformity; prog=label progression; "
        "purity=sector label purity; sym=0.6·unif+0.4·prog"
    )
    return "\n".join(lines)
