"""Tests for practical orbit metrics (src.analysis)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (
    angular_uniformity_score,
    compute_all_metrics,
    label_progression_smoothness,
    metrics_for_experiment,
    metrics_sweep,
    sector_label_dispersion,
    symmetry_score,
)
from src.core import TWO_PI, circle_angles, labels_for_orbit, step_radians_for


class TestAngularUniformity:
    def test_perfectly_uniform_high(self):
        # Equal samples in 12 sectors
        n_sec = 12
        angles = []
        for s in range(n_sec):
            mid = (s + 0.5) * TWO_PI / n_sec
            angles.extend([mid] * 20)
        u, counts = angular_uniformity_score(angles, n_sectors=n_sec)
        assert u > 0.99
        assert counts.tolist() == [20] * n_sec

    def test_clustered_low(self):
        angles = np.zeros(100)  # all at 0
        u, counts = angular_uniformity_score(angles, n_sectors=12)
        assert u < 0.5
        assert counts[0] == 100


class TestLabelProgression:
    def test_monotonic_labels_high(self):
        # Labels increase with angle
        n = 100
        angles = np.linspace(0, TWO_PI, n, endpoint=False)
        labels = np.arange(n)
        s = label_progression_smoothness(labels, angles)
        assert s > 0.95

    def test_random_lower_than_monotonic(self):
        rng = np.random.default_rng(0)
        n = 200
        angles = np.linspace(0, TWO_PI, n, endpoint=False)
        mono = label_progression_smoothness(np.arange(n), angles)
        rnd = label_progression_smoothness(rng.integers(0, 37, size=n), angles)
        assert mono > rnd


class TestSectorDispersion:
    def test_pure_sectors_high_purity(self):
        # Each sector one label
        n_sec = 8
        angles = []
        labels = []
        for s in range(n_sec):
            mid = (s + 0.5) * TWO_PI / n_sec
            angles.extend([mid] * 10)
            labels.extend([s] * 10)
        d = sector_label_dispersion(labels, angles, n_sectors=n_sec)
        assert d["purity"] > 0.99
        assert d["mean_normalized_entropy"] < 0.01


class TestComposite:
    def test_symmetry_bounds(self):
        angles = np.linspace(0, TWO_PI, 120, endpoint=False)
        labels = np.arange(120) % 37
        s = symmetry_score(angles, labels)
        assert 0.0 < s <= 1.0

    def test_compute_all_keys(self):
        angles = circle_angles(80)
        labels = labels_for_orbit(80, modulus=37, method="mod")
        r = compute_all_metrics(angles, labels, modulus=37)
        for key in (
            "angular_uniformity",
            "label_progression",
            "sector_purity",
            "symmetry_score",
        ):
            assert key in r
            assert 0.0 <= r[key] <= 1.0 + 1e-9

    def test_metrics_for_experiment_mod37(self):
        r = metrics_for_experiment(37, "m_over_pi", num_steps=200)
        assert r["modulus"] == 37
        assert r["step_mode"] == "m_over_pi"
        assert abs(r["step_radians"] - step_radians_for(37, "m_over_pi")) < 1e-12

    def test_metrics_sweep_core(self):
        rows = metrics_sweep([9, 37], num_steps=100)
        assert len(rows) == 4  # 2 mods × 2 modes
        assert all("symmetry_score" in r for r in rows)

    def test_orbit_stats_includes_metrics(self):
        from src.core import orbit_stats

        r = orbit_stats(37, "nine_over_pi", num_steps=100)
        assert "symmetry_score" in r
        assert "angular_uniformity" in r
        assert "label_progression" in r
