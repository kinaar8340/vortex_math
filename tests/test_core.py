"""Unit tests for vortex_math pure math (src.core)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import (
    DEFAULT_LABEL_MODULUS,
    DEFAULT_STEP_RADIANS,
    SUGGESTED_MODULI,
    TRINITY_DIGITS,
    TWO_PI,
    VORTEX_CYCLE,
    circle_angles,
    circle_positions,
    digital_root,
    digits_for_orbit,
    doubling_cycle_structure,
    doubling_edges,
    doubling_orbit,
    labels_for_orbit,
    modular_label,
    modulus_sweep_report,
    position_to_label,
    position_to_vortex_digit,
    register_mapping,
    trinity_related_sequence,
    vortex_doubling_sequence,
    vortex_number_circle_coords,
)


class TestDigitalRoot:
    def test_single_digits(self):
        for d in range(1, 10):
            assert digital_root(d) == d
        assert digital_root(0) == 0

    def test_multiples_of_nine(self):
        assert digital_root(9) == 9
        assert digital_root(18) == 9
        assert digital_root(99) == 9
        assert digital_root(999) == 9

    def test_known_values(self):
        assert digital_root(10) == 1
        assert digital_root(247) == 4  # 2+4+7=13 → 1+3=4
        assert digital_root(38) == 2

    def test_negative_uses_abs(self):
        assert digital_root(-18) == 9
        assert digital_root(-7) == 7

    def test_type_error(self):
        with pytest.raises(TypeError):
            digital_root(3.5)  # type: ignore[arg-type]


class TestVortexSequences:
    def test_doubling_cycle_prefix(self):
        seq = vortex_doubling_sequence(12)
        assert seq[:6] == list(VORTEX_CYCLE)
        assert seq[6:12] == list(VORTEX_CYCLE)

    def test_doubling_never_hits_trinity(self):
        seq = vortex_doubling_sequence(200)
        assert all(d not in TRINITY_DIGITS for d in seq)
        assert set(seq) == set(VORTEX_CYCLE)

    def test_empty_and_errors(self):
        assert vortex_doubling_sequence(0) == []
        with pytest.raises(ValueError):
            vortex_doubling_sequence(-1)

    def test_trinity_sequence(self):
        seq = trinity_related_sequence(6)
        assert seq == [3, 6, 9, 3, 6, 9]


class TestModularDoubling:
    def test_modular_label(self):
        assert modular_label(10, 9) == 1
        assert modular_label(37, 37) == 0
        assert modular_label(111, 111) == 0
        assert modular_label(-1, 9) == 8  # Python % for negatives
        with pytest.raises(ValueError):
            modular_label(1, 0)

    def test_doubling_orbit_mod9_classic(self):
        orbit, length = doubling_orbit(1, 9)
        assert orbit == [1, 2, 4, 8, 7, 5]
        assert length == 6

    def test_doubling_orbit_mod37_primitive(self):
        # 2 is a primitive root mod 37 → single 36-cycle through nonzero residues
        orbit, length = doubling_orbit(1, 37)
        assert length == 36
        assert len(set(orbit)) == 36
        assert 0 not in orbit

    def test_doubling_orbit_mod111(self):
        orbit, length = doubling_orbit(1, 111)
        assert length == 36
        assert orbit[0] == 1

    def test_cycle_structure_mod9(self):
        info = doubling_cycle_structure(9)
        assert info["modulus"] == 9
        assert info["length_from_1"] == 6
        assert info["num_cycles"] >= 1
        # 0 is always a fixed point under ×2
        assert any(c == [0] for c in info["cycles"])

    def test_sweep_suggested(self):
        rows = modulus_sweep_report()
        assert [r["modulus"] for r in rows] == list(SUGGESTED_MODULI)
        by_m = {r["modulus"]: r for r in rows}
        assert by_m[9]["length_from_1"] == 6
        assert by_m[37]["length_from_1"] == 36
        assert by_m[111]["length_from_1"] == 36


class TestCirclePositions:
    def test_shapes_and_unit_norm(self):
        x, y = circle_positions(50)
        assert x.shape == (50,)
        assert y.shape == (50,)
        norms = np.hypot(x, y)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_start_at_one_zero(self):
        x, y = circle_positions(1)
        np.testing.assert_allclose(x[0], 1.0, atol=1e-12)
        np.testing.assert_allclose(y[0], 0.0, atol=1e-12)

    def test_step_advances(self):
        step = DEFAULT_STEP_RADIANS
        x, y = circle_positions(2, step_radians=step)
        expected_angle = step % TWO_PI
        np.testing.assert_allclose(x[1], np.cos(expected_angle), atol=1e-12)
        np.testing.assert_allclose(y[1], np.sin(expected_angle), atol=1e-12)

    def test_empty(self):
        x, y = circle_positions(0)
        assert len(x) == 0 and len(y) == 0

    def test_angles_match_positions(self):
        n = 30
        angles = circle_angles(n)
        x, y = circle_positions(n)
        np.testing.assert_allclose(x, np.cos(angles))
        np.testing.assert_allclose(y, np.sin(angles))

    def test_default_step_is_nine_over_pi(self):
        assert math.isclose(DEFAULT_STEP_RADIANS, 9.0 / math.pi, rel_tol=0, abs_tol=1e-15)


class TestPositionToVortexDigit:
    def test_step_index_default(self):
        # step 0 → 9; step 1 → 1; step 9 → 9; step 10 → 1
        assert position_to_vortex_digit(0.0, method="step_index", step_index=0) == 9
        assert position_to_vortex_digit(0.0, method="step_index", step_index=1) == 1
        assert position_to_vortex_digit(0.0, method="step_index", step_index=9) == 9
        assert position_to_vortex_digit(0.0, method="step_index", step_index=10) == 1

    def test_angle_bin(self):
        # θ=0 → bin 1; near 2π-ε still high bin
        assert position_to_vortex_digit(0.0, method="angle_bin") == 1
        assert position_to_vortex_digit(TWO_PI / 9 * 0.5, method="angle_bin") == 1
        assert position_to_vortex_digit(TWO_PI / 9 * 1.5, method="angle_bin") == 2
        d = position_to_vortex_digit(TWO_PI * 0.99, method="angle_bin")
        assert d == 9

    def test_doubling_cycle_method(self):
        for i, expected in enumerate(VORTEX_CYCLE):
            assert (
                position_to_vortex_digit(0.0, method="doubling_cycle", step_index=i)
                == expected
            )

    def test_digits_for_orbit_range(self):
        digs = digits_for_orbit(100)
        assert digs.shape == (100,)
        assert digs.min() >= 1 and digs.max() <= 9

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            position_to_vortex_digit(0.0, method="nope")

    def test_custom_mapping(self):
        register_mapping("always_three", lambda theta, step_index=None, **kw: 3)
        assert position_to_vortex_digit(1.0, method="always_three") == 3


class TestLabelModulus:
    """Labeling modulus is independent of geometric step (default 9/π)."""

    def test_default_modulus_is_nine(self):
        assert DEFAULT_LABEL_MODULUS == 9

    def test_mod37_step_index_is_plain_residue(self):
        # step_index method with m≠9 uses k % m (not digital root)
        assert position_to_label(0.0, method="step_index", step_index=37, modulus=37) == 0
        assert position_to_label(0.0, method="step_index", step_index=38, modulus=37) == 1
        assert position_to_label(0.0, method="mod", step_index=10, modulus=9) == 1

    def test_angle_bin_mod37(self):
        # 37 equal sectors, labels 0..36
        d0 = position_to_label(0.0, method="angle_bin", modulus=37)
        assert d0 == 0
        d = position_to_label(TWO_PI * 0.5, method="angle_bin", modulus=37)
        assert 0 <= d < 37

    def test_labels_for_orbit_mod37_range(self):
        labs = labels_for_orbit(50, method="mod", modulus=37)
        assert labs.shape == (50,)
        assert labs.min() >= 0 and labs.max() < 37

    def test_geometry_independent_of_modulus(self):
        # Same default step → same positions; only labels change with m
        x9, y9 = circle_positions(20)
        x37, y37 = circle_positions(20)
        np.testing.assert_allclose(x9, x37)
        np.testing.assert_allclose(y9, y37)
        l9 = labels_for_orbit(20, modulus=9)
        l37 = labels_for_orbit(20, modulus=37, method="mod")
        assert l9.shape == l37.shape == (20,)
        assert l37.min() >= 0 and l37.max() < 37
        # Classic m=9 digital roots are in 1..9; mod labels are not the same map
        assert set(l9).issubset(set(range(1, 10)))

    def test_doubling_cycle_method_mod37(self):
        orbit, _ = doubling_orbit(1, 37)
        for i in range(10):
            assert (
                position_to_label(0.0, method="doubling_cycle", step_index=i, modulus=37)
                == orbit[i % len(orbit)]
            )


class TestVortexLayout:
    def test_nine_rim_coords(self):
        coords = vortex_number_circle_coords(radius=1.0)
        assert set(coords.keys()) == set(range(1, 10))
        for d, (x, y) in coords.items():
            assert math.isclose(math.hypot(x, y), 1.0, abs_tol=1e-9)

    def test_doubling_edges_cycle(self):
        edges = doubling_edges()
        assert len(edges) == 6
        assert edges[0] == (1, 2)
        assert edges[-1] == (5, 1)
