"""Cross-validation of Rust bspline_sdf against Python golden contracts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from diffcfd._diffcfd_rust import bspline_sdf


GOLDEN_POLYGON_PATH = Path(
    __file__).resolve().parents[2].parent / "diff-surrogate" / "tests" / "golden" / "polygon_sdf.npz"


def _circle_polygon(n_verts: int = 32, radius: float = 1.0):
    theta = np.linspace(0, 2 * math.pi, n_verts, endpoint=False)
    vx = radius * np.cos(theta)
    vy = radius * np.sin(theta)
    return vx, vy


def _exact_circle_sdf(gx: np.ndarray, gy: np.ndarray, radius: float = 1.0):
    return np.sqrt(gx ** 2 + gy ** 2) - radius


def _python_polygon_sdf(gx, gy, vx, vy):
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()
    n_verts = len(vx)
    from diff_surrogate.geometry import sdf_from_curve
    import torch

    verts = torch.tensor(np.column_stack([vx, vy]), dtype=torch.float64)
    gx_t = torch.tensor(gx, dtype=torch.float64)
    gy_t = torch.tensor(gy, dtype=torch.float64)
    return sdf_from_curve(
        gx_t, gy_t, verts, softmin_temp=500.0, winding_sharpness=100.0,
    ).numpy()


def _compute_rust_sdf(gx: np.ndarray, gy: np.ndarray, vx: np.ndarray, vy: np.ndarray):
    return bspline_sdf(
        gx.ravel().astype(np.float64),
        gy.ravel().astype(np.float64),
        vx.astype(np.float64),
        vy.astype(np.float64),
    )


class TestGoldenContractGoldenNpz:
    """Tests using the pre-generated golden contract from diff-surrogate."""

    @pytest.fixture(scope="class")
    def golden(self):
        if not GOLDEN_POLYGON_PATH.exists():
            pytest.skip("diff-surrogate golden contract not found")
        return dict(np.load(GOLDEN_POLYGON_PATH))

    def test_sign_convention(self, golden):
        rust = _compute_rust_sdf(
            golden["grid_x"], golden["grid_y"],
            golden["polygon_vx"], golden["polygon_vy"],
        )
        expected = golden["sdf"].ravel()
        assert np.all(rust[expected < 0] < 0), "Rust SDF must be negative inside polygon"
        assert np.all(rust[expected > 0] > 0), "Rust SDF must be positive outside polygon"

    def test_sign_agreement_everywhere(self, golden):
        rust = _compute_rust_sdf(
            golden["grid_x"], golden["grid_y"],
            golden["polygon_vx"], golden["polygon_vy"],
        )
        expected = golden["sdf"].ravel()
        assert np.all(rust * expected >= 0), "Every point must have matching sign"

    def test_boundary_accuracy(self, golden):
        rust = _compute_rust_sdf(
            golden["grid_x"], golden["grid_y"],
            golden["polygon_vx"], golden["polygon_vy"],
        )
        expected = golden["sdf"].ravel()
        near = np.abs(expected) < 0.1
        if near.any():
            assert np.max(np.abs(rust[near] - expected[near])) < 0.1

    def test_far_field_accuracy(self, golden):
        rust = _compute_rust_sdf(
            golden["grid_x"], golden["grid_y"],
            golden["polygon_vx"], golden["polygon_vy"],
        )
        expected = golden["sdf"].ravel()
        far = np.abs(expected) > 0.5
        if far.any():
            rel_err = np.abs(rust[far] - expected[far]) / np.abs(expected[far])
            assert np.max(rel_err) < 0.1


class TestGoldenContractInline:
    """Tests with inline-generated reference (no diff-surrogate dependency)."""

    @pytest.fixture()
    def circle_grid(self):
        H, W = 32, 32
        t = np.linspace(-2, 2, W)
        gx, gy = np.meshgrid(t, t)
        return gx, gy

    def test_sign_inside_outside(self, circle_grid):
        gx, gy = circle_grid
        vx, vy = _circle_polygon(32)
        rust = _compute_rust_sdf(gx, gy, vx, vy).reshape(gx.shape)

        center_dist = np.sqrt(gx ** 2 + gy ** 2)
        clearly_inside = center_dist < 0.7
        clearly_outside = center_dist > 1.5

        assert np.all(rust[clearly_inside] < 0), "Points inside polygon must have negative SDF"
        assert np.all(rust[clearly_outside] > 0), "Points outside polygon must have positive SDF"

    def test_accuracy_vs_analytical_circle(self, circle_grid):
        gx, gy = circle_grid
        vx, vy = _circle_polygon(32)
        rust = _compute_rust_sdf(gx, gy, vx, vy).reshape(gx.shape)

        analytical = _exact_circle_sdf(gx, gy)
        away = np.abs(analytical) > 0.3
        abs_err = np.abs(rust[away] - analytical[away])
        assert np.max(abs_err) < 0.05, (
            f"Max error vs analytical circle SDF: {np.max(abs_err):.6f}"
        )

    def test_symmetry(self, circle_grid):
        gx, gy = circle_grid
        vx, vy = _circle_polygon(32)
        rust = _compute_rust_sdf(gx, gy, vx, vy).reshape(gx.shape)

        assert np.allclose(rust, rust[::-1, ::-1], atol=1e-12), (
            "Circle SDF must be symmetric under 180-degree rotation"
        )

    def test_monotonic_radial(self, circle_grid):
        gx, gy = circle_grid
        vx, vy = _circle_polygon(32)
        rust = _compute_rust_sdf(gx, gy, vx, vy).reshape(gx.shape)

        along_x = rust[gy.shape[0] // 2, :]
        outside_region = gx[gy.shape[0] // 2, :] > 1.5
        if outside_region.any():
            radial_values = along_x[outside_region]
            assert np.all(np.diff(radial_values) > -1e-10), (
                "SDF must be non-decreasing along radial direction outside polygon"
            )

    @pytest.mark.skipif(
        not GOLDEN_POLYGON_PATH.exists(),
        reason="diff-surrogate golden contract not found",
    )
    def test_python_rust_consistency(self, circle_grid):
        gx, gy = circle_grid
        vx, vy = _circle_polygon(32)
        rust = _compute_rust_sdf(gx, gy, vx, vy).reshape(gx.shape)

        python_sdf = _python_polygon_sdf(gx, gy, vx, vy)

        assert np.allclose(rust, python_sdf, atol=0.2, rtol=0.1), (
            f"Max diff Rust vs Python: {np.max(np.abs(rust - python_sdf)):.6f}"
        )

    @pytest.mark.skipif(
        not GOLDEN_POLYGON_PATH.exists(),
        reason="diff-surrogate golden contract not found",
    )
    def test_python_rust_boundary_tight(self, circle_grid):
        gx, gy = circle_grid
        vx, vy = _circle_polygon(32)
        rust = _compute_rust_sdf(gx, gy, vx, vy).reshape(gx.shape)

        python_sdf = _python_polygon_sdf(gx, gy, vx, vy)

        near = np.abs(python_sdf) < 0.1
        if near.any():
            assert np.max(np.abs(rust[near] - python_sdf[near])) < 0.05
