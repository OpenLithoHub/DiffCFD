"""Shared pytest fixtures for DiffCFD tests."""

import pytest

from diffcfd.geometry.mesh import CartesianMesh


@pytest.fixture
def small_mesh():
    """8×8 unit-square Cartesian mesh for fast unit tests."""
    return CartesianMesh(nx=8, ny=8, lx=1.0, ly=1.0, device="cpu")


@pytest.fixture
def tiny_mesh():
    """4×4 mesh — minimum size for smoke tests."""
    return CartesianMesh(nx=4, ny=4, lx=1.0, ly=1.0, device="cpu")


@pytest.fixture
def poiseuille_params():
    """Parameters for 2D Poiseuille analytical solution.

    Fully-developed channel flow between y=0 and y=1.
    U_max = 1, μ = 1, L = 1, h = 1.
    Analytical: u(y) = 4·U_max·y·(1-y)
    Analytical ΔP/L = 8·μ·U_max = 8
    Analytical ∂ΔP/∂U_inlet = 12·μ·L / h² = 12 (after conversion for mean velocity)
    """
    return {
        "reynolds_number": 1.0,
        "grid": (32, 16),
        "lx": 4.0,
        "ly": 1.0,
        "u_inlet": 1.0,
        "mu": 1.0,
    }
