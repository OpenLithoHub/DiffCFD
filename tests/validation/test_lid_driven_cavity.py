"""Validation: lid-driven cavity Re=100 and Re=1000 vs Ghia et al. 1982.

Acceptance gates (v0.05 milestone):
- Re=100:  L2 RMS error on u-velocity centerline < 1%  (64x64)
- Re=1000: L2 RMS error on u-velocity centerline < 2%  (128x128)
"""

import pathlib

import numpy as np
import pytest
import torch
from scipy.interpolate import interp1d


GHIA_DIR = pathlib.Path(__file__).parent / "ghia1982"


def _load_ghia(filename):
    data = np.loadtxt(GHIA_DIR / filename, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]   # y, u


def _l2_error(solver_ux, ny, ghia_y, ghia_u):
    ux_center = solver_ux[:, ny // 2].detach().numpy()
    y_ux = np.linspace(0.0, 1.0, ny)
    u_at_ghia = interp1d(y_ux, ux_center, kind="linear")(ghia_y)
    return float(np.sqrt(np.mean((u_at_ghia - ghia_u) ** 2)))


@pytest.mark.slow
@pytest.mark.validation
def test_lid_driven_cavity_re100():
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    y_ref, u_ref = _load_ghia("re100_u.csv")
    nx = ny = 64
    solver = NavierStokes2D(
        reynolds_number=100, grid=(nx, ny), device="cpu",
        max_iter=2000, tol=1e-5, alpha_u=0.7, alpha_p=0.3,
    )
    ux, uy, p = solver.solve_steady(sdf=None, inlet_velocity=0.0, lid_velocity=1.0, case="cavity")
    l2 = _l2_error(ux, ny, y_ref, u_ref)
    assert l2 < 0.01, f"Re=100 L2 error {l2*100:.2f}% exceeds 1% gate"


@pytest.mark.slow
@pytest.mark.validation
def test_lid_driven_cavity_re1000():
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    y_ref, u_ref = _load_ghia("re1000_u.csv")
    nx = ny = 128
    solver = NavierStokes2D(
        reynolds_number=1000, grid=(nx, ny), device="cpu",
        max_iter=3000, tol=1e-5, alpha_u=0.5, alpha_p=0.1,
    )
    ux, uy, p = solver.solve_steady(sdf=None, inlet_velocity=0.0, lid_velocity=1.0, case="cavity")
    l2 = _l2_error(ux, ny, y_ref, u_ref)
    assert l2 < 0.02, f"Re=1000 L2 error {l2*100:.2f}% exceeds 2% gate"
