"""Validation: lid-driven cavity Re=100 and Re=1000 vs Ghia et al. 1982.

Acceptance gates (v0.05/v0.1 before CN filing):
- Re=100:  L2 error on u-velocity centerline < 1%
- Re=1000: L2 error on u-velocity centerline < 2%
Grid convergence study required: run at 32², 64², 128² before reporting final error.
"""

import pathlib

import numpy as np
import pytest
import torch


GHIA_DIR = pathlib.Path(__file__).parent / "ghia1982"


@pytest.mark.skip(reason="NavierStokes2D not yet implemented (v0.05 target)")
@pytest.mark.slow
@pytest.mark.validation
def test_lid_driven_cavity_re100():
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    ref = np.loadtxt(GHIA_DIR / "re100_u.csv", delimiter=",", skiprows=1)
    y_ref = ref[:, 0]
    u_ref = ref[:, 1]

    solver = NavierStokes2D(reynolds_number=100, grid=(64, 64), device="cpu")
    u, p = solver.solve_steady(sdf=None, inlet_velocity=0.0)

    # Extract u-velocity along vertical centerline x=0.5
    nx = 64
    mid = nx // 2
    u_center = u[0, :, mid].detach().numpy()
    ny = 64
    y_center = np.linspace(1 / (2 * ny), 1 - 1 / (2 * ny), ny)

    u_interp = np.interp(y_ref, y_center, u_center)
    l2_err = np.sqrt(np.mean((u_interp - u_ref) ** 2)) / np.sqrt(np.mean(u_ref ** 2))
    assert l2_err < 0.01, f"Re=100 L2 error {l2_err:.4f} exceeds 1% gate"


@pytest.mark.skip(reason="NavierStokes2D not yet implemented (v0.05 target)")
@pytest.mark.slow
@pytest.mark.validation
def test_lid_driven_cavity_re1000():
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    ref = np.loadtxt(GHIA_DIR / "re1000_u.csv", delimiter=",", skiprows=1)
    y_ref = ref[:, 0]
    u_ref = ref[:, 1]

    solver = NavierStokes2D(reynolds_number=1000, grid=(128, 128), device="cpu")
    u, p = solver.solve_steady(sdf=None, inlet_velocity=0.0)

    nx = 128
    mid = nx // 2
    u_center = u[0, :, mid].detach().numpy()
    ny = 128
    y_center = np.linspace(1 / (2 * ny), 1 - 1 / (2 * ny), ny)

    u_interp = np.interp(y_ref, y_center, u_center)
    l2_err = np.sqrt(np.mean((u_interp - u_ref) ** 2)) / np.sqrt(np.mean(u_ref ** 2))
    assert l2_err < 0.02, f"Re=1000 L2 error {l2_err:.4f} exceeds 2% gate"
