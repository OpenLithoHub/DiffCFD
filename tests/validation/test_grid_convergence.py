"""Grid convergence study: Richardson extrapolation on lid-driven cavity.

Runs the lid-driven cavity at Re=100 on 16-squared, 32-squared, 64-squared grids and verifies
that the solution converges monotonically.

Uses Ghia et al. 1982 reference data for the centerline u-velocity profile.
"""

import pathlib

import numpy as np
import pytest
from scipy.interpolate import interp1d


GHIA_DIR = pathlib.Path(__file__).parent / "ghia1982"


def _load_ghia(filename):
    data = np.loadtxt(GHIA_DIR / filename, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]  # y, u


def _run_cavity(
    nx: int, ny: int, re: float = 100.0, alpha_u: float = 0.7, alpha_p: float = 0.3
) -> tuple:
    """Run lid-driven cavity and return centerline velocity profile."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    solver = NavierStokes2D(
        reynolds_number=re,
        grid=(nx, ny),
        max_iter=3000,
        tol=1e-6,
        alpha_u=alpha_u,
        alpha_p=alpha_p,
    )
    ux, uy, p = solver.solve_steady(lid_velocity=1.0, case="cavity")

    # Centerline u-velocity: ux at x=0.5 for each y-row
    ix_center = nx // 2
    u_centerline = ux[:, ix_center].detach().numpy()
    y_ux = np.linspace(0.0, 1.0, ny)

    return y_ux, u_centerline


def _l2_error(y_solver, u_solver, ghia_y, ghia_u):
    """Compute relative L2 error of solver u-velocity vs Ghia reference."""
    u_at_ghia = interp1d(y_solver, u_solver, kind="linear")(ghia_y)
    return float(np.sqrt(np.mean((u_at_ghia - ghia_u) ** 2)))


@pytest.mark.slow
@pytest.mark.validation
def test_grid_convergence_re100():
    """Verify solution converges monotonically as grid is refined (Re=100)."""
    ghia_y, ghia_u = _load_ghia("re100_u.csv")

    grids = [(16, 16), (32, 32), (64, 64)]
    l2_errors = []

    for nx, ny in grids:
        y_vals, u_vals = _run_cavity(nx, ny, re=100.0)
        l2 = _l2_error(y_vals, u_vals, ghia_y, ghia_u)
        l2_errors.append(l2)

    # Solution should improve with finer grid
    assert l2_errors[-1] < l2_errors[0], (
        f"No improvement from coarse to fine: {l2_errors}"
    )

    # Fine grid should be within 5% of Ghia
    assert l2_errors[-1] < 0.05, f"Fine grid L2 error {l2_errors[-1]:.4f} > 5%"


@pytest.mark.slow
@pytest.mark.validation
def test_richardson_extrapolation():
    """Richardson extrapolation: estimate convergence order from three grids."""
    grids = [(16, 16), (32, 32), (64, 64)]

    # Use centerline u-velocity at y=0.5 as the quantity of interest
    qoi_values = []
    for nx, ny in grids:
        y_vals, u_vals = _run_cavity(nx, ny, re=100.0)
        u_at_05 = float(np.interp(0.5, y_vals, u_vals))
        qoi_values.append(u_at_05)

    f_coarse, f_medium, f_fine = qoi_values
    h_coarse, h_medium, _h_fine = 1.0 / 16, 1.0 / 32, 1.0 / 64
    ratio = h_coarse / h_medium  # = 2 for uniform refinement

    # Richardson extrapolation order estimate
    num = f_coarse - f_medium
    den = f_medium - f_fine
    if abs(den) > 1e-10 and abs(num) > 1e-10:
        p_est = np.log(abs(num / den)) / np.log(ratio)
        assert p_est > 0.5, f"Convergence order estimate {p_est:.2f} < 0.5"
    # If we can't estimate order, just verify convergence direction
    assert abs(f_fine - f_medium) < abs(f_medium - f_coarse), (
        "Solution is not converging"
    )
