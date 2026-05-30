"""Gradient verification tests for implicit differentiation.

Verifies that matrix-free GMRES implicit differentiation gives exact
gradients. Verified by:
1. torch.autograd.gradcheck on implicit diff backward
2. Cross-validation: implicit diff vs finite differences
"""

import pytest
import torch


@pytest.mark.slow
@pytest.mark.validation
def test_poiseuille_gradcheck():
    """torch.autograd.gradcheck on the implicit diff Poiseuille solve.

    This is the primary gradient verification for the C1 claim.
    Uses double precision (required by gradcheck).
    """
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    grid = (16, 8)
    lx, ly = 2.0, 1.0

    u_inlet = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    solver = NavierStokes2D(
        reynolds_number=1.0,
        grid=grid,
        lx=lx,
        ly=ly,
        backward="implicit_diff",
        tol=1e-8,
    )

    def fn(u_in):
        ux, uy, p = solver.solve_steady(sdf=None, inlet_velocity=u_in, case="channel")
        return solver.pressure_drop(ux, uy, p).unsqueeze(0)

    # gradcheck uses finite differences with eps=1e-6 by default
    # With implicit diff, the gradient should match FD to ~1e-4 precision
    torch.autograd.gradcheck(fn, (u_inlet,), eps=1e-4, atol=1e-3, rtol=1e-3)


@pytest.mark.slow
@pytest.mark.validation
def test_lid_driven_cavity_gradcheck():
    """gradcheck on lid-driven cavity with implicit diff."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    u_lid = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    solver = NavierStokes2D(
        reynolds_number=100,
        grid=(16, 16),
        backward="implicit_diff",
        tol=1e-6,
        max_iter=2000,
    )

    def fn(u_l):
        ux, uy, p = solver.solve_steady(sdf=None, lid_velocity=u_l, case="cavity")
        return p.mean().unsqueeze(0)

    torch.autograd.gradcheck(fn, (u_lid,), eps=1e-4, atol=1e-3, rtol=1e-3)
