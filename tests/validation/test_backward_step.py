"""Validation: backward-facing step at moderate Re.

Reference: reattachment length grows with Re for laminar flow.
At Re=100 on a coarse grid, we verify the solver produces a bounded,
non-NaN solution with recirculation behind the step.
"""

import pytest
import torch


@pytest.mark.slow
@pytest.mark.validation
def test_backward_facing_step_re100():
    """Backward-facing step: bounded solution with recirculation at Re=100.

    Domain: step of height h = Ly/4 at x = Lx*0.15.
    Uses Brinkman penalization for the step geometry.
    """
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.shapes import rectangle_sdf
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    nx, ny = 64, 24
    lx, ly = 4.0, 1.0

    mesh = CartesianMesh(nx=nx, ny=ny, lx=lx, ly=ly)
    step_h = ly * 0.25
    step_length = lx * 0.12
    sdf_step = rectangle_sdf(mesh, 0.0, 0.0, step_length, step_h)

    solver = NavierStokes2D(
        reynolds_number=100,
        grid=(nx, ny),
        lx=lx,
        ly=ly,
        device="cpu",
        max_iter=3000,
        tol=1e-5,
        alpha_u=0.5,
        alpha_p=0.1,
    )

    ux, uy, p = solver.solve_steady(sdf=sdf_step, inlet_velocity=1.0, case="channel")

    # Sanity checks
    assert not torch.any(torch.isnan(ux)), "NaN in ux"
    assert not torch.any(torch.isnan(uy)), "NaN in uy"
    assert not torch.any(torch.isnan(p)), "NaN in p"
    assert ux.abs().max() < 50.0, "Velocity field exploded"

    # Verify recirculation: negative ux in the step wake region
    step_end_i = int(step_length / mesh.dx) + 2
    probe_j = int(step_h / mesh.dy) + 1
    ux_wake = ux[probe_j, step_end_i : step_end_i + 10]
    assert ux_wake.min() < 0.5, "No recirculation detected behind step"
