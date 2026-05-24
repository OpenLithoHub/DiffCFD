"""Validation test for aerodynamic shape optimization workflow."""

import pytest
import torch


@pytest.mark.slow
@pytest.mark.validation
def test_airfoil_forces_nonzero():
    """compute_forces returns non-zero drag/lift for a NACA 0012 in channel flow."""
    from diffcfd.geometry.airfoil import NACA4Digit, compute_forces
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    nx, ny = 48, 24
    lx, ly = 2.0, 1.0
    mesh = CartesianMesh(nx, ny, lx=lx, ly=ly)

    solver = NavierStokes2D(
        reynolds_number=100, grid=(nx, ny), lx=lx, ly=ly,
        backward="implicit_diff", max_iter=1000, tol=1e-5,
    )

    naca = NACA4Digit(chord=0.4, leading_edge_x=0.5, center_y=0.5, angle_deg=5.0)
    sdf = naca.sdf(mesh)
    ux, uy, p = solver.solve_steady(sdf=sdf, inlet_velocity=1.0, case="channel")

    drag, lift = compute_forces(p, ux, uy, mesh, sdf, mu=1.0 / 100.0)

    assert drag.item() != 0.0, "Drag should be non-zero"
    assert lift.item() != 0.0, "Lift should be non-zero (5° AoA)"
    # Drag should be positive (force in flow direction)
    assert drag.item() > 0, f"Drag should be positive, got {drag.item():.6f}"


def test_naca_sdf_shapes():
    """NACA SDF has correct shape and sign convention."""
    from diffcfd.geometry.airfoil import NACA4Digit
    from diffcfd.geometry.mesh import CartesianMesh

    mesh = CartesianMesh(32, 16, lx=2.0, ly=1.0)
    naca = NACA4Digit(chord=0.3, leading_edge_x=0.5)
    sdf = naca.sdf(mesh)

    assert sdf.shape == (16, 32), f"Expected (16, 32), got {sdf.shape}"
    # Most cells should be in fluid (positive SDF)
    assert (sdf > 0).sum() > sdf.numel() * 0.5, "Majority of domain should be fluid"


def test_bspline_sdf_gradient():
    """B-spline control point gradients flow through SDF computation."""
    from diffcfd.geometry.airfoil import BSplineAirfoil
    from diffcfd.geometry.mesh import CartesianMesh

    mesh = CartesianMesh(16, 16, lx=2.0, ly=1.0)
    bspline = BSplineAirfoil(n_control_points=4, chord=0.3, leading_edge_x=0.5)

    cp = bspline.initial_control_points().clone().detach().requires_grad_(True)
    sdf = bspline.sdf(mesh, cp)
    loss = sdf.sum()
    loss.backward()

    assert cp.grad is not None, "Gradient should flow through SDF"
    assert cp.grad.norm().item() > 0, "Gradient should be non-zero"
