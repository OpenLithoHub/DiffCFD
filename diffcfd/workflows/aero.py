"""Aerodynamic shape optimization workflow.

Uses DiffCFD's differentiable SIMPLE + implicit diff to optimize airfoil
shapes for minimum drag / maximum lift via gradient descent through the
autograd graph.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffcfd.geometry.airfoil import BSplineAirfoil, NACA4Digit, compute_forces
from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D


def optimize_airfoil(
    re: float = 1000.0,
    grid: tuple[int, int] = (64, 32),
    n_steps: int = 20,
    lr: float = 1e-2,
    chord: float = 0.5,
    le_x: float = 0.5,
    cy: float = 0.5,
    angle_deg: float = 5.0,
    lx: float = 2.0,
    ly: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Optimize airfoil shape to minimize drag.

    Uses B-spline control points as design variables. Adam optimizer with
    implicit differentiation through SIMPLE for O(N) memory backward.

    Args:
        re: Reynolds number.
        grid: Grid resolution.
        n_steps: Optimization steps.
        lr: Learning rate.
        chord: Airfoil chord length.
        le_x: Leading edge x-position.
        cy: Chord line y-position.
        angle_deg: Angle of attack.
        lx, ly: Domain dimensions.
        device: PyTorch device.

    Returns:
        Dict with optimization history (drag, lift per step).
    """
    nx, ny = grid
    mesh = CartesianMesh(nx, ny, lx=lx, ly=ly, device=device)

    solver = NavierStokes2D(
        reynolds_number=re,
        grid=grid,
        lx=lx,
        ly=ly,
        device=device,
        backward="implicit_diff",
        max_iter=2000,
        tol=1e-5,
    )

    bspline = BSplineAirfoil(
        n_control_points=8, chord=chord, leading_edge_x=le_x, center_y=cy
    )
    control_points = bspline.initial_control_points().clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([control_points], lr=lr)

    history = {"drag": [], "lift": []}

    for step in range(n_steps):
        optimizer.zero_grad()

        sdf = bspline.sdf(mesh, control_points)
        ux, uy, p = solver.solve_steady(sdf=sdf, inlet_velocity=1.0, case="channel")

        drag, lift = compute_forces(p, ux, uy, mesh, sdf, mu=1.0 / re)
        loss = drag  # Minimize drag

        loss.backward()
        optimizer.step()

        history["drag"].append(drag.item())
        history["lift"].append(lift.item())

        if step % 5 == 0:
            print(f"Step {step}: drag={drag.item():.4f}, lift={lift.item():.4f}")

    return history
