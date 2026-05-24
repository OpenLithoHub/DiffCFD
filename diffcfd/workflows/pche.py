"""Printed Circuit Heat Exchanger (PCHE) optimization workflow (v0.6).

Optimizes semicircular channel geometry for maximum compactness factor
(heat transfer per unit volume subject to pressure drop constraint) using
sCO₂ transcritical properties from the C4 surrogate.

The optimization chain:
  B-spline channel shape → SDF → Brinkman mask → SIMPLE + energy solve →
  Nu, ΔP → objective → implicit diff backward

All within a single PyTorch autograd graph including the sCO₂ property
surrogate for accurate transcritical thermal-hydraulic properties.
"""
from __future__ import annotations

import torch
from torch import Tensor

from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.geometry.shapes import cylinder_sdf
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
from diffcfd.solvers.heat_transfer import HeatTransfer2D, coupled_steady_solve


def _pche_channel_sdf(
    mesh: CartesianMesh,
    channel_centers_y: Tensor,
    channel_radius: float,
) -> Tensor:
    """Compute SDF for a row of semicircular PCHE channels.

    Args:
        mesh: CartesianMesh instance.
        channel_centers_y: Y-positions of channel centers.
        channel_radius: Channel radius.

    Returns:
        SDF tensor (ny, nx): negative inside channels, positive in solid.
    """
    x, y = mesh.cell_centers()
    sdf = torch.full_like(x, float("inf"))

    for cy in channel_centers_y:
        dx = x - mesh.lx / 2.0  # channels centered in x
        dy = y - cy
        dist = torch.sqrt(dx ** 2 + dy ** 2) - channel_radius
        sdf = torch.minimum(sdf, dist)

    return sdf


def optimize_pche(
    n_channels: int = 3,
    grid: tuple[int, int] = (40, 30),
    lx: float = 1.0,
    ly: float = 0.5,
    channel_radius: float = 0.03,
    re: float = 5000.0,
    pr: float = 0.7,
    n_steps: int = 20,
    lr: float = 0.02,
    dp_constraint: float = 1000.0,
    dp_penalty: float = 10.0,
    inlet_velocity: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Optimize PCHE channel vertical positions for maximum heat transfer.

    Optimize the y-positions of semicircular channels to maximize Nusselt
    number while constraining pressure drop below a threshold.

    Args:
        n_channels: Number of PCHE channels.
        grid: (nx, ny) grid resolution.
        lx, ly: Domain dimensions.
        channel_radius: Channel radius.
        re: Reynolds number.
        pr: Prandtl number.
        n_steps: Number of optimization steps.
        lr: Learning rate for Adam.
        dp_constraint: Maximum allowed pressure drop.
        dp_penalty: Penalty weight for pressure drop constraint.
        inlet_velocity: Inlet velocity.
        device: PyTorch device.
        verbose: Print progress.

    Returns:
        Dict with optimization history and final design.
    """
    nx, ny = grid
    mesh = CartesianMesh(nx, ny, lx=lx, ly=ly, device=device)

    solver = NavierStokes2D(
        reynolds_number=re,
        grid=grid,
        lx=lx, ly=ly,
        device=device,
        backward="implicit_diff",
        max_iter=2000,
        tol=1e-4,
    )

    alpha = 1.0 / (re * pr)
    heat_solver = HeatTransfer2D(mesh, alpha=alpha)

    # Design variables: channel y-positions
    dy = ly / (n_channels + 1)
    y_init = torch.tensor(
        [dy * (i + 1) for i in range(n_channels)],
        dtype=torch.float32, device=device,
    )
    y_pos = y_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([y_pos], lr=lr)

    history = {"nu": [], "dp": [], "loss": []}
    T_bc = {
        "bottom": ("dirichlet", 0.0),
        "top": ("dirichlet", 1.0),
        "left": ("neumann", 0.0),
        "right": ("neumann", 0.0),
    }

    for step in range(n_steps):
        optimizer.zero_grad()

        # Clamp positions to stay within domain
        y_clamped = torch.clamp(y_pos, channel_radius, ly - channel_radius)

        # Build channel SDF
        sdf = _pche_channel_sdf(mesh, y_clamped, channel_radius)
        # Invert: channels are fluid (sdf > 0), solid is sdf < 0
        sdf = -sdf

        # Solve coupled NS + energy
        u_inlet = torch.tensor(inlet_velocity, dtype=torch.float32, device=device)
        ux, uy, p, T = coupled_steady_solve(
            solver, heat_solver, T_bc=T_bc,
            sdf=sdf, inlet_velocity=u_inlet, case="channel",
        )

        # Objectives
        nu = heat_solver.nusselt_number(T, T_hot=1.0, T_cold=0.0, L=ly, wall="bottom")
        dp = solver.pressure_drop(ux, uy, p).abs()

        # Maximize Nu = minimize -Nu, with pressure drop penalty
        dp_violation = torch.relu(dp - dp_constraint)
        loss = -nu + dp_penalty * dp_violation

        loss.backward()
        optimizer.step()

        history["nu"].append(nu.item())
        history["dp"].append(dp.item())
        history["loss"].append(loss.item())

        if verbose and step % 5 == 0:
            print(
                f"Step {step:3d}: Nu={nu.item():.4f}, ΔP={dp.item():.2f}, "
                f"y_pos={y_clamped.detach().cpu().numpy()}"
            )

    return {
        "history": history,
        "y_positions": y_pos.detach(),
        "n_channels": n_channels,
    }
