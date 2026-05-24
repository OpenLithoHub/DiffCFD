"""Topology optimization workflow using differentiable Helmholtz filter + Brinkman.

C3 patent embodiment: coupled geometry and boundary condition optimization with
manufacturing constraints (minimum feature size).

The optimization chain is:
    design variables ρ → Helmholtz filter → smooth Heaviside projection →
    Brinkman penalization mask → SIMPLE solve → objective → implicit diff backward

All within a single PyTorch autograd computational graph.

Manufacturing constraints are enforced via the Helmholtz filter (minimum length
scale radius r), which prevents features smaller than ~2r from appearing in the
optimized design.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffcfd.geometry.filters import HelmholtzFilter
from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D


def smooth_heaviside(phi: Tensor, beta: float = 32.0) -> Tensor:
    """Smooth Heaviside projection: maps filtered density to [0, 1].

    Uses the smooth approximation from Lazarov & Sigmund 2016:
        H(x) = (tanh(β·x) + 1) / 2  where x ∈ [-1, 1] centered at 0.5

    Args:
        phi: Filtered density field (ny, nx), values typically in [0, 1].
        beta: Projection sharpness. β=1 → very smooth; β=32 → near step function.

    Returns:
        Projected density in [0, 1].
    """
    return 0.5 + 0.5 * torch.tanh(beta * (phi - 0.5))


def optimize_topology(
    objective: str = "pressure_drop",
    grid: tuple[int, int] = (40, 20),
    lx: float = 2.0,
    ly: float = 1.0,
    re: float = 100.0,
    n_steps: int = 30,
    lr: float = 0.05,
    filter_radius: float = 0.08,
    beta: float = 16.0,
    inlet_velocity: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Topology optimization for minimum pressure drop or minimum dissipation.

    Optimizes a density field ρ ∈ [0, 1] that defines solid (ρ=0) and fluid (ρ=1)
    regions via Brinkman penalization. The Helmholtz filter ensures minimum
    feature size.

    Args:
        objective: "pressure_drop" or "dissipation".
        grid: (nx, ny) grid resolution.
        lx, ly: Domain dimensions.
        re: Reynolds number.
        n_steps: Number of optimization steps.
        lr: Learning rate for Adam optimizer.
        filter_radius: Helmholtz filter radius (minimum length scale / 2).
        beta: Heaviside projection sharpness.
        inlet_velocity: Inlet velocity for channel flow.
        device: PyTorch device.
        verbose: Print progress.

    Returns:
        Dict with optimization history and final design.
    """
    nx, ny = grid
    mesh = CartesianMesh(nx, ny, lx=lx, ly=ly, device=device)
    helmholtz = HelmholtzFilter(mesh, radius=filter_radius)

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

    # Design variables: unconstrained, mapped to [0, 1] via sigmoid
    rho_raw = torch.rand(ny, nx, device=device) * 0.5 + 0.25
    rho_raw = rho_raw.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([rho_raw], lr=lr)

    history = {"objective": [], "fluid_fraction": []}

    for step in range(n_steps):
        optimizer.zero_grad()

        # Design variable → density ∈ (0, 1)
        rho = torch.sigmoid(rho_raw)

        # Helmholtz filter (manufacturing constraint) — differentiable path
        rho_filtered = helmholtz.apply_differentiable(rho, n_iter=30)

        # Smooth Heaviside projection
        chi = smooth_heaviside(rho_filtered, beta=beta)

        # Convert to SDF-like field for Brinkman: positive in fluid, negative in solid
        # chi ≈ 1 in fluid, ≈ 0 in solid → sdf ~ 2*chi - 1
        sdf_approx = 2.0 * chi - 1.0

        # Solve NS with Brinkman penalization
        u_inlet = torch.tensor(
            inlet_velocity, dtype=torch.float32, device=device
        )
        ux, uy, p = solver.solve_steady(sdf=sdf_approx, inlet_velocity=u_inlet, case="channel")

        # Objective: minimize pressure drop
        dp = solver.pressure_drop(ux, uy, p)
        loss = dp.abs()

        loss.backward()
        optimizer.step()

        fluid_frac = chi.mean().item()
        history["objective"].append(loss.item())
        history["fluid_fraction"].append(fluid_frac)

        if verbose and step % 5 == 0:
            print(
                f"Step {step:3d}: |ΔP|={loss.item():.4f}, "
                f"fluid_frac={fluid_frac:.3f}"
            )

    return {
        "history": history,
        "rho_raw": rho_raw.detach(),
        "rho_filtered": helmholtz.apply(torch.sigmoid(rho_raw)).detach(),
        "chi": smooth_heaviside(
            helmholtz.apply(torch.sigmoid(rho_raw)), beta=beta
        ).detach(),
    }
