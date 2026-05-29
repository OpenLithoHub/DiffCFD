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


def total_variation(x: Tensor) -> Tensor:
    """Isotropic total variation for anti-checkerboard regularisation.

    Borrowed from OpenLithoHub's Level-Set ILT (``_total_variation``).
    Penalises per-pixel differences along both axes, suppressing
    checkerboard artefacts that Helmholtz filtering alone may not
    eliminate in topology optimisation.
    """
    diff_h = (x[1:, :] - x[:-1, :]).pow(2)
    diff_w = (x[:, 1:] - x[:, :-1]).pow(2)
    return diff_h.sum() + diff_w.sum()


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
    beta_continuation: bool = True,
    tv_weight: float = 0.0,
    inlet_velocity: float = 1.0,
    volume_fraction: float = 0.5,
    volume_penalty: float = 100.0,
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
        beta: Heaviside projection sharpness (initial if continuation enabled).
        beta_continuation: If True, ramp beta from 1 to target over first half of steps.
        inlet_velocity: Inlet velocity for channel flow.
        volume_fraction: Target fluid fraction (default 0.5).
        volume_penalty: Penalty weight for volume constraint violation.
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
    rho_raw = torch.full((ny, nx), 0.5, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([rho_raw], lr=lr)

    history = {"objective": [], "fluid_fraction": [], "penalty": []}

    for step in range(n_steps):
        optimizer.zero_grad()

        # Beta continuation: ramp from 1 to target beta over first half
        if beta_continuation:
            ramp_end = max(n_steps // 2, 1)
            beta_eff = 1.0 + (beta - 1.0) * min(step / ramp_end, 1.0)
        else:
            beta_eff = beta

        # Design variable → density ∈ (0, 1)
        rho = torch.sigmoid(rho_raw)

        # Helmholtz filter (manufacturing constraint) — differentiable path
        rho_filtered = helmholtz.apply_differentiable(rho, n_iter=30)

        # Smooth Heaviside projection
        chi = smooth_heaviside(rho_filtered, beta=beta_eff)

        # Volume constraint penalty: penalize deviation from target fluid fraction
        vol_error = chi.mean() - volume_fraction
        vol_penalty = volume_penalty * vol_error ** 2

        # Total variation regularisation (borrowed from OpenLithoHub ILT)
        tv_loss = total_variation(chi) * tv_weight if tv_weight > 0 else 0.0

        # Convert to SDF-like field for Brinkman: positive in fluid, negative in solid
        sdf_approx = 2.0 * chi - 1.0

        # Solve NS with Brinkman penalization
        u_inlet = torch.tensor(
            inlet_velocity, dtype=torch.float32, device=device
        )
        ux, uy, p = solver.solve_steady(sdf=sdf_approx, inlet_velocity=u_inlet, case="channel")

        # Objective: minimize pressure drop
        dp = solver.pressure_drop(ux, uy, p)
        loss = dp.abs() + vol_penalty + tv_loss

        loss.backward()
        optimizer.step()

        fluid_frac = chi.mean().item()
        history["objective"].append(dp.abs().item())
        history["fluid_fraction"].append(fluid_frac)
        history["penalty"].append(vol_penalty.item())

        if verbose and step % 5 == 0:
            print(
                f"Step {step:3d}: |ΔP|={dp.abs().item():.4f}, "
                f"fluid_frac={fluid_frac:.3f}, β={beta_eff:.1f}, "
                f"penalty={vol_penalty.item():.4f}"
            )

    return {
        "history": history,
        "rho_raw": rho_raw.detach(),
        "rho_filtered": helmholtz.apply(torch.sigmoid(rho_raw)).detach(),
        "chi": smooth_heaviside(
            helmholtz.apply(torch.sigmoid(rho_raw)), beta=beta
        ).detach(),
    }
