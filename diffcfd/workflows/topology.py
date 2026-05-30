"""Topology optimization workflow using differentiable Helmholtz filter + Brinkman.

Coupled geometry and boundary condition optimization with
manufacturing constraints (minimum feature size).

The optimization chain is:
    design variables rho -> Helmholtz filter -> smooth Heaviside projection ->
    Brinkman penalization mask -> SIMPLE solve -> objective -> implicit diff backward

All within a single PyTorch autograd computational graph.

Manufacturing constraints are enforced via the Helmholtz filter (minimum length
scale radius r), which prevents features smaller than ~2r from appearing in the
optimized design.

Multi-corner robust optimization (B.3) borrows from OpenLithoHub's
``pw_fidelity_loss()`` — evaluates the objective at multiple operating points
(e.g. different Reynolds numbers) and combines the weighted losses to produce
a design that performs well across the entire operating envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from diff_surrogate.convergence import ConvergenceAction

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
    convergence_monitor: object | None = None,
) -> dict:
    """Topology optimization for minimum pressure drop or minimum dissipation.

    Optimizes a density field rho in [0, 1] that defines solid (rho=0) and fluid (rho=1)
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
        convergence_monitor: Optional ConvergenceMonitor instance. When provided,
            the monitor checks convergence after each step and may trigger
            early stopping or learning-rate reduction.

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

    # Pre-load convergence action enum if monitor is provided
    _ConvergenceAction = None
    if convergence_monitor is not None:
        _ConvergenceAction = ConvergenceAction

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
        vol_penalty = volume_penalty * vol_error**2

        # Total variation regularisation (borrowed from OpenLithoHub ILT)
        tv_loss = total_variation(chi) * tv_weight if tv_weight > 0 else 0.0

        # Convert to SDF-like field for Brinkman: positive in fluid, negative in solid
        sdf_approx = 2.0 * chi - 1.0

        # Solve NS with Brinkman penalization
        u_inlet = torch.tensor(inlet_velocity, dtype=torch.float32, device=device)
        ux, uy, p = solver.solve_steady(
            sdf=sdf_approx, inlet_velocity=u_inlet, case="channel"
        )

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
                f"Step {step:3d}: |dp|={dp.abs().item():.4f}, "
                f"fluid_frac={fluid_frac:.3f}, beta={beta_eff:.1f}, "
                f"penalty={vol_penalty.item():.4f}"
            )

            # Convergence monitoring (B.1)
        if convergence_monitor is not None:
            action = convergence_monitor.update(dp.abs().item(), step)
            if action == _ConvergenceAction.EARLY_STOP:
                if verbose:
                    print(f"  >> Convergence: early stop at step {step}")
                break
            elif action == _ConvergenceAction.REDUCE_LR:
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                if verbose:
                    print(
                        f"  >> Convergence: reducing LR to {optimizer.param_groups[0]['lr']:.6f}"
                    )

    return {
        "history": history,
        "rho_raw": rho_raw.detach(),
        "rho_filtered": helmholtz.apply(torch.sigmoid(rho_raw)).detach(),
        "chi": smooth_heaviside(
            helmholtz.apply(torch.sigmoid(rho_raw)), beta=beta
        ).detach(),
    }


# ---------------------------------------------------------------------------
# Multi-corner robust topology optimization (B.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CornerSpec:
    """One operating-point corner in the multi-corner optimization sweep.

    Mirrors OpenLithoHub's ``ProcessWindowCorner`` — each corner specifies
    a distinct operating condition (e.g. a different Reynolds number) and
    a weight for combining losses.

    Args:
        reynolds: Reynolds number for this corner.
        weight: Relative weight in the combined loss. Higher weight means
            this corner's performance matters more.
        label: Human-readable label for logging.
    """

    reynolds: float
    weight: float = 1.0
    label: str = ""


DEFAULT_CORNER_SPECS: tuple[CornerSpec, ...] = (
    CornerSpec(reynolds=80.0, weight=1.0, label="under"),
    CornerSpec(reynolds=100.0, weight=2.0, label="nominal"),
    CornerSpec(reynolds=120.0, weight=1.0, label="over"),
)


def _solve_corner_loss(
    solver: NavierStokes2D,
    sdf_approx: Tensor,
    inlet_velocity: float,
    objective: str,
) -> Tensor:
    """Run forward simulation and compute objective for one corner.

    Args:
        solver: NavierStokes2D solver configured for this corner's Re.
        sdf_approx: SDF-like field (positive=fluid, negative=solid).
        inlet_velocity: Inlet velocity magnitude.
        objective: "pressure_drop" or "dissipation".

    Returns:
        Scalar loss tensor with autograd connection to sdf_approx.
    """
    u_inlet = torch.tensor(
        inlet_velocity, dtype=torch.float32, device=sdf_approx.device
    )
    ux, uy, p = solver.solve_steady(
        sdf=sdf_approx, inlet_velocity=u_inlet, case="channel"
    )

    if objective == "pressure_drop":
        return solver.pressure_drop(ux, uy, p).abs()
    else:
        return solver.dissipation(ux, uy, p)


def multi_corner_optimize(
    objective: str = "pressure_drop",
    grid: tuple[int, int] = (40, 20),
    lx: float = 2.0,
    ly: float = 1.0,
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
    corners: Sequence[CornerSpec] = DEFAULT_CORNER_SPECS,
    convergence_monitor: object | None = None,
) -> dict:
    """Multi-corner robust topology optimization.

    Borrowed from OpenLithoHub's ``pw_fidelity_loss()`` pattern: instead of
    optimizing for a single operating point, evaluates the objective at multiple
    Reynolds numbers (corners) and combines the weighted losses:

        loss = Σ_i  w_i * L_i  /  Σ_i  w_i

    This ensures the resulting topology performs well across the operating
    envelope rather than being fragile at off-nominal conditions.

    All corners share the same design variables (ρ_raw), Helmholtz filter, and
    Heaviside projection — only the solver's Reynolds number changes per corner.
    The combined loss is fully differentiable w.r.t. ρ_raw via autograd.

    Args:
        objective: "pressure_drop" or "dissipation".
        grid: (nx, ny) grid resolution.
        lx, ly: Domain dimensions.
        n_steps: Number of optimization steps.
        lr: Learning rate for Adam optimizer.
        filter_radius: Helmholtz filter radius (minimum length scale / 2).
        beta: Heaviside projection sharpness (initial if continuation enabled).
        beta_continuation: If True, ramp beta from 1 to target over first half.
        tv_weight: Total-variation regularisation weight.
        inlet_velocity: Inlet velocity for channel flow.
        volume_fraction: Target fluid fraction (default 0.5).
        volume_penalty: Penalty weight for volume constraint violation.
        device: PyTorch device.
        verbose: Print progress.
        corners: Sequence of CornerSpec defining operating points and weights.
            Defaults to Re=80/100/120 with weights 1/2/1.
        convergence_monitor: Optional ConvergenceMonitor instance. When provided,
            the monitor checks convergence after each step and may trigger
            early stopping or learning-rate reduction.

    Returns:
        Dict with optimization history (per-corner and combined), final design,
        and convergence metadata.
    """
    if len(corners) == 0:
        raise ValueError("multi_corner_optimize requires at least one corner")
    weight_sum = sum(c.weight for c in corners)
    if weight_sum <= 0.0:
        raise ValueError("Corner weights must sum to a positive value")

    nx, ny = grid
    mesh = CartesianMesh(nx, ny, lx=lx, ly=ly, device=device)
    helmholtz = HelmholtzFilter(mesh, radius=filter_radius)

    # Build one solver per corner (each has different Re)
    solvers = []
    for corner in corners:
        solver = NavierStokes2D(
            reynolds_number=corner.reynolds,
            grid=grid,
            lx=lx,
            ly=ly,
            device=device,
            backward="implicit_diff",
            max_iter=2000,
            tol=1e-5,
        )
        solvers.append(solver)

    # Design variables: unconstrained, mapped to [0, 1] via sigmoid
    rho_raw = torch.full((ny, nx), 0.5, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([rho_raw], lr=lr)

    _ConvergenceAction = None
    if convergence_monitor is not None:
        _ConvergenceAction = ConvergenceAction

    history: dict = {
        "combined": [],
        "fluid_fraction": [],
        "penalty": [],
    }
    for corner in corners:
        label = corner.label or f"Re{corner.reynolds:.0f}"
        history[label] = []

    convergence_info: dict = {}

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

        # Helmholtz filter (manufacturing constraint)
        rho_filtered = helmholtz.apply_differentiable(rho, n_iter=30)

        # Smooth Heaviside projection
        chi = smooth_heaviside(rho_filtered, beta=beta_eff)

        # Volume constraint penalty
        vol_error = chi.mean() - volume_fraction
        vol_penalty = volume_penalty * vol_error**2

        # Total variation regularisation
        tv_loss = total_variation(chi) * tv_weight if tv_weight > 0 else 0.0

        # SDF-like field for Brinkman
        sdf_approx = 2.0 * chi - 1.0

        # Evaluate loss at each corner (weighted combination)
        combined_loss = rho.new_zeros(())
        for i, corner in enumerate(corners):
            corner_loss = _solve_corner_loss(
                solvers[i],
                sdf_approx,
                inlet_velocity,
                objective,
            )
            combined_loss = combined_loss + corner.weight * corner_loss
            label = corner.label or f"Re{corner.reynolds:.0f}"
            history[label].append(corner_loss.item())

        combined_loss = combined_loss / weight_sum

        loss = combined_loss + vol_penalty + tv_loss

        loss.backward()
        optimizer.step()

        fluid_frac = chi.mean().item()
        history["combined"].append(combined_loss.item())
        history["fluid_fraction"].append(fluid_frac)
        history["penalty"].append(vol_penalty.item())

        if verbose and step % 5 == 0:
            corner_strs = []
            for corner in corners:
                label = corner.label or f"Re{corner.reynolds:.0f}"
                corner_strs.append(f"{label}={history[label][-1]:.4f}")
            print(
                f"Step {step:3d}: combined={combined_loss.item():.4f}, "
                f"{', '.join(corner_strs)}, "
                f"fluid_frac={fluid_frac:.3f}, β={beta_eff:.1f}"
            )

        # Convergence monitoring
        if convergence_monitor is not None:
            action = convergence_monitor.update(combined_loss.item(), step)
            if action == _ConvergenceAction.EARLY_STOP:
                convergence_info = {
                    "early_stopped": True,
                    "step": step,
                    "message": f"Early stop at step {step}",
                }
                if verbose:
                    print(f"  >> Convergence: early stop at step {step}")
                break
            elif action == _ConvergenceAction.REDUCE_LR:
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                if verbose:
                    print(
                        f"  >> Convergence: reducing LR to {optimizer.param_groups[0]['lr']:.6f}"
                    )

    result = {
        "history": history,
        "rho_raw": rho_raw.detach(),
        "rho_filtered": helmholtz.apply(torch.sigmoid(rho_raw)).detach(),
        "chi": smooth_heaviside(
            helmholtz.apply(torch.sigmoid(rho_raw)), beta=beta
        ).detach(),
        "corners": [(c.reynolds, c.weight, c.label) for c in corners],
    }
    if convergence_info:
        result["convergence"] = convergence_info
    return result
