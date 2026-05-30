"""Joint workflow for Spin Coating and Lithography Co-Optimization.

Couples:
  Spin trajectory omega(t) -> Film Thickness (h) & Solvent (C) ->
  Dill Exposure & Mack Development -> Developed Feature Thickness
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.optim as optim

from diffcfd.solvers.litho import LithoSolver
from diffcfd.solvers.spin_coating import MeyerhoferSolver

RPM_TO_RAD = 2.0 * math.pi / 60.0
RAD_TO_RPM = 60.0 / (2.0 * math.pi)


def _forward_chain(
    spin_solver: MeyerhoferSolver,
    litho_solver: LithoSolver,
    omega_profile: torch.Tensor,
    exposure_dose: torch.Tensor,
    spin_dt: float,
    h0: float,
    c0: float,
    dev_time: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the spin-coating + lithography forward chain.

    Returns (loss, h_final, h_dry, c_dry, dose_clamped).
    Target is always fixed at the module level; this helper avoids
    duplication between joint and decoupled paths.
    """
    omega_clamped = torch.clamp(
        omega_profile, min=1000.0 * RPM_TO_RAD, max=5000.0 * RPM_TO_RAD
    )
    dose_clamped = torch.clamp(exposure_dose, min=10.0, max=300.0)

    h_hist, c_hist = spin_solver(omega_clamped, spin_dt, h0, c0)
    h_dry = h_hist[-1]
    c_dry = c_hist[-1]

    h_final = litho_solver(
        thickness=h_dry,
        residual_solvent=c_dry,
        exposure_dose=dose_clamped,
        dev_time=dev_time,
    )
    return h_final, h_dry, c_dry, dose_clamped, omega_clamped


def optimize_joint_process(
    target_developed_h_nm: float = 50.0,
    total_spin_time: float = 10.0,
    spin_dt: float = 0.001,
    n_epochs: int = 50,
    lr: float = 5.0,
    lr_dose: float | None = None,
    verbose: bool = True,
    h0: float = 8e-6,
    c0: float = 0.85,
    init_omega_rpm: float = 2500.0,
) -> dict:
    """Jointly optimize spin-coating speed profile AND litho exposure dose.  :stable:

    Uses separate learning rates for the spin profile and dose via Adam
    parameter groups.  If *lr_dose* is None, both groups share *lr*.
    """
    n_spin_steps = int(total_spin_time / spin_dt)

    init_omega = init_omega_rpm * RPM_TO_RAD
    omega_profile = torch.full((n_spin_steps,), init_omega, requires_grad=True)
    exposure_dose = torch.tensor(80.0, requires_grad=True)

    spin_solver = MeyerhoferSolver()
    litho_solver = LithoSolver()

    if lr_dose is not None:
        optimizer = optim.Adam([
            {"params": [omega_profile], "lr": lr},
            {"params": [exposure_dose], "lr": lr_dose},
        ])
    else:
        optimizer = optim.Adam([omega_profile, exposure_dose], lr=lr)

    target_m = target_developed_h_nm * 1e-9

    loss_history: list[float] = []

    if verbose:
        print("Executing Joint Spin-Lithography Optimization Pipeline...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        h_final, h_dry, c_dry, dose_clamped, _ = _forward_chain(
            spin_solver, litho_solver, omega_profile, exposure_dose,
            spin_dt, h0, c0,
        )

        loss = ((h_final - target_m) ** 2) / (target_m**2)

        if not torch.isfinite(loss):
            warnings.warn(
                f"Epoch {epoch}: non-finite loss ({loss.item()}), skipping backward/step.",
                stacklevel=2,
            )
            loss_history.append(float("nan"))
            continue

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(
                f"Epoch {epoch:2d} | Loss: {loss_val:.4e} | "
                f"Spin Dry Thick: {h_dry.item() * 1e9:.2f} nm | "
                f"Residual Solvent: {c_dry.item() * 100:.2f}% | "
                f"Dose: {dose_clamped.item():.2f} mJ/cm2 | "
                f"Developed Thick: {h_final.item() * 1e9:.2f} nm"
            )

    return {
        "opt_spin_rpm": (omega_profile.detach() * RAD_TO_RPM).numpy(),
        "opt_dose_mj": exposure_dose.detach().clamp(10.0, 300.0).item(),
        "final_developed_nm": h_final.detach().item() * 1e9,
        "loss_history": loss_history,
        "omega_profile": omega_profile.detach().clone(),
        "dose_tensor": exposure_dose.detach().clone(),
    }


def optimize_decoupled_process(
    target_developed_h_nm: float = 50.0,
    total_spin_time: float = 10.0,
    spin_dt: float = 0.001,
    n_spin_epochs: int = 25,
    n_dose_epochs: int = 25,
    lr_spin: float = 10.0,
    lr_dose: float = 2.0,
    verbose: bool = True,
    h0: float = 8e-6,
    c0: float = 0.85,
    init_omega_rpm: float = 2500.0,
) -> dict:
    """Decoupled baseline: optimize spin profile for thickness, then sweep dose.  :stable:

    Phase 1: Optimize omega(t) to reach a target dry thickness that is
    heuristically estimated to leave the correct amount of resist after
    development.

    Phase 2: Freeze the spin profile and optimize only the exposure dose.
    """
    n_spin_steps = int(total_spin_time / spin_dt)
    spin_solver = MeyerhoferSolver()
    litho_solver = LithoSolver()

    target_m = target_developed_h_nm * 1e-9

    # Heuristic: target dry thickness ≈ target_developed * 5x to account for
    # development dissolution.  The exact ratio depends on dose but this gives
    # a reasonable starting point.
    target_dry_nm = target_developed_h_nm * 5.0
    target_dry_m = target_dry_nm * 1e-9

    # --- Phase 1: Optimize spin profile for dry thickness ---
    init_omega = init_omega_rpm * RPM_TO_RAD
    omega_profile = torch.full((n_spin_steps,), init_omega, requires_grad=True)
    opt_spin = optim.Adam([omega_profile], lr=lr_spin)

    loss_history: list[float] = []

    if verbose:
        print("Decoupled Phase 1: Spin Profile Optimization...")

    for epoch in range(n_spin_epochs):
        opt_spin.zero_grad()
        omega_clamped = torch.clamp(
            omega_profile, min=1000.0 * RPM_TO_RAD, max=5000.0 * RPM_TO_RAD
        )
        h_hist, c_hist = spin_solver(omega_clamped, spin_dt, h0, c0)
        h_dry = h_hist[-1]

        spin_loss = ((h_dry - target_dry_m) ** 2) / (target_dry_m**2)

        if not torch.isfinite(spin_loss):
            warnings.warn(
                f"Spin epoch {epoch}: non-finite loss, skipping backward/step.",
                stacklevel=2,
            )
            continue

        spin_loss.backward()
        opt_spin.step()

        if verbose and (epoch % 5 == 0 or epoch == n_spin_epochs - 1):
            print(
                f"  Spin Epoch {epoch:2d} | Dry Thick: {h_dry.item() * 1e9:.2f} nm "
                f"(target {target_dry_nm:.1f} nm)"
            )

    # --- Phase 2: Freeze spin, optimize dose ---
    if verbose:
        print("Decoupled Phase 2: Dose Optimization (spin frozen)...")

    exposure_dose = torch.tensor(80.0, requires_grad=True)
    opt_dose = optim.Adam([exposure_dose], lr=lr_dose)

    omega_frozen = omega_profile.detach().clone()

    for epoch in range(n_dose_epochs):
        opt_dose.zero_grad()
        h_final, _, _, dose_clamped, _ = _forward_chain(
            spin_solver, litho_solver, omega_frozen, exposure_dose,
            spin_dt, h0, c0,
        )
        dose_loss = ((h_final - target_m) ** 2) / (target_m**2)

        if not torch.isfinite(dose_loss):
            warnings.warn(
                f"Dose epoch {epoch}: non-finite loss, skipping backward/step.",
                stacklevel=2,
            )
            loss_history.append(float("nan"))
            continue

        dose_loss.backward()
        opt_dose.step()

        loss_history.append(dose_loss.item())

        if verbose and (epoch % 5 == 0 or epoch == n_dose_epochs - 1):
            print(
                f"  Dose Epoch {epoch:2d} | Dose: {dose_clamped.item():.2f} mJ/cm2 | "
                f"Developed: {h_final.item() * 1e9:.2f} nm | "
                f"Loss: {dose_loss.item():.4e}"
            )

    # Final evaluation
    with torch.no_grad():
        h_final, h_dry, c_dry, dose_clamped, _ = _forward_chain(
            spin_solver, litho_solver, omega_frozen, exposure_dose,
            spin_dt, h0, c0,
        )

    return {
        "opt_spin_rpm": (omega_frozen * RAD_TO_RPM).numpy(),
        "opt_dose_mj": exposure_dose.detach().clamp(10.0, 300.0).item(),
        "final_developed_nm": h_final.item() * 1e9,
        "loss_history": loss_history,
        "omega_profile": omega_frozen,
        "dose_tensor": exposure_dose.detach().clone(),
    }


def process_window_analysis(
    omega_profile: torch.Tensor,
    nominal_dose: torch.Tensor,
    spin_dt: float,
    h0: float = 8e-6,
    c0: float = 0.85,
    n_sweep: int = 21,
    dose_range_frac: float = 0.10,
    dev_time: float = 30.0,
    tolerance_frac: float = 0.02,
) -> dict:
    """Sweep dose around nominal and evaluate process window.  :stable:

    Target and tolerance are **self-derived** from the nominal-dose forward
    pass rather than hardcoded.  The nominal developed thickness becomes the
    target and tolerance is a relative fraction of that value.  This ensures
    the process window metric is meaningful regardless of the absolute scale
    of the lithography output.

    Args:
        omega_profile: Spin speed profile tensor (rad/s).
        nominal_dose: Nominal exposure dose tensor (mJ/cm2).
        spin_dt: Time step used for the spin solver.
        h0: Initial wet film thickness (m).
        c0: Initial solvent fraction.
        n_sweep: Number of dose sweep points.
        dose_range_frac: Fraction of nominal dose for the sweep range (±).
        dev_time: Development time (s).
        tolerance_frac: Relative tolerance as fraction of nominal developed
            thickness (e.g. 0.02 = ±2%).

    Returns:
        Dictionary with sweep results and process window bounds.
    """
    spin_solver = MeyerhoferSolver()
    litho_solver = LithoSolver()

    dose_val = nominal_dose.item()
    doses = torch.linspace(
        dose_val * (1.0 - dose_range_frac),
        dose_val * (1.0 + dose_range_frac),
        n_sweep,
    )

    with torch.no_grad():
        h_hist, c_hist = spin_solver(omega_profile, spin_dt, h0, c0)
        h_dry = h_hist[-1]
        c_dry = c_hist[-1]

        h_nom = litho_solver(h_dry, c_dry, nominal_dose, dev_time=dev_time)
        dev_nm_nominal = h_nom.item() * 1e9

    if not math.isfinite(dev_nm_nominal):
        return {
            "nominal_dose_mj": dose_val,
            "nominal_developed_nm": float("nan"),
            "sweep_results": [],
            "process_window_low_mj": dose_val,
            "process_window_high_mj": dose_val,
            "process_window_width_mj": 0.0,
            "target_nm": float("nan"),
            "tolerance_nm": float("nan"),
            "tolerance_frac": tolerance_frac,
        }

    target_nm = dev_nm_nominal
    tolerance_nm = abs(dev_nm_nominal) * tolerance_frac

    results: list[dict] = []
    acceptable_doses: list[float] = []

    for d in doses:
        with torch.no_grad():
            h_dev = litho_solver(h_dry, c_dry, d.unsqueeze(0), dev_time=dev_time)
        dev_nm = h_dev.item() * 1e9
        if not math.isfinite(dev_nm):
            results.append({
                "dose_mj": d.item(),
                "developed_nm": float("nan"),
                "error_nm": float("nan"),
                "acceptable": False,
            })
            continue
        err_nm = abs(dev_nm - target_nm)
        acceptable = err_nm <= tolerance_nm
        results.append({
            "dose_mj": d.item(),
            "developed_nm": dev_nm,
            "error_nm": err_nm,
            "acceptable": acceptable,
        })
        if acceptable:
            acceptable_doses.append(d.item())

    window_low = min(acceptable_doses) if acceptable_doses else dose_val
    window_high = max(acceptable_doses) if acceptable_doses else dose_val
    window_width_mj = window_high - window_low

    return {
        "nominal_dose_mj": dose_val,
        "nominal_developed_nm": dev_nm_nominal,
        "sweep_results": results,
        "process_window_low_mj": window_low,
        "process_window_high_mj": window_high,
        "process_window_width_mj": window_width_mj,
        "target_nm": target_nm,
        "tolerance_nm": tolerance_nm,
        "tolerance_frac": tolerance_frac,
    }


if __name__ == "__main__":
    optimize_joint_process(target_developed_h_nm=60.0, n_epochs=30)
