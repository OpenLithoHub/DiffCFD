"""Joint workflow for Spin Coating and Lithography Co-Optimization.

Couples:
  Spin trajectory omega(t) -> Film Thickness (h) & Solvent (C) ->
  Dill Exposure & Mack Development -> Developed Feature Thickness
"""

from __future__ import annotations

import math

import torch
import torch.optim as optim

from diffcfd.solvers.litho import LithoSolver
from diffcfd.solvers.spin_coating import MeyerhoferSolver

RPM_TO_RAD = 2.0 * math.pi / 60.0
RAD_TO_RPM = 60.0 / (2.0 * math.pi)


def optimize_joint_process(
    target_developed_h_nm: float = 50.0,
    total_spin_time: float = 10.0,
    spin_dt: float = 0.05,
    n_epochs: int = 50,
) -> dict:
    """Jointly optimize spin-coating speed profile AND litho exposure dose."""
    n_spin_steps = int(total_spin_time / spin_dt)

    init_omega = 2500.0 * RPM_TO_RAD
    omega_profile = torch.full((n_spin_steps,), init_omega, requires_grad=True)

    exposure_dose = torch.tensor(80.0, requires_grad=True)

    spin_solver = MeyerhoferSolver()
    litho_solver = LithoSolver()

    optimizer = optim.Adam([omega_profile, exposure_dose], lr=5.0)

    h0 = 8e-6
    c0 = 0.85

    print("Executing Joint Spin-Lithography Optimization Pipeline...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()

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
            dev_time=30.0,
        )

        target_m = target_developed_h_nm * 1e-9
        loss = ((h_final - target_m) ** 2) / (target_m**2)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(
                f"Epoch {epoch:2d} | Loss: {loss.item():.4e} | "
                f"Spin Dry Thick: {h_dry.item() * 1e9:.2f} nm | "
                f"Residual Solvent: {c_dry.item() * 100:.2f}% | "
                f"Dose: {dose_clamped.item():.2f} mJ/cm2 | "
                f"Developed Thick: {h_final.item() * 1e9:.2f} nm"
            )

    return {
        "opt_spin_rpm": (omega_profile.detach() * RAD_TO_RPM).numpy(),
        "opt_dose_mj": dose_clamped.detach().item(),
        "final_developed_nm": h_final.detach().item() * 1e9,
    }


if __name__ == "__main__":
    optimize_joint_process(target_developed_h_nm=60.0, n_epochs=30)
