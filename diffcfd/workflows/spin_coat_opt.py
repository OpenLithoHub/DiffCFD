"""Workflow for spin coating profile optimization.

Optimizes the temporal spin curve ω(t) to achieve a target dry film thickness
and maximize thickness uniformity across the wafer surface.
"""

from __future__ import annotations

import math

import torch
import torch.optim as optim

from diffcfd.solvers.spin_coating import MeyerhoferSolver, RadialThinFilmSolver

RPM_TO_RAD = 2.0 * math.pi / 60.0
RAD_TO_RPM = 60.0 / (2.0 * math.pi)


def optimize_spin_profile(
    target_thickness_nm: float = 100.0,
    r_max: float = 0.1,
    nr: int = 40,
    total_time: float = 30.0,
    dt: float = 0.001,
    n_epochs: int = 40,
    lr: float = 10.0,
) -> dict:
    """Optimize the spin speed trajectory ω(t) to hit target thickness.

    Uses the 1D RadialThinFilmSolver to evaluate radial film profile.
    Gradients flow through the unrolled time integration via PyTorch autograd.
    """
    n_steps = int(total_time / dt)
    target_h = target_thickness_nm * 1e-9

    init_omega = 2000.0 * RPM_TO_RAD
    omega_profile = torch.full((n_steps,), init_omega, requires_grad=True)

    solver = RadialThinFilmSolver(r_max=r_max, nr=nr)

    h0 = torch.full((nr,), 10e-6)
    c0 = torch.full((nr,), 0.82)

    optimizer = optim.Adam([omega_profile], lr=lr)

    history = {"loss": [], "mean_thickness_nm": [], "uniformity": []}

    print(f"Starting Spin Profile Optimization (Target: {target_thickness_nm} nm)...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        omega_clamped = torch.clamp(
            omega_profile,
            min=500.0 * RPM_TO_RAD,
            max=6000.0 * RPM_TO_RAD,
        )

        h_final = solver(omega_clamped, dt, h0, c0)

        loss_target = torch.mean((h_final - target_h) ** 2) / target_h**2
        loss_uniformity = torch.var(h_final) / target_h**2
        loss = loss_target + 0.1 * loss_uniformity

        loss.backward()
        optimizer.step()

        mean_h_nm = h_final.mean().item() * 1e9
        std_h_nm = h_final.std().item() * 1e9
        history["loss"].append(loss.item())
        history["mean_thickness_nm"].append(mean_h_nm)
        history["uniformity"].append(std_h_nm)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(
                f"Epoch {epoch:2d}: Loss={loss.item():.4e} | "
                f"Mean Thickness={mean_h_nm:.2f} nm (Std={std_h_nm:.2f} nm)"
            )

    return {
        "optimized_profile_rpm": (omega_profile.detach() * RAD_TO_RPM).numpy(),
        "final_thickness_nm": (h_final.detach() * 1e9).numpy(),
        "history": history,
    }


if __name__ == "__main__":
    res = optimize_spin_profile(target_thickness_nm=120.0, n_epochs=20)
    print("\nOptimization Finished.")
    print(f"Final Average Thickness: {res['final_thickness_nm'].mean():.2f} nm")
