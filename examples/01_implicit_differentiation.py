"""Example: implicit differentiation through Poiseuille channel flow.

Demonstrates the C1 patent claim:
    solve SIMPLE to steady state → compute pressure drop →
    backward via matrix-free GMRES on unrelaxed physics residual →
    exact analytical gradient dΔP/dU_inlet

Compares against the analytical solution: dΔP/dU = 12μL/h² for
fully-developed Poiseuille flow.
"""

import torch
from diffcfd import NavierStokes2D

# Poiseuille channel: Re=1, Lx=4, Ly=1, grid (32, 16)
# Analytical pressure drop: ΔP = 12 * ν * U * Lx / h²
# With ν=1/Re=1, h=1, Lx=4: ΔP = 48 * U
# So dΔP/dU = 48

solver = NavierStokes2D(
    reynolds_number=1.0,
    grid=(32, 16),
    lx=4.0,
    ly=1.0,
    backward="implicit_diff",
    max_iter=2000,
    tol=1e-5,
)

u_inlet = torch.tensor(1.0, requires_grad=True)
ux, uy, p = solver.solve_steady(inlet_velocity=u_inlet, case="channel")

dp = solver.pressure_drop(ux, uy, p)
dp.backward()

analytical_grad = 48.0  # dΔP/dU for Poiseuille flow
computed_grad = u_inlet.grad.item()

print(f"Pressure drop:  {dp.item():.4f}")
print(f"Analytical:     dΔP/dU = {analytical_grad}")
print(f"Computed:       dΔP/dU = {computed_grad:.4f}")
print(
    f"Relative error: {abs(computed_grad - analytical_grad) / analytical_grad * 100:.4f}%"
)
