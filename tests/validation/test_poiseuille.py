"""Validation: Poiseuille flow analytical gradient verification.

Acceptance gate (v0.1 before CN filing):
- ∂ΔP/∂U_inlet must match analytical 12·μ·L/h² to < 0.01%
  This is the PRIMARY proof of gradient exactness for the full implicit-diff solver.
  (torch.autograd.gradcheck and complex-step cover sub-components only.)
"""

import pytest
import torch


@pytest.mark.skip(reason="NavierStokes2D not yet implemented (v0.05 target)")
@pytest.mark.slow
@pytest.mark.validation
def test_poiseuille_forward(poiseuille_params):
    """Fully-developed Poiseuille: u(y) matches 4·U_max·y·(1-y) parabolic profile."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    p = poiseuille_params
    solver = NavierStokes2D(reynolds_number=p["reynolds_number"], grid=p["grid"])
    u, _ = solver.solve_steady(sdf=None, inlet_velocity=p["u_inlet"])

    ny = p["grid"][1]
    y = torch.linspace(p["ly"] / (2 * ny), p["ly"] - p["ly"] / (2 * ny), ny)
    u_analytical = 4 * p["u_inlet"] * y * (p["ly"] - y) / p["ly"] ** 2

    u_outlet = u[0, :, -1]
    l2_err = torch.norm(u_outlet - u_analytical) / torch.norm(u_analytical)
    assert l2_err.item() < 0.01, f"Poiseuille forward L2 error {l2_err:.4f}"


@pytest.mark.skip(reason="implicit_diff not yet implemented (v0.1 target)")
@pytest.mark.slow
@pytest.mark.validation
def test_poiseuille_gradient(poiseuille_params):
    """Gradient verification (C1 patent claim): ∂ΔP/∂U_inlet < 0.01% vs analytical.

    Analytical: for mean velocity U_mean = (2/3)·U_max in parabolic profile,
    ΔP = 12·μ·U_mean·L / h²
    ∂ΔP/∂U_mean = 12·μ·L / h²
    """
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    p = poiseuille_params
    mu = p["mu"]
    L = p["lx"]
    h = p["ly"]
    analytical_grad = 12 * mu * L / h ** 2

    u_inlet = torch.tensor(p["u_inlet"], dtype=torch.float64, requires_grad=True)
    solver = NavierStokes2D(
        reynolds_number=p["reynolds_number"],
        grid=p["grid"],
        backward="implicit_diff",
    )
    u, pf = solver.solve_steady(sdf=None, inlet_velocity=u_inlet)
    dp = solver.pressure_drop(u, pf)
    dp.backward()

    computed_grad = u_inlet.grad.item()
    rel_err = abs(computed_grad - analytical_grad) / abs(analytical_grad)
    assert rel_err < 1e-4, (
        f"∂ΔP/∂U_inlet: computed={computed_grad:.6f}, "
        f"analytical={analytical_grad:.6f}, rel_err={rel_err:.2e}"
    )
