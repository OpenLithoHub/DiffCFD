"""Validation: Poiseuille flow analytical gradient verification.

Acceptance gate (v0.1 before CN filing):
- forward: ux outlet profile matches parabolic to L2 < 1%
- gradient: implicit-diff ∂ΔP/∂U_inlet must match FD reference to < 0.01%

The C1 patent claim: matrix-free GMRES implicit differentiation through SIMPLE
gives EXACT gradients. Verified against FD with eps=0.01 (in the linear regime
for Stokes flow at Re=1, larger eps avoids floating-point noise).
"""

import pytest
import torch


@pytest.mark.slow
@pytest.mark.validation
def test_poiseuille_forward(poiseuille_params):
    """Fully-developed Poiseuille: u(y) at outlet matches parabolic profile.

    The MAC grid places ux faces at y = j*dy; walls are at j=0 and j=ny-1,
    so the effective channel height is h_eff = (ny-1)*dy < ly.
    Bulk mean velocity is computed from the outlet mass flux over h_eff.
    """
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    p = poiseuille_params
    solver = NavierStokes2D(
        reynolds_number=p["reynolds_number"],
        grid=p["grid"],
        lx=p["lx"],
        ly=p["ly"],
    )
    ux, uy, pf = solver.solve_steady(sdf=None, inlet_velocity=p["u_inlet"], case="channel")

    ny = p["grid"][1]
    ly = p["ly"]
    dy = ly / ny
    h_eff = (ny - 1) * dy   # wall-to-wall distance on the MAC grid
    y = torch.arange(ny, dtype=torch.float32) * dy

    u_outlet = ux[:, -1]
    Q = (u_outlet[1:-1].sum() * dy).item()   # mass flow rate per unit width
    U_mean = Q / h_eff                        # bulk mean velocity

    # Fully-developed parabolic profile: u(y) = 6*U_mean*(y/h_eff)*(1-y/h_eff)
    u_analytical = 6.0 * U_mean * (y / h_eff) * (1.0 - y / h_eff)

    l2_err = torch.norm(u_outlet - u_analytical) / torch.norm(u_analytical)
    assert l2_err.item() < 0.01, f"Poiseuille forward L2 error {l2_err:.4f} >= 1%"


@pytest.mark.slow
@pytest.mark.validation
def test_poiseuille_gradient(poiseuille_params):
    """C1 gradient gate: implicit-diff matches FD reference to < 0.01%.

    FD reference uses eps=0.01 to stay in the linear convergence regime (large
    enough to avoid floating-point noise from SIMPLE iteration, small enough
    that the nonlinear convection terms are negligible for Re=1 Stokes flow).
    """
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    p = poiseuille_params
    u_inlet_val = float(p["u_inlet"])
    lx, ly = p["lx"], p["ly"]
    re = p["reynolds_number"]
    grid = p["grid"]

    # FD reference: eps=0.01 avoids both floating-point noise (too small eps)
    # and nonlinear effects (too large eps) for Re=1 Stokes flow
    eps = 0.01
    solver_fd = NavierStokes2D(
        reynolds_number=re, grid=grid, lx=lx, ly=ly, tol=1e-8
    )
    ux_p, uy_p, p_p = solver_fd._run_simple(
        None, inlet_velocity=u_inlet_val + eps, case="channel"
    )
    ux_m, uy_m, p_m = solver_fd._run_simple(
        None, inlet_velocity=u_inlet_val - eps, case="channel"
    )
    dp_p = solver_fd.pressure_drop(ux_p, uy_p, p_p)
    dp_m = solver_fd.pressure_drop(ux_m, uy_m, p_m)
    fd_grad = float((dp_p - dp_m) / (2 * eps))

    # Implicit-diff gradient
    u_inlet = torch.tensor(u_inlet_val, dtype=torch.float64, requires_grad=True)
    solver_id = NavierStokes2D(
        reynolds_number=re, grid=grid, lx=lx, ly=ly, backward="implicit_diff", tol=1e-8
    )
    ux, uy, pf = solver_id.solve_steady(sdf=None, inlet_velocity=u_inlet, case="channel")
    dp = solver_id.pressure_drop(ux, uy, pf)
    dp.backward()

    computed_grad = u_inlet.grad.item()
    rel_err = abs(computed_grad - fd_grad) / abs(fd_grad)
    assert rel_err < 1e-4, (
        f"∂ΔP/∂U_inlet: implicit_diff={computed_grad:.6f}, "
        f"FD={fd_grad:.6f}, rel_err={rel_err:.2e}"
    )
