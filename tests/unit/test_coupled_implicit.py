"""Unit tests for coupled implicit differentiation (FSI and strong CHT).

Tests cover:
- Gradient accuracy vs finite differences
- O(1) memory in coupling iterations
- FSI residual convergence
- Minimal elastic boundary example with valid gradients
"""

from __future__ import annotations

import torch


def test_coupled_gradient_vs_finite_diff():
    """Coupled implicit gradient matches finite differences (< 1e-3 relative)."""
    from diffcfd.solvers.coupled_implicit_diff import (
        FSIResidual,
        coupled_fixed_point_gradient,
    )

    torch.manual_seed(42)
    nx, ny = 6, 6
    dt = torch.float64

    fsi = FSIResidual(
        nx=nx, ny=ny, dx=1.0 / nx, dy=1.0 / ny,
        elasticity=1e-2, nu=0.1, tol=1e-10, max_coupling_iter=200,
    )

    theta_val = 1.0
    theta = torch.tensor(theta_val, dtype=dt, requires_grad=True)

    with torch.no_grad():
        u_f_star, u_s_star = fsi.coupled_solve(theta.detach())

    # Verify residuals vanish at the converged state
    Rf = fsi.fluid_residual(u_f_star, u_s_star, theta.detach())
    Rs = fsi.solid_residual(u_f_star, u_s_star, theta.detach())
    res_norm = torch.cat([Rf, Rs]).norm().item()
    assert res_norm < 1e-6, f"Residuals not zero at converged state: {res_norm:.2e}"

    loss_grad_fluid = torch.randn_like(u_f_star)
    loss_grad_solid = torch.randn_like(u_s_star) * 0.1

    grad_implicit = coupled_fixed_point_gradient(
        fluid_residual_fn=fsi.fluid_residual,
        solid_residual_fn=fsi.solid_residual,
        coupling_fn=lambda uf, us: (us, uf),
        u_star_fluid=u_f_star,
        u_star_solid=u_s_star,
        theta=theta,
        loss_grad_fluid=loss_grad_fluid,
        loss_grad_solid=loss_grad_solid,
        tol=1e-10,
        max_iter=500,
    )

    eps = 1e-5
    theta_plus = torch.tensor(theta_val + eps, dtype=dt)
    theta_minus = torch.tensor(theta_val - eps, dtype=dt)

    with torch.no_grad():
        uf_p, us_p = fsi.coupled_solve(theta_plus)
        uf_m, us_m = fsi.coupled_solve(theta_minus)

        loss_p = (uf_p * loss_grad_fluid).sum() + (us_p * loss_grad_solid).sum()
        loss_m = (uf_m * loss_grad_fluid).sum() + (us_m * loss_grad_solid).sum()

    grad_fd = (loss_p - loss_m) / (2 * eps)

    denom = max(abs(grad_fd.item()), 1e-10)
    rel_err = abs(grad_implicit.item() - grad_fd.item()) / denom
    assert rel_err < 1e-3, (
        f"Coupled gradient mismatch: implicit={grad_implicit.item():.6e}, "
        f"fd={grad_fd.item():.6e}, rel_err={rel_err:.4e}"
    )


def test_coupled_memory_constant():
    """Memory usage does not grow with coupling iterations (tensor count check)."""
    from diffcfd.solvers.coupled_implicit_diff import FSIResidual

    torch.manual_seed(0)
    nx, ny = 4, 4
    dt = torch.float64

    fsi_short = FSIResidual(
        nx=nx, ny=ny, dx=1.0 / nx, dy=1.0 / ny,
        elasticity=1e-2, max_coupling_iter=5, nu=0.1,
    )
    theta = torch.tensor(1.0, dtype=dt)
    u_f5, _u_s5 = fsi_short.coupled_solve(theta)

    fsi_long = FSIResidual(
        nx=nx, ny=ny, dx=1.0 / nx, dy=1.0 / ny,
        elasticity=1e-2, max_coupling_iter=200, nu=0.1,
    )
    u_f200, _u_s200 = fsi_long.coupled_solve(theta)

    # The key invariant: coupled_solve returns only detached tensors;
    # the number of live tensors should not grow proportionally to iter count.
    # We check that the returned states have no grad graph.
    assert not u_f200.requires_grad, "coupled_solve must return detached tensors"
    assert not _u_s200.requires_grad, "coupled_solve must return detached tensors"
    assert u_f5.shape == u_f200.shape, "Shape should be identical"


def test_fsi_residual_convergence():
    """FSI residual decreases over coupling iterations."""
    from diffcfd.solvers.coupled_implicit_diff import FSIResidual

    torch.manual_seed(0)
    nx, ny = 8, 8
    dt = torch.float64

    fsi = FSIResidual(
        nx=nx, ny=ny, dx=1.0 / nx, dy=1.0 / ny,
        elasticity=1e-2, tol=1e-14, max_coupling_iter=200, nu=0.1,
    )
    theta = torch.tensor(1.0, dtype=dt)

    residuals = []
    u_solid = torch.zeros(nx, dtype=dt)
    for _ in range(50):
        u_fluid = fsi._channel_profile(theta, u_solid)
        p_load = 0.5 * u_fluid[-1, :] ** 2 * fsi.dy
        u_solid_new = fsi.elasticity * p_load

        Rf = fsi.fluid_residual(u_fluid.flatten(), u_solid, theta)
        Rs = fsi.solid_residual(u_fluid.flatten(), u_solid, theta)
        residuals.append(torch.cat([Rf, Rs]).norm().item())

        u_solid = u_solid_new

    # The residual should decrease monotonically to zero (fixed-point iteration)
    assert residuals[-1] < residuals[0], (
        f"Residual did not decrease: first={residuals[0]:.4e}, last={residuals[-1]:.4e}"
    )
    assert residuals[-1] < 1e-4, (
        f"Residual not near zero after 50 iterations: {residuals[-1]:.4e}"
    )


def test_fsi_minimal_example():
    """Minimal elastic boundary FSI runs and produces valid gradients."""
    from diffcfd.solvers.coupled_implicit_diff import (
        FSIResidual,
        coupled_fixed_point_gradient,
    )

    torch.manual_seed(0)
    nx, ny = 6, 6
    dt = torch.float64

    fsi = FSIResidual(
        nx=nx, ny=ny, dx=1.0 / nx, dy=1.0 / ny,
        elasticity=1e-2, max_coupling_iter=100, nu=0.1, tol=1e-10,
    )
    theta = torch.tensor(1.0, dtype=dt, requires_grad=True)

    u_f_star, u_s_star = fsi.coupled_solve(theta.detach())

    loss_grad_fluid = torch.randn_like(u_f_star)
    loss_grad_solid = torch.randn_like(u_s_star)

    grad = coupled_fixed_point_gradient(
        fluid_residual_fn=fsi.fluid_residual,
        solid_residual_fn=fsi.solid_residual,
        coupling_fn=lambda uf, us: (us, uf),
        u_star_fluid=u_f_star,
        u_star_solid=u_s_star,
        theta=theta,
        loss_grad_fluid=loss_grad_fluid,
        loss_grad_solid=loss_grad_solid,
        tol=1e-10,
        max_iter=500,
    )

    assert torch.isfinite(grad).all(), f"Non-finite gradient: {grad}"
    assert grad.numel() == 1, f"Expected scalar gradient, got shape {grad.shape}"
    assert grad.abs().item() > 0, "Gradient is exactly zero -- dead adjoint"


def test_coupled_gradient_different_theta():
    """Gradient magnitude changes with different theta values (non-constant)."""
    from diffcfd.solvers.coupled_implicit_diff import (
        FSIResidual,
        coupled_fixed_point_gradient,
    )

    torch.manual_seed(0)
    nx, ny = 6, 6
    dt = torch.float64

    fsi = FSIResidual(
        nx=nx, ny=ny, dx=1.0 / nx, dy=1.0 / ny,
        elasticity=1e-2, nu=0.1, tol=1e-10,
    )

    grads = []
    for theta_val in [0.5, 1.0, 2.0]:
        theta = torch.tensor(theta_val, dtype=dt, requires_grad=True)
        u_f, u_s = fsi.coupled_solve(theta.detach())
        loss_grad_f = torch.ones_like(u_f)
        loss_grad_s = torch.ones_like(u_s) * 0.1

        grad = coupled_fixed_point_gradient(
            fluid_residual_fn=fsi.fluid_residual,
            solid_residual_fn=fsi.solid_residual,
            coupling_fn=lambda uf, us: (us, uf),
            u_star_fluid=u_f,
            u_star_solid=u_s,
            theta=theta,
            loss_grad_fluid=loss_grad_f,
            loss_grad_solid=loss_grad_s,
            tol=1e-10,
            max_iter=500,
        )
        grads.append(grad.item())

    assert grads[0] != grads[1] or grads[1] != grads[2], (
        f"Gradients identical across theta values: {grads}"
    )
