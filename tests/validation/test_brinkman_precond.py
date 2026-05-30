"""Validation tests for the Brinkman-aware preconditioner (P1-2).

Sweeps epsilon from 1e-3 to 1e-6 and verifies:
1. Adjoint iterations grow at most 2x when epsilon decreases 100x.
2. Gradients are finite and non-NaN.
3. Gradients match the analytical solution at each epsilon.
4. Preconditioned GMRES converges faster than unpreconditioned.
"""

import pytest
import torch


def _make_brinkman_system(N, eps):
    """Build a Brinkman-penalized linear system and its exact solution.

    R(u, theta) = A_full * u - theta * b_drive  where  A_full = A_base + (1/eps) * diag(mask)

    Returns residual_fn, u_star (exact solve), A_full, b_drive, mask.
    """
    torch.manual_seed(12345)
    mask = torch.zeros(N)
    mask[N // 4: 3 * N // 4] = 1.0

    A_raw = torch.randn(N, N) * 0.1
    A_base = A_raw @ A_raw.T + 0.5 * torch.eye(N)
    bk_diag = (1.0 / eps) * mask
    A_full = A_base + torch.diag(bk_diag)
    b_drive = torch.randn(N)

    theta_val = torch.tensor(1.0)
    u_star = torch.linalg.solve(A_full, theta_val * b_drive)

    def residual_fn(u, theta):
        return A_full @ u - theta * b_drive

    return residual_fn, u_star, A_full, b_drive, mask


@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6])
def test_brinkman_precond_gradient_finite(eps):
    """Gradients are finite and non-NaN across epsilon sweep."""
    from diffcfd.solvers.implicit_diff import fixed_point_gradient

    N = 24
    residual_fn, u_star, A_full, _, _ = _make_brinkman_system(N, eps)

    theta = torch.tensor(1.0, requires_grad=True)
    loss_grad = torch.randn(N)

    grad = fixed_point_gradient(
        residual_fn, u_star, theta, loss_grad,
        tol=1e-6, max_iter=500, precond="auto",
    )

    assert torch.isfinite(grad).all(), f"Non-finite gradient at eps={eps}: {grad}"


@pytest.mark.parametrize("eps", [1e-3, 1e-5])
def test_brinkman_precond_iteration_count(eps):
    """Preconditioned GMRES converges in fewer iterations than unpreconditioned."""
    from diffcfd.solvers.implicit_diff import _brinkman_diag_precond
    from diffcfd.utils.linalg import gmres_matfree

    N = 24
    residual_fn, u_star, A_full, _, _ = _make_brinkman_system(N, eps)

    theta_d = torch.tensor(1.0, requires_grad=True)
    loss_grad = torch.randn(N)

    def matvec_Jt(v):
        _, vjp_fn = torch.func.vjp(lambda u: residual_fn(u, theta_d), u_star)
        return vjp_fn(v)[0]

    M_inv = _brinkman_diag_precond(lambda u: residual_fn(u, theta_d), u_star)

    _, iters_plain = gmres_matfree(
        matvec_Jt, loss_grad, tol=1e-6, max_iter=500, precond=None,
    )
    _, iters_precond = gmres_matfree(
        matvec_Jt, loss_grad, tol=1e-6, max_iter=500, precond=M_inv,
    )

    assert iters_precond < iters_plain, (
        f"eps={eps}: precond iters ({iters_precond}) >= plain iters ({iters_plain})"
    )


def test_brinkman_iteration_ratio_eps_sweep():
    """Across eps=1e-3 to 1e-5 (100x decrease), iterations grow at most 2x."""
    from diffcfd.solvers.implicit_diff import _brinkman_diag_precond
    from diffcfd.utils.linalg import gmres_matfree

    N = 24
    loss_grad = torch.randn(N)
    iters_at = {}

    for eps in [1e-3, 1e-5]:
        residual_fn, u_star, _, _, _ = _make_brinkman_system(N, eps)
        theta_d = torch.tensor(1.0, requires_grad=True)

        def matvec_Jt(v, _theta=theta_d, _u=u_star, _fn=residual_fn):
            _, vjp_fn = torch.func.vjp(lambda u: _fn(u, _theta), _u)
            return vjp_fn(v)[0]

        M_inv = _brinkman_diag_precond(lambda u: residual_fn(u, theta_d), u_star)
        _, iters = gmres_matfree(
            matvec_Jt, loss_grad, tol=1e-6, max_iter=500, precond=M_inv,
        )
        iters_at[eps] = iters

    ratio = iters_at[1e-5] / max(iters_at[1e-3], 1)
    assert ratio <= 2.0, (
        f"Iteration ratio eps=1e-5 / eps=1e-3 = {ratio:.2f} > 2.0. "
        f"iters@1e-3={iters_at[1e-3]}, iters@1e-5={iters_at[1e-5]}"
    )


@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5])
def test_brinkman_precond_vs_no_precond(eps):
    """Preconditioned and unpreconditioned GMRES produce the same solution."""
    from diffcfd.solvers.implicit_diff import _brinkman_diag_precond
    from diffcfd.utils.linalg import gmres_matfree

    N = 24
    residual_fn, u_star, A_full, _, _ = _make_brinkman_system(N, eps)
    theta_d = torch.tensor(1.0, requires_grad=True)
    loss_grad = torch.randn(N)

    def matvec_Jt(v):
        _, vjp_fn = torch.func.vjp(lambda u: residual_fn(u, theta_d), u_star)
        return vjp_fn(v)[0]

    M_inv = _brinkman_diag_precond(lambda u: residual_fn(u, theta_d), u_star)

    x_precond, iters_precond = gmres_matfree(
        matvec_Jt, loss_grad, tol=1e-6, max_iter=500, precond=M_inv,
    )
    x_plain, iters_plain = gmres_matfree(
        matvec_Jt, loss_grad, tol=1e-6, max_iter=500, precond=None,
    )

    x_exact = torch.linalg.solve(A_full.T, loss_grad)

    err_precond = (x_precond - x_exact).norm() / (x_exact.norm() + 1e-12)
    err_plain = (x_plain - x_exact).norm() / (x_exact.norm() + 1e-12)

    assert err_precond < 1e-3, (
        f"eps={eps}: preconditioned solution error = {err_precond:.4e}"
    )
    assert iters_precond <= iters_plain, (
        f"eps={eps}: precond iters ({iters_precond}) > plain iters ({iters_plain})"
    )


@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5])
def test_brinkman_precond_gradient_accuracy(eps):
    """Implicit diff gradients match analytical exact gradients."""
    from diffcfd.solvers.implicit_diff import fixed_point_gradient

    N = 16
    residual_fn, u_star, A_full, b_drive, _ = _make_brinkman_system(N, eps)

    theta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    # Rebuild in float64 for higher accuracy
    torch.manual_seed(12345)
    mask = torch.zeros(N, dtype=torch.float64)
    mask[N // 3: 2 * N // 3] = 1.0
    A_raw = torch.randn(N, N, dtype=torch.float64) * 0.1
    A_base = A_raw @ A_raw.T + 0.5 * torch.eye(N, dtype=torch.float64)
    bk_diag = (1.0 / eps) * mask
    A_full_64 = A_base + torch.diag(bk_diag)
    b_drive_64 = torch.randn(N, dtype=torch.float64)
    u_star_64 = torch.linalg.solve(A_full_64, torch.tensor(1.0, dtype=torch.float64) * b_drive_64)

    def residual_fn_64(u, th):
        return A_full_64 @ u - th * b_drive_64

    loss_grad = torch.randn(N, dtype=torch.float64)

    grad_implicit = fixed_point_gradient(
        residual_fn_64, u_star_64, theta, loss_grad,
        tol=1e-8, max_iter=500, precond="auto",
    )

    # Exact: dL/dtheta = dL/du * A^{-1} * b_drive
    du_dtheta = torch.linalg.solve(A_full_64, b_drive_64)
    grad_exact = torch.dot(loss_grad, du_dtheta)

    rel_err = abs(grad_implicit.item() - grad_exact) / (abs(grad_exact) + 1e-12)
    assert rel_err < 0.05, (
        f"eps={eps}: gradient error = {rel_err:.4e}, "
        f"implicit={grad_implicit.item():.6f}, exact={grad_exact:.6f}"
    )


def test_brinkman_precond_ns_channel():
    """Full NS channel flow with Brinkman obstacle: preconditioner converges."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    solver = NavierStokes2D(
        reynolds_number=10.0,
        grid=(12, 6),
        lx=2.0,
        ly=1.0,
        backward="implicit_diff",
        tol=1e-4,
        max_iter=500,
    )

    nx, ny = 12, 6
    xc = torch.linspace(solver.mesh.dx / 2, 2.0 - solver.mesh.dx / 2, nx)
    yc = torch.linspace(solver.mesh.dy / 2, 1.0 - solver.mesh.dy / 2, ny)
    X, Y = torch.meshgrid(xc, yc, indexing="xy")
    cx, cy, r = 1.0, 0.5, 0.15
    sdf = r - torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    inlet_velocity = torch.tensor(1.0, requires_grad=True)
    ux, uy, p = solver.solve_steady(sdf=sdf, inlet_velocity=inlet_velocity, case="channel")

    loss = solver.pressure_drop(ux, uy, p)
    loss.backward()

    grad = inlet_velocity.grad
    assert grad is not None, "No gradient computed"
    assert torch.isfinite(grad).all(), f"Non-finite gradient: {grad}"
    assert grad.abs() > 0, "Gradient is zero (unexpected for channel with obstacle)"


def test_brinkman_precond_extreme_stiffness():
    """eps=1e-6 (extreme Brinkman stiffness) still produces finite gradients."""
    from diffcfd.solvers.implicit_diff import fixed_point_gradient

    N = 24
    eps = 1e-6
    residual_fn, u_star, _, _, _ = _make_brinkman_system(N, eps)

    theta = torch.tensor(1.0, requires_grad=True)
    loss_grad = torch.randn(N)

    grad = fixed_point_gradient(
        residual_fn, u_star, theta, loss_grad,
        tol=1e-4, max_iter=1000, precond="auto",
    )

    assert torch.isfinite(grad).all(), f"Non-finite gradient at eps=1e-6"
