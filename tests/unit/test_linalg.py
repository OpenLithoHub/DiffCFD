"""Unit tests for matrix-free GMRES utilities."""

import torch


def test_linalg_import():
    from diffcfd.utils.linalg import gmres_matfree, scipy_gmres

    assert callable(gmres_matfree)
    assert callable(scipy_gmres)


def test_gmres_diagonal_system():
    """Solve a diagonal system A·x = b where A = diag(1,2,3,4)."""
    from diffcfd.utils.linalg import gmres_matfree

    d = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x, iters = gmres_matfree(lambda v: d * v, b, tol=1e-6)
    assert torch.allclose(x, torch.ones(4), atol=1e-5), f"Expected [1,1,1,1], got {x}"
    assert iters <= 8  # at most 2 restart cycles for 4-dim system


def test_scipy_gmres_diagonal_system():
    """Same test via scipy bridge."""
    from diffcfd.utils.linalg import scipy_gmres

    d = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x, iters = scipy_gmres(lambda v: d * v, b, tol=1e-10)
    assert torch.allclose(x, torch.ones(4), atol=1e-8)


def test_gmres_random_spd():
    """Solve a random 20×20 SPD system; compare to torch.linalg.solve."""
    from diffcfd.utils.linalg import gmres_matfree

    torch.manual_seed(42)
    A_raw = torch.randn(20, 20)
    A = A_raw @ A_raw.T + 5 * torch.eye(20)  # SPD, well-conditioned
    b = torch.randn(20)
    x_ref = torch.linalg.solve(A, b)
    x, iters = gmres_matfree(lambda v: A @ v, b, tol=1e-6, restart=20)
    assert torch.allclose(x, x_ref, atol=1e-4), f"Max diff: {(x - x_ref).abs().max()}"
    assert iters <= 40  # at most 2 restarts of size 20


def test_gmres_zero_rhs():
    """Zero RHS should return zero solution in 0 iterations."""
    from diffcfd.utils.linalg import gmres_matfree

    A = torch.eye(5)
    b = torch.zeros(5)
    x, iters = gmres_matfree(lambda v: A @ v, b)
    assert torch.allclose(x, torch.zeros(5))
    assert iters == 0


# --- Preconditioner tests (WS-C) ---


def test_gmres_precond_backward_compatible():
    """precond=None produces identical results to the old signature."""
    from diffcfd.utils.linalg import gmres_matfree

    torch.manual_seed(123)
    A = torch.randn(10, 10)
    A = A @ A.T + 3 * torch.eye(10)
    b = torch.randn(10)

    x_no_precond, iters_no = gmres_matfree(lambda v: A @ v, b, tol=1e-6)
    x_none, iters_none = gmres_matfree(lambda v: A @ v, b, tol=1e-6, precond=None)
    assert torch.allclose(x_no_precond, x_none, atol=1e-10)


def test_gmres_jacobi_precond_ill_conditioned():
    """Jacobi preconditioner improves convergence on ill-conditioned system."""
    from diffcfd.utils.linalg import gmres_matfree

    torch.manual_seed(42)
    d = torch.tensor([1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    A = torch.diag(d) + 0.01 * torch.randn(8, 8)
    A = A + A.T
    b = torch.randn(8)
    x_ref = torch.linalg.solve(A, b)

    diag_inv = 1.0 / A.diag().clamp(min=1e-10)
    x_precond, _ = gmres_matfree(
        lambda v: A @ v, b, tol=1e-6, max_iter=200,
        precond=lambda v: diag_inv * v,
    )
    assert torch.allclose(x_precond, x_ref, atol=1e-3)


def test_gmres_identity_precond_unchanged():
    """Identity preconditioner gives same result as no preconditioner."""
    from diffcfd.utils.linalg import gmres_matfree

    torch.manual_seed(99)
    A_raw = torch.randn(6, 6)
    A = A_raw @ A_raw.T + 2 * torch.eye(6)
    b = torch.randn(6)

    x_plain, _ = gmres_matfree(lambda v: A @ v, b, tol=1e-8)
    x_identity, _ = gmres_matfree(
        lambda v: A @ v, b, tol=1e-8,
        precond=lambda v: v,
    )
    assert torch.allclose(x_plain, x_identity, atol=1e-8)


def test_implicit_diff_precond_option():
    """fixed_point_gradient accepts precond='jacobi' without error."""
    from diffcfd.solvers.implicit_diff import fixed_point_gradient

    torch.manual_seed(7)
    N = 8
    u_star = torch.randn(N)
    theta = torch.randn(N, requires_grad=True)
    loss_grad = torch.randn(N)

    def residual_fn(u, th):
        return 2.0 * u - th

    grad_plain = fixed_point_gradient(residual_fn, u_star, theta, loss_grad, tol=1e-6)
    grad_jacobi = fixed_point_gradient(
        residual_fn, u_star, theta, loss_grad, tol=1e-6, precond="jacobi"
    )

    assert not torch.isnan(grad_jacobi).any()
    # Both should give similar gradients for this simple system
    rel_err = (grad_plain - grad_jacobi).norm() / (grad_plain.norm() + 1e-12)
    assert rel_err < 0.1, f"Gradient mismatch: rel_err={rel_err:.4e}"


def test_gmres_stiff_brinkman_like():
    """Stiff Brinkman-like system converges with preconditioner."""
    from diffcfd.utils.linalg import gmres_matfree

    torch.manual_seed(2024)
    N = 16
    eps = 1e-3  # Moderate Brinkman penalty

    # A = small random SPD + eps^{-1} * diag(mask)
    mask = torch.zeros(N)
    mask[N // 4: 3 * N // 4] = 1.0
    A_raw = torch.randn(N, N)
    A_base = A_raw @ A_raw.T + 0.1 * torch.eye(N)
    A = A_base + (1.0 / eps) * torch.diag(mask)
    b = torch.randn(N)
    x_ref = torch.linalg.solve(A, b)

    diag_inv = 1.0 / A.diag().clamp(min=1e-10)
    x_precond, _ = gmres_matfree(
        lambda v: A @ v, b, tol=1e-5, max_iter=300,
        precond=lambda v: diag_inv * v,
    )

    assert torch.allclose(x_precond, x_ref, atol=1e-2)
