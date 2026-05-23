"""Unit tests for matrix-free GMRES utilities."""

import pytest
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
    assert iters <= 8   # at most 2 restart cycles for 4-dim system


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
    A = A_raw @ A_raw.T + 5 * torch.eye(20)   # SPD, well-conditioned
    b = torch.randn(20)
    x_ref = torch.linalg.solve(A, b)
    x, iters = gmres_matfree(lambda v: A @ v, b, tol=1e-6, restart=20)
    assert torch.allclose(x, x_ref, atol=1e-4), f"Max diff: {(x - x_ref).abs().max()}"
    assert iters <= 40   # at most 2 restarts of size 20


def test_gmres_zero_rhs():
    """Zero RHS should return zero solution in 0 iterations."""
    from diffcfd.utils.linalg import gmres_matfree

    A = torch.eye(5)
    b = torch.zeros(5)
    x, iters = gmres_matfree(lambda v: A @ v, b)
    assert torch.allclose(x, torch.zeros(5))
    assert iters == 0
