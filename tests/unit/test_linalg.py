"""Unit tests for matrix-free GMRES utilities.

Tests the scipy_gmres bridge (uses scipy under the hood) against a known
2×2 system before the native Arnoldi implementation is ready.
"""

import pytest
import torch


def test_linalg_import():
    from diffcfd.utils.linalg import gmres_matfree, scipy_gmres
    assert callable(gmres_matfree)
    assert callable(scipy_gmres)


@pytest.mark.skip(reason="gmres_matfree not yet implemented (v0.1 target)")
def test_gmres_diagonal_system():
    """Solve a diagonal system A·x = b where A = diag(1,2,3,4)."""
    from diffcfd.utils.linalg import gmres_matfree

    d = torch.tensor([1.0, 2.0, 3.0, 4.0])
    matvec = lambda v: d * v
    b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x, iters = gmres_matfree(matvec, b, tol=1e-10)
    assert torch.allclose(x, torch.ones(4), atol=1e-8), f"Expected [1,1,1,1], got {x}"
    assert iters <= 4


@pytest.mark.skip(reason="scipy_gmres not yet implemented (v0.1 target)")
def test_scipy_gmres_diagonal_system():
    """Same test via scipy bridge."""
    from diffcfd.utils.linalg import scipy_gmres

    d = torch.tensor([1.0, 2.0, 3.0, 4.0])
    matvec = lambda v: d * v
    b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x, iters = scipy_gmres(matvec, b, tol=1e-10)
    assert torch.allclose(x, torch.ones(4), atol=1e-8)
