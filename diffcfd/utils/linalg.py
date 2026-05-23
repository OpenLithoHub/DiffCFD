"""Matrix-free GMRES for the implicit differentiation adjoint solve.

The adjoint equation (∂R/∂u)ᵀ λ = b is solved without forming the Jacobian
explicitly — only matvec products (∂R/∂u)·v via torch.func.jvp are needed.
Memory: O(N · restart) where restart is the GMRES restart parameter (~30–50).

Acceptance gate (v0.1): converge within 200 iterations at Re=1000 with
pyamg AMG preconditioner or scipy ILU.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def gmres_matfree(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    restart: int = 30,
) -> tuple[Tensor, int]:
    """Solve A·x = b with matrix-free restarted GMRES (GMRES-m / Arnoldi).

    The matrix A is accessed only through its matvec product A·v.  No explicit
    matrix is formed or stored.  Krylov vectors dominate memory at O(N · restart).

    Args:
        matvec: Callable y = A·v.  Must be differentiable if gradients are needed
                through the solve (not required for the adjoint backward pass itself).
        b: Right-hand side tensor, shape (N,).
        x0: Initial guess; zero vector if None.
        tol: Relative residual tolerance ‖r‖/‖b‖ < tol.
        max_iter: Maximum total iterations across all restarts.
        restart: Krylov subspace size per restart cycle.

    Returns:
        x: Solution tensor, shape (N,).
        iters: Number of iterations used.
    """
    raise NotImplementedError("Implement in v0.1 — Arnoldi GMRES-m.")


def scipy_gmres(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> tuple[Tensor, int]:
    """Thin wrapper around scipy.sparse.linalg.gmres for CPU tensors.

    Fallback path: less control over preconditioning but useful for debugging
    and cross-checking the native gmres_matfree implementation.

    Args:
        matvec: Callable A·v (operates on CPU float64 tensors).
        b: Right-hand side tensor (CPU).
        x0: Initial guess or None.
        tol: GMRES tolerance.
        max_iter: Maximum iterations.

    Returns:
        x: Solution tensor.
        iters: Number of iterations used (0 if converged in 0 steps).
    """
    raise NotImplementedError("Implement scipy GMRES bridge in v0.1.")
