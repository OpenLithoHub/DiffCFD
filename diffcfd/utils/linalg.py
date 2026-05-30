"""Matrix-free GMRES for the implicit differentiation adjoint solve.

The adjoint equation (∂R/∂u)ᵀ λ = b is solved without forming the Jacobian
explicitly — only matvec products via torch.func.vjp are needed.
Memory: O(N · restart) where restart is the GMRES restart parameter (~30–50).

Acceptance gate (v0.1): converge within 200 iterations at Re=1000.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse.linalg as spla
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

    Uses Modified Gram-Schmidt Arnoldi + Givens rotations for numerical stability.
    All operations are PyTorch tensors; no matrix is formed explicitly.
    Memory: O(N · restart) for the Krylov basis.

    Args:
        matvec: Callable y = A·v.
        b: Right-hand side tensor, shape (N,).
        x0: Initial guess; zero vector if None.
        tol: Relative residual tolerance ‖r‖/‖b‖ < tol.
        max_iter: Maximum total iterations across all restarts.
        restart: Krylov subspace size per restart cycle.

    Returns:
        x: Solution tensor, shape (N,).
        iters: Number of iterations used.
    """
    dtype = b.dtype
    device = b.device
    b_norm = b.norm()
    if b_norm == 0:
        return torch.zeros_like(b), 0

    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    total_iters = 0
    converged = False
    r = None  # reuse residual across restart cycles

    for _ in range(max(1, max_iter // restart + 1)):
        if r is None:
            r = b - matvec(x)
        r_norm = r.norm()
        if r_norm < tol * b_norm:
            converged = True
            break

        m = min(restart, max_iter - total_iters)
        if m <= 0:
            break

        Q = [r / r_norm]
        H = torch.zeros(m + 1, m, dtype=dtype, device=device)
        cs = torch.zeros(m, dtype=dtype, device=device)
        sn = torch.zeros(m, dtype=dtype, device=device)
        e1 = torch.zeros(m + 1, dtype=dtype, device=device)
        e1[0] = r_norm

        j_used = 0
        for j in range(m):
            w = matvec(Q[j])
            for i in range(j + 1):
                H[i, j] = torch.dot(w, Q[i])
                w = w - H[i, j] * Q[i]
            H[j + 1, j] = w.norm()
            if H[j + 1, j] > 1e-14:
                Q.append(w / H[j + 1, j])
            else:
                Q.append(torch.zeros_like(w))

            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp

            h_jj = H[j, j]
            h_j1j = H[j + 1, j]
            denom = torch.sqrt(h_jj * h_jj + h_j1j * h_j1j)
            if denom < 1e-14:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = h_jj / denom
                sn[j] = h_j1j / denom

            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            e1[j + 1] = -sn[j] * e1[j]
            e1[j] = cs[j] * e1[j]

            total_iters += 1
            j_used = j + 1
            if abs(e1[j + 1].item()) < tol * b_norm.item():
                converged = True
                break

        if j_used == 0:
            break

        H_sq = H[:j_used, :j_used]
        e_sq = e1[:j_used]
        y = torch.linalg.solve_triangular(H_sq, e_sq.unsqueeze(1), upper=True).squeeze(
            1
        )
        Q_mat = torch.stack(Q[:j_used], dim=1)
        x = x + Q_mat @ y

        if converged:
            break

        # Compute residual for next restart cycle (avoids redundant matvec)
        r = b - matvec(x)

    return x, total_iters


def scipy_gmres(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> tuple[Tensor, int]:
    """Thin wrapper around scipy.sparse.linalg.gmres for CPU tensors.

    Fallback path for debugging and cross-checking gmres_matfree.

    Args:
        matvec: Callable A·v (CPU float32 or float64 tensors).
        b: Right-hand side tensor (CPU).
        x0: Initial guess or None.
        tol: GMRES tolerance.
        max_iter: Maximum iterations.

    Returns:
        x: Solution tensor.
        iters: Iteration count (approximated via callback).
    """
    N = b.numel()
    dtype_np = np.float64

    def matvec_np(v_np):
        v = torch.tensor(v_np, dtype=b.dtype, device=b.device)
        return matvec(v).detach().cpu().numpy().astype(dtype_np)

    A_lo = spla.LinearOperator((N, N), matvec=matvec_np)
    b_np = b.detach().cpu().numpy().astype(dtype_np)
    x0_np = x0.detach().cpu().numpy().astype(dtype_np) if x0 is not None else None

    iters_count = [0]

    def callback(_):
        iters_count[0] += 1

    x_np, info = spla.gmres(
        A_lo,
        b_np,
        x0=x0_np,
        rtol=tol,
        maxiter=max_iter,
        callback=callback,
        callback_type="legacy",
    )
    x = torch.tensor(x_np, dtype=b.dtype, device=b.device)
    return x, iters_count[0]
