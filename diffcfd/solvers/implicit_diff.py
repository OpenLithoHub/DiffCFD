"""Fixed-point implicit differentiation through SIMPLE-converged steady state.

C1 patent claim core: the dual-function architecture.
- Forward: under-relaxed SIMPLE (stable convergence, not used in backward).
- Backward: matrix-free GMRES on unrelaxed physics residual R(u*, θ) = 0.

This separation ensures exact gradients at the converged state independent of
the relaxation factors used in the forward pass.

Reference: Bai et al. 2019 (Deep Equilibrium Models) for fixed-point implicit diff.
GMRES matvec oracle uses torch.func.jvp on the pure physics residual.

v0.7: optional physics-based preconditioner for stiff Brinkman (ε ≤ 1e-5) systems.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from diffcfd.utils.linalg import gmres_matfree


def _jacobi_precond(
    residual_fn: Callable[[Tensor, Tensor], Tensor],
    u_star: Tensor,
    theta_d: Tensor,
    eps: float = 1e-6,
) -> Callable[[Tensor], Tensor]:
    """Build a Jacobi (diagonal) right preconditioner from the residual Jacobian diagonal.

    Approximates ``diag(∂R/∂u)ᵀ`` via forward finite differences and returns
    ``M⁻¹ v = v / diag``.  For Brinkman penalisation the diagonal is dominated
    by the momentum + penalty terms, making Jacobi an effective low-cost
    preconditioner.

    Args:
        residual_fn: Pure physics residual R(u, θ).
        u_star: Converged state (detached).
        theta_d: Design parameters (detached, with grad).
        eps: Finite-difference step for diagonal approximation.

    Returns:
        Right preconditioner callable.
    """
    u = u_star.detach().clone()
    R0 = residual_fn(u, theta_d).detach()

    N = u.numel()
    diag = torch.zeros_like(u)

    u_flat = u.flatten()
    for i in range(N):
        u_pert = u_flat.clone()
        u_pert[i] += eps
        R_pert = residual_fn(u_pert.reshape(u.shape), theta_d).detach()
        diag_flat = (R_pert - R0).flatten()
        diag[i] = diag_flat[i] / eps

    # Regularise to avoid division by near-zero diagonal entries
    diag = torch.where(diag.abs() < 1e-10, torch.ones_like(diag), diag)

    def precond(v: Tensor) -> Tensor:
        return v / diag

    return precond


def _block_diag_precond(
    residual_fn: Callable[[Tensor, Tensor], Tensor],
    u_star: Tensor,
    theta_d: Tensor,
    block_size: int = 4,
    eps: float = 1e-6,
) -> Callable[[Tensor], Tensor]:
    """Build a block-diagonal right preconditioner.

    Partitions the state vector into blocks of ``block_size`` and inverts each
    block's Jacobian sub-matrix.  More effective than Jacobi for coupled
    momentum equations where off-diagonal coupling is significant.

    Args:
        residual_fn: Pure physics residual R(u, θ).
        u_star: Converged state (detached).
        theta_d: Design parameters (detached, with grad).
        block_size: Number of state variables per block (e.g. 4 for u,v,p,k).
        eps: Finite-difference step for Jacobian approximation.

    Returns:
        Right preconditioner callable.
    """
    u = u_star.detach().clone()
    R0 = residual_fn(u, theta_d).detach()

    N = u.numel()
    n_blocks = (N + block_size - 1) // block_size
    inv_blocks: list[torch.Tensor] = []

    u_flat = u.flatten()
    R0_flat = R0.flatten()
    out_N = R0_flat.shape[0]

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, N)
        bs = end - start

        J_block = torch.zeros(out_N, bs, dtype=u.dtype, device=u.device)
        for j in range(bs):
            u_pert = u_flat.clone()
            u_pert[start + j] += eps
            R_pert = residual_fn(u_pert.reshape(u.shape), theta_d).detach().flatten()
            J_block[:, j] = (R_pert - R0_flat) / eps

        # Extract the corresponding output rows and invert
        out_start = min(start, out_N - bs)
        out_end = min(out_start + bs, out_N)
        actual_bs = out_end - out_start
        if actual_bs > 0:
            sub = J_block[out_start:out_end, :actual_bs]
            try:
                inv_sub = torch.linalg.inv(sub + 1e-8 * torch.eye(
                    actual_bs, dtype=u.dtype, device=u.device))
            except torch.linalg.LinAlgError:
                inv_sub = torch.eye(actual_bs, dtype=u.dtype, device=u.device)
            inv_blocks.append(inv_sub)
        else:
            inv_blocks.append(torch.eye(bs, dtype=u.dtype, device=u.device))

    def precond(v: Tensor) -> Tensor:
        v_flat = v.flatten()
        result = torch.zeros_like(v_flat)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, N)
            bs = end - start
            result[start:end] = inv_blocks[b] @ v_flat[start:end]
        return result.reshape(v.shape)

    return precond


def fixed_point_gradient(
    residual_fn: Callable[[Tensor, Tensor], Tensor],
    u_star: Tensor,
    theta: Tensor,
    loss_grad: Tensor,
    tol: float = 1e-6,
    max_iter: int = 200,
    precond: str | Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Compute dL/dθ via the implicit function theorem at the fixed point u*.

    Solves the adjoint equation (∂R/∂u)ᵀ λ = ∂L/∂u via matrix-free GMRES,
    then returns dL/dθ = -(∂R/∂θ)ᵀ λ.

    Args:
        residual_fn: Pure physics residual R(u, θ) — no relaxation, no damping.
        u_star: Converged steady-state solution (gradient-detached).
        theta: Design parameters tensor (requires_grad=True).
        loss_grad: ∂L/∂u — gradient of loss w.r.t. u at u*.
        tol: GMRES convergence tolerance.
        max_iter: Maximum GMRES iterations (acceptance gate: < 200 at Re=1000).
        precond: Preconditioner strategy.  Options:
            - ``None`` (default): no preconditioning, backward compatible.
            - ``"jacobi"``: Jacobi (diagonal) preconditioner.
            - ``"block_diag"``: Block-diagonal preconditioner (block_size=4).
            - ``Callable``: Custom right preconditioner M⁻¹.

    Returns:
        dL/dθ tensor, same shape as theta.
    """
    u_star_d = u_star.detach()
    theta_d = theta.detach().requires_grad_(True)

    def matvec_Jt(v: Tensor) -> Tensor:
        _, vjp_fn = torch.func.vjp(lambda u: residual_fn(u, theta_d), u_star_d)
        return vjp_fn(v)[0]

    # Build preconditioner if requested
    M_inv: Callable[[Tensor], Tensor] | None = None
    if isinstance(precond, str):
        if precond == "jacobi":
            M_inv = _jacobi_precond(residual_fn, u_star_d, theta_d)
        elif precond == "block_diag":
            M_inv = _block_diag_precond(residual_fn, u_star_d, theta_d)
        else:
            raise ValueError(f"Unknown preconditioner: {precond!r}")
    elif callable(precond):
        M_inv = precond

    lambda_, _ = gmres_matfree(
        matvec_Jt, loss_grad.detach(), tol=tol, max_iter=max_iter, precond=M_inv
    )

    # dL/dθ = -(∂R/∂θ)ᵀ λ
    _, vjp_fn_theta = torch.func.vjp(lambda th: residual_fn(u_star_d, th), theta_d)
    dL_dtheta = -vjp_fn_theta(lambda_.detach())[0]

    return dL_dtheta
