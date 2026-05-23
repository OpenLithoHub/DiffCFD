"""Fixed-point implicit differentiation through SIMPLE-converged steady state.

C1 patent claim core: the dual-function architecture.
- Forward: under-relaxed SIMPLE (stable convergence, not used in backward).
- Backward: matrix-free GMRES on unrelaxed physics residual R(u*, θ) = 0.

This separation ensures exact gradients at the converged state independent of
the relaxation factors used in the forward pass.

Reference: Bai et al. 2019 (Deep Equilibrium Models) for fixed-point implicit diff.
GMRES matvec oracle uses torch.func.jvp on the pure physics residual.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from diffcfd.utils.linalg import gmres_matfree


def fixed_point_gradient(
    residual_fn: Callable[[Tensor, Tensor], Tensor],
    u_star: Tensor,
    theta: Tensor,
    loss_grad: Tensor,
    tol: float = 1e-6,
    max_iter: int = 200,
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

    Returns:
        dL/dθ tensor, same shape as theta.
    """
    u_star_d = u_star.detach()
    theta_d = theta.detach().requires_grad_(True)

    def matvec_Jt(v: Tensor) -> Tensor:
        # (∂R/∂u)ᵀ v via vjp: vjp(u → R(u, θ), u*)[v]
        _, vjp_fn = torch.func.vjp(lambda u: residual_fn(u, theta_d), u_star_d)
        return vjp_fn(v)[0]

    lambda_, _ = gmres_matfree(matvec_Jt, loss_grad.detach(), tol=tol, max_iter=max_iter)

    # dL/dθ = -(∂R/∂θ)ᵀ λ
    _, vjp_fn_theta = torch.func.vjp(lambda th: residual_fn(u_star_d, th), theta_d)
    dL_dtheta = -vjp_fn_theta(lambda_.detach())[0]

    return dL_dtheta
