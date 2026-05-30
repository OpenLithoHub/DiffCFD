"""Fixed-point implicit differentiation through SIMPLE-converged steady state.

Dual-function architecture:
- Forward: under-relaxed SIMPLE (stable convergence, not used in backward).
- Backward: matrix-free GMRES on unrelaxed physics residual R(u*, theta) = 0.

This separation ensures exact gradients at the converged state independent of
the relaxation factors used in the forward pass.

Reference: Bai et al. 2019 (Deep Equilibrium Models) for fixed-point implicit diff.
GMRES matvec oracle uses torch.func.jvp on the pure physics residual.

v0.8: Brinkman-aware diagonal preconditioner via single JVP — replaces
v0.7 Jacobi/block-diagonal preconditioners that required O(N) residual
evaluations and could not handle eps <= 1e-5.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from diffcfd.utils.linalg import gmres_matfree


def _brinkman_diag_precond(
    residual_fn: Callable[[Tensor], Tensor],
    z_star: Tensor,
) -> Callable[[Tensor], Tensor]:
    """Build a diagonal right preconditioner via a single forward JVP.

    For the adjoint system (∂R/∂z)^T λ = b, the diagonal of the transposed
    Jacobian equals the diagonal of the Jacobian itself. We extract this
    diagonal with one JVP using the all-ones vector:

        diag(∂R/∂z) ≈ JVP(R, z*, e)  (element-wise, when R_i depends weakly
        on off-diagonal z_j for j ≠ i)

    More precisely, the JVP with e gives the row-sum of the Jacobian.
    For the NS+Brinkman system, the momentum equations are dominated by the
    diagonal term a_P0 = a_e + a_w + a_n + a_s + bk (where bk = 1/epsilon
    in solid cells), so the row-sum is an excellent approximation to the
    diagonal. In fluid cells the row-sum equals the diagonal exactly
    (conservation form); in solid cells the diagonal dominates by O(1/eps).

    This avoids the O(N) cost of the previous finite-difference Jacobi
    preconditioner and correctly captures the 1/epsilon stiffness that
    causes GMRES to stagnate at eps <= 1e-5.

    The preconditioner M^{-1} = 1 / diag(J) is applied on the right:
        GMRES solves  A M^{-1} y = b,  then  x = M^{-1} y.

    Since (A M^{-1})^T = M^{-T} A^T and M is diagonal, M^{-T} = M^{-1},
    so this is equivalent to left-preconditioning A^T with M^{-1} — exactly
    what we want for the adjoint system.

    Args:
        residual_fn: Pure physics residual R(z), already bound to theta.
        z_star: Converged state (detached).

    Returns:
        Right preconditioner callable.
    """
    z = z_star.detach()
    ones = torch.ones_like(z)

    _, jvp_val = torch.func.jvp(residual_fn, (z,), (ones,))

    # jvp_val = J @ ones = row sums of J ≈ diagonal for dominant-diagonal systems.
    # Regularize to avoid division by near-zero entries (e.g. pressure gauge pin).
    diag = jvp_val.detach()
    diag = torch.where(diag.abs() < 1e-10, torch.ones_like(diag), diag)
    inv_diag = 1.0 / diag

    def precond(v: Tensor) -> Tensor:
        return v * inv_diag

    return precond


def fixed_point_gradient(
    residual_fn: Callable[[Tensor, Tensor], Tensor],
    u_star: Tensor,
    theta: Tensor,
    loss_grad: Tensor,
    tol: float = 1e-6,
    max_iter: int = 200,
    precond: str | Callable[[Tensor], Tensor] | None = "auto",
) -> Tensor:
    """Compute dL/dtheta via the implicit function theorem at the fixed point u*.

    Solves the adjoint equation (∂R/∂u)^T lambda = ∂L/∂u via matrix-free GMRES,
    then returns dL/dtheta = -(∂R/∂theta)^T lambda.

    Args:
        residual_fn: Pure physics residual R(u, theta) -- no relaxation, no damping.
        u_star: Converged steady-state solution (gradient-detached).
        theta: Design parameters tensor (requires_grad=True).
        loss_grad: ∂L/∂u -- gradient of loss w.r.t. u at u*.
        tol: GMRES convergence tolerance.
        max_iter: Maximum GMRES iterations.
        precond: Preconditioner strategy.  Options:
            - ``"auto"`` (default): Diagonal preconditioner via single JVP.
              Handles Brinkman stiffness down to eps ~ 1e-6.
            - ``None``: No preconditioning.
            - ``Callable``: Custom right preconditioner M^{-1}.

    Returns:
        dL/dtheta tensor, same shape as theta.
    """
    u_star_d = u_star.detach()
    theta_d = theta.detach().requires_grad_(True)

    def matvec_Jt(v: Tensor) -> Tensor:
        _, vjp_fn = torch.func.vjp(lambda u: residual_fn(u, theta_d), u_star_d)
        return vjp_fn(v)[0]

    M_inv: Callable[[Tensor], Tensor] | None = None
    if precond == "auto":
        M_inv = _brinkman_diag_precond(
            lambda u: residual_fn(u, theta_d), u_star_d
        )
    elif callable(precond):
        M_inv = precond

    lambda_, _ = gmres_matfree(
        matvec_Jt, loss_grad.detach(), tol=tol, max_iter=max_iter, precond=M_inv
    )

    # dL/dtheta = -(∂R/∂theta)^T lambda
    _, vjp_fn_theta = torch.func.vjp(lambda th: residual_fn(u_star_d, th), theta_d)
    dL_dtheta = -vjp_fn_theta(lambda_.detach())[0]

    return dL_dtheta
