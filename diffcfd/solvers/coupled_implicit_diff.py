"""Implicit differentiation for strongly-coupled fluid-solid systems.

Extends the fixed-point adjoint from implicit_diff.py to block-structured
fluid-solid coupling (FSI and strongly-coupled CHT).  The coupled adjoint
system is:

    [dRf/du_f^T  dRc/du_f^T] [lambda_f]   [dL/du_f]
    [dRf/du_s^T  dRc/du_s^T] [lambda_s] = [dL/du_s]

where Rf = fluid residual, Rc = coupling + solid residual, solved via
matrix-free GMRES on the concatenated state vector z = cat([u_f, u_s]).

Memory O(1) in coupling iterations: only the converged states u_f*, u_s*
are stored; intermediate iterations are discarded.

Reference: Bai et al. 2019 (Deep Equilibrium Models) extended to
multi-physics block-structured adjoint via monolithic GMRES.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from diffcfd.utils.linalg import gmres_matfree


def coupled_fixed_point_gradient(
    fluid_residual_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    solid_residual_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    coupling_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    u_star_fluid: Tensor,
    u_star_solid: Tensor,
    theta: Tensor,
    loss_grad_fluid: Tensor,
    loss_grad_solid: Tensor,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Tensor:
    """Compute dL/dtheta via implicit function theorem for coupled fluid-solid system.

    Assembles the block-structured adjoint as a single GMRES matvec on the
    concatenated state z = cat([u_f, u_s]).  Each matvec computes:

        [J_ff^T  J_fs^T] [v_f]     where J_ff = dRf/du_f, J_sf = dRf/du_s,
        [J_sf^T  J_ss^T] [v_s]           J_fs = dRc/du_f, J_ss = dRc/du_s

    using torch.func.vjp for the full coupled residual.

    Args:
        fluid_residual_fn: R_f(u_f, u_s, theta) -> fluid residual vector.
        solid_residual_fn: R_s(u_f, u_s, theta) -> solid residual vector.
        coupling_fn: Maps (u_fluid, u_solid) -> (solid_bc, fluid_bc).
            Present for API completeness; residuals may use it internally.
        u_star_fluid: Converged fluid state (detached).
        u_star_solid: Converged solid state (detached).
        theta: Design parameters tensor (requires_grad=True).
        loss_grad_fluid: dL/d(u_fluid) at the converged state.
        loss_grad_solid: dL/d(u_solid) at the converged state.
        tol: GMRES convergence tolerance.
        max_iter: Maximum GMRES iterations.

    Returns:
        dL/dtheta tensor, same shape as theta.
    """
    uf_d = u_star_fluid.detach()
    us_d = u_star_solid.detach()
    theta_d = theta.detach().requires_grad_(True)

    def _coupled_residual(z: Tensor, th: Tensor) -> Tensor:
        nf = uf_d.numel()
        uf = z[:nf].reshape(uf_d.shape)
        us = z[nf:].reshape(us_d.shape)
        Rf = fluid_residual_fn(uf, us, th)
        Rs = solid_residual_fn(uf, us, th)
        return torch.cat([Rf.flatten(), Rs.flatten()])

    z_star = torch.cat([uf_d.flatten(), us_d.flatten()])

    def matvec_Jt(v: Tensor) -> Tensor:
        _, vjp_fn = torch.func.vjp(lambda z: _coupled_residual(z, theta_d), z_star)
        return vjp_fn(v)[0]

    rhs = torch.cat([loss_grad_fluid.detach().flatten(), loss_grad_solid.detach().flatten()])

    lambda_sol, _ = gmres_matfree(
        matvec_Jt, rhs, tol=tol, max_iter=max_iter,
    )

    _, vjp_fn_theta = torch.func.vjp(
        lambda th: _coupled_residual(z_star, th), theta_d
    )
    dL_dtheta = -vjp_fn_theta(lambda_sol.detach())[0]

    return dL_dtheta


class FSIResidual:
    """Residual function for fluid-structure interaction.

    Minimal FSI model: 2D channel flow with an elastic top wall.

    Fluid state u_f: velocity field (ny, nx) representing streamwise velocity.
    Solid state u_s: boundary displacement (nx,) -- vertical displacement of
    the elastic top wall at each column.

    The fluid residual is a Poiseuille-like momentum balance:
        R_f = u - f(theta, u_s)
    where f gives the velocity profile that satisfies diffusion + BCs on the
    deformed domain.  This is formulated as a relaxation residual so that
    the fixed-point iteration u = f(theta, u_s) converges.

    The solid residual is a spring law:
        R_s = u_s - elasticity * p_load(u_f)
    where p_load is the pressure load from the fluid on the top wall.

    Both residuals vanish at the coupled fixed point.  The adjoint is exact
    regardless of the forward iteration path.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        elasticity: float = 1e-2,
        max_coupling_iter: int = 100,
        tol: float = 1e-8,
        nu: float = 0.1,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.elasticity = elasticity
        self.max_coupling_iter = max_coupling_iter
        self.tol = tol
        self.nu = nu

    def _channel_profile(self, theta: Tensor, u_solid: Tensor) -> Tensor:
        """Compute parabolic channel velocity profile on deformed domain.

        For a channel of height H = ly + displacement, the Poiseuille profile
        with mean velocity theta is:
            u(y) = 6 * theta * y/H * (1 - y/H)

        The displacement at each column shifts the effective channel height.
        """
        nx, ny = self.nx, self.ny
        dy = self.dy
        disp = u_solid.reshape(nx)

        H = ny * dy + disp  # effective height per column, shape (nx,)
        H = H.clamp(min=dy)  # prevent collapse

        y_frac = torch.arange(1, ny + 1, device=theta.device, dtype=theta.dtype) * dy
        y_norm = (y_frac.unsqueeze(1) - 0.5 * dy) / H.unsqueeze(0)  # (ny, nx)
        y_norm = y_norm.clamp(0, 1)

        profile = 6.0 * theta * y_norm * (1.0 - y_norm)
        return profile

    def fluid_residual(self, u_fluid: Tensor, u_solid: Tensor, theta: Tensor) -> Tensor:
        """Fluid residual: difference between current velocity and equilibrium profile.

        R_f = u_f - channel_profile(theta, u_s)

        This vanishes when the fluid velocity matches the Poiseuille solution
        on the domain defined by the solid displacement.  The coupling to the
        solid state is explicit: the profile shape depends on u_s.

        Args:
            u_fluid: Flattened velocity field (ny * nx,).
            u_solid: Flattened boundary displacement field (nx,).
            theta: Inlet velocity (scalar design parameter).

        Returns:
            Fluid residual vector (ny * nx,).
        """
        u_2d = u_fluid.reshape(self.ny, self.nx)
        target = self._channel_profile(theta, u_solid)
        return (u_2d - target).flatten()

    def solid_residual(self, u_fluid: Tensor, u_solid: Tensor, theta: Tensor) -> Tensor:
        """Elastic boundary residual under fluid pressure load.

        R_s = u_s - elasticity * p_load(u_f)

        where p_load is the velocity-squared pressure at the top wall (Bernoulli).
        At equilibrium the displacement balances the fluid load.

        Args:
            u_fluid: Flattened velocity field (ny * nx,).
            u_solid: Flattened boundary displacement (nx,).
            theta: Inlet velocity (scalar, not directly used by solid residual).

        Returns:
            Solid residual vector (nx,).
        """
        u_2d = u_fluid.reshape(self.ny, self.nx)
        disp = u_solid.reshape(self.nx)

        p_load = 0.5 * u_2d[-1, :] ** 2 * self.dy
        return disp - self.elasticity * p_load

    def coupled_solve(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """Run coupled FSI fixed-point iteration to convergence.

        Alternates between updating the fluid velocity to match the channel
        profile on the current domain, and updating the displacement from
        the fluid load.  The fixed-point is:

            u_f^{k+1} = channel_profile(theta, u_s^k)
            u_s^{k+1} = elasticity * p_load(u_f^{k+1})

        Only the final converged states are returned (O(1) memory).

        Args:
            theta: Inlet velocity (scalar design parameter).

        Returns:
            (u_fluid_star, u_solid_star) -- converged states, detached.
        """
        dev = theta.device
        dt = theta.dtype
        nx = self.nx

        u_solid = torch.zeros(nx, device=dev, dtype=dt)

        for _ in range(self.max_coupling_iter):
            u_solid_old = u_solid.clone()

            u_fluid = self._channel_profile(theta, u_solid)
            p_load = 0.5 * u_fluid[-1, :] ** 2 * self.dy
            u_solid = self.elasticity * p_load

            change = (u_solid - u_solid_old).norm().item()
            if change < self.tol:
                break

        return u_fluid.flatten().detach(), u_solid.detach()
