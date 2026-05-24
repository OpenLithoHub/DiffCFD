"""2D incompressible Navier-Stokes solver on staggered MAC grid.

v0.05: Steady-state SIMPLE with implicit diffusion + Picard convection linearisation.
       autograd flows through the full iteration stack (unrolled).
v0.1:  replace autograd unrolling with matrix-free GMRES implicit diff (C1 claim).

MAC grid layout for an (ny × nx) cell domain:
  ux : (ny,   nx+1)  x-velocity on vertical faces
  uy : (ny+1, nx  )  y-velocity on horizontal faces
  p  : (ny,   nx  )  pressure at cell centres

SIMPLE algorithm — steady-state form (no time derivative):
  1. Momentum solve: solve A_u · u* = b_u(u_prev, p)  and  A_v · v* = b_v
     Coefficients:
       a_P (diagonal) = Σ a_nb  (from convection + diffusion)
       With under-relaxation: a_P' = a_P / α_u;  b' += (1-α_u)/α_u * a_P * u_prev
     Convection: hybrid upwind (Patankar 1980, Pe-dependent switching)
     Diffusion:  fully implicit central
  2. Pressure correction: solve  A_p · p' = b_p = (∇·u*)
     A_p built from 1/a_P coefficients at faces (proper SIMPLE weighting)
  3. Velocity correction: u = u* - (1/a_P) * ∇p'
  4. Pressure update: p = p + α_p * p'
  5. Apply BCs, check ‖∇·u‖∞, repeat.

This formulation finds the true steady-state solution directly; no time-step
stability condition on diffusion (implicit diffusion allows dt=∞ effectively).
Memory for unrolled autograd: O(N·K) where K = number of outer SIMPLE iterations.

References:
  Patankar (1980) "Numerical Heat Transfer and Fluid Flow" — SIMPLE chapter
  van Doormaal & Raithby (1984) SIMPLEC variant
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.solvers.boundary import BoundaryConditions
from diffcfd.solvers.implicit_diff import fixed_point_gradient


class _SteadyStateNS(torch.autograd.Function):
    """Custom autograd Function for implicit differentiation through SIMPLE.

    Forward pass: run SIMPLE (non-differentiable, under-relaxed).
    Backward pass: solve adjoint via matrix-free GMRES on unrelaxed residual.
    Gradients computed w.r.t. both theta (scalar BC parameter) and sdf (geometry).
    """

    @staticmethod
    def forward(ctx, theta, solver, case, sdf):
        theta_val = theta.detach()
        sdf_val = sdf.detach() if sdf is not None else None
        if case == "channel":
            ux, uy, p, a_ux, a_uy = solver._run_simple(
                sdf_val, inlet_velocity=theta_val, lid_velocity=0.0, case=case,
                return_aP=True
            )
        else:
            ux, uy, p, a_ux, a_uy = solver._run_simple(
                sdf_val, inlet_velocity=0.0, lid_velocity=theta_val, case=case,
                return_aP=True
            )
        u_star = solver._pack_interior(ux, uy).detach()
        # Save both theta and sdf for backward (both need requires_grad tracking)
        ctx.save_for_backward(u_star, theta, sdf_val if sdf_val is not None else torch.tensor([]))
        ctx.solver = solver
        ctx.case = case
        ctx.p = p.detach()
        ctx.has_sdf = sdf is not None
        ctx.a_ux = a_ux.detach()
        ctx.a_uy = a_uy.detach()
        return ux.detach(), uy.detach(), p.detach()

    @staticmethod
    def backward(ctx, dL_dux, dL_duy, dL_dp):
        u_star, theta, sdf_saved = ctx.saved_tensors
        solver = ctx.solver
        p = ctx.p
        case = ctx.case
        nx, ny = solver.nx, solver.ny
        has_sdf = ctx.has_sdf

        # Build Brinkman field differentiably (for backward)
        if has_sdf:
            bk_eps = 1e-3
            sdf_for_grad = sdf_saved.detach().requires_grad_(True)
            chi = solver.mesh.sdf_to_mask(sdf_for_grad, epsilon=bk_eps)
            brinkman = (1.0 - chi) / bk_eps
        else:
            brinkman = None
            sdf_for_grad = None

        p_star = p.flatten()
        z_star = torch.cat([u_star, p_star])
        n_u = u_star.shape[0]

        loss_u = torch.cat([
            dL_dux[1:-1, 1:-1].flatten(),
            dL_duy[1:-1, :].flatten(),
        ])
        loss_z = torch.cat([loss_u, dL_dp.flatten()])

        theta_d = theta.detach().requires_grad_(True)

        def combined_res(z, th, sd):
            bk = None
            if sd is not None:
                ch = solver.mesh.sdf_to_mask(sd, epsilon=bk_eps)
                bk = (1.0 - ch) / bk_eps
            return solver._combined_residual(z, th, n_u, case, bk)

        def matvec_Jt(v):
            _, vjp_fn = torch.func.vjp(
                lambda z: combined_res(z, theta_d, sdf_for_grad), z_star.detach()
            )
            return vjp_fn(v)[0]

        from diffcfd.utils.linalg import gmres_matfree
        lambda_sol, _ = gmres_matfree(
            matvec_Jt, loss_z.detach(), tol=1e-5, max_iter=2000, restart=200
        )

        # Gradient w.r.t. theta
        _, vjp_th = torch.func.vjp(
            lambda th: combined_res(z_star.detach(), th, sdf_for_grad), theta_d
        )
        dL_dtheta = -vjp_th(lambda_sol.detach())[0]

        # Gradient w.r.t. sdf
        dL_dsdf = None
        if has_sdf:
            _, vjp_sdf = torch.func.vjp(
                lambda sd: combined_res(z_star.detach(), theta_d, sd), sdf_for_grad
            )
            dL_dsdf = -vjp_sdf(lambda_sol.detach())[0]

        return dL_dtheta, None, None, dL_dsdf


class NavierStokes2D:
    """Steady-state 2D incompressible NS via SIMPLE.

    Args:
        reynolds_number: Re = U·L / ν.
        grid: (nx, ny) cell counts.
        device: PyTorch device.
        backward: "unrolled" (v0.05) or "implicit_diff" (v0.1).
        alpha_u: Velocity under-relaxation factor (0.3–0.8 typical).
        alpha_p: Pressure under-relaxation factor (0.1–0.4 typical).
        max_iter: Maximum SIMPLE outer iterations.
        tol: Convergence on max |∇·u| (dimensional).
        lx, ly: Domain dimensions.
    """

    def __init__(
        self,
        reynolds_number: float,
        grid: tuple[int, int],
        device: str = "cpu",
        backward: str = "unrolled",
        alpha_u: float = 0.5,
        alpha_p: float = 0.1,
        max_iter: int = 2000,
        tol: float = 1e-5,
        lx: float = 1.0,
        ly: float = 1.0,
        anderson_depth: int = 0,
    ) -> None:
        self.re = reynolds_number
        self.nx, self.ny = grid
        self.device = torch.device(device)
        self.backward = backward
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p
        self.max_iter = max_iter
        self.tol = tol
        self.anderson_depth = anderson_depth
        self.mesh = CartesianMesh(self.nx, self.ny, lx=lx, ly=ly, device=device)
        self.bc = BoundaryConditions(self.mesh)
        self._nu = 1.0 / reynolds_number

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_steady(
        self,
        sdf: Tensor | None = None,
        inlet_velocity: float | Tensor = 1.0,
        lid_velocity: float | Tensor = 0.0,
        case: str = "channel",
        buoyancy_src: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run SIMPLE to steady state.

        When backward="implicit_diff", wraps the SIMPLE forward pass with a
        custom autograd Function that uses matrix-free GMRES for the backward.

        Returns: (ux, uy, p).
        """
        if self.backward == "implicit_diff":
            # theta is the differentiable scalar design parameter
            if case == "channel":
                theta = (
                    inlet_velocity
                    if isinstance(inlet_velocity, Tensor)
                    else torch.tensor(float(inlet_velocity), dtype=torch.float32,
                                      device=self.device)
                )
            else:
                theta = (
                    lid_velocity
                    if isinstance(lid_velocity, Tensor)
                    else torch.tensor(float(lid_velocity), dtype=torch.float32,
                                      device=self.device)
                )
            return _SteadyStateNS.apply(theta, self, case, sdf)
        else:
            return self._run_simple(
                sdf, inlet_velocity, lid_velocity, case,
                buoyancy_src=buoyancy_src,
            )

    def _run_simple(
        self,
        sdf: Tensor | None = None,
        inlet_velocity: float | Tensor = 1.0,
        lid_velocity: float | Tensor = 0.0,
        case: str = "channel",
        return_aP: bool = False,
        buoyancy_src: Tensor | None = None,
    ) -> tuple:
        """Core SIMPLE iteration loop.

        Args:
            return_aP: If True, return (ux, uy, p, a_ux, a_uy); else (ux, uy, p).
        """
        dx = self.mesh.dx
        dy = self.mesh.dy
        nx = self.nx
        ny = self.ny
        nu = self._nu
        dev = self.device

        bk_eps = 1e-3
        if sdf is not None:
            chi = self.mesh.sdf_to_mask(sdf, epsilon=bk_eps)
            brinkman = (1.0 - chi) / bk_eps
        else:
            brinkman = torch.zeros(ny, nx, device=dev)

        # Initialise
        ux = torch.zeros(ny, nx + 1, device=dev)
        uy = torch.zeros(ny + 1, nx, device=dev)
        p  = torch.zeros(ny, nx, device=dev)
        ux, uy, p = self._apply_bcs(ux, uy, p, inlet_velocity, lid_velocity, case)

        # Anderson acceleration state
        m = self.anderson_depth
        hist_x = []  # previous iterates (flattened ux+uy+p)
        hist_g = []  # previous residuals (g_k = x_k - x_{k-1})

        for it in range(self.max_iter):
            ux_old, uy_old, p_old = ux, uy, p

            # Step 1: implicit momentum solve
            ux_star, a_ux = _solve_u(
                ux, uy, p, nu, dx, dy, self.alpha_u, brinkman, nx, ny,
                inlet_velocity, lid_velocity, case
            )
            uy_star, a_uy = _solve_v(
                ux, uy, p, nu, dx, dy, self.alpha_u, brinkman, nx, ny,
                inlet_velocity, lid_velocity, case, buoyancy_src=buoyancy_src
            )

            ux_star, uy_star, _ = self._apply_bcs(
                ux_star, uy_star, p_old, inlet_velocity, lid_velocity, case
            )

            # Step 2: pressure correction with SIMPLE weighting
            div_star = _divergence(ux_star, uy_star, dx, dy)   # (ny, nx)
            L_p, pin = _build_pressure_system(a_ux, a_uy, dx, dy, nx, ny)
            p_prime = _solve_sparse(-div_star.detach().numpy(), L_p, pin, nx, ny, dev)

            # Step 3: velocity correction  Δu = (dy/dx) * ∇p' / a_P
            ux_new = ux_star - _vcorr_x(p_prime, a_ux, dx, dy, nx, ny)
            uy_new = uy_star - _vcorr_y(p_prime, a_uy, dx, dy, nx, ny)

            # Step 4: pressure update with relaxation
            # Velocity under-relaxation is already embedded in the momentum system
            # via a_P = a_P0 / alpha_u; do NOT re-relax the corrected velocity.
            p_new = p_old + self.alpha_p * p_prime

            ux_new, uy_new, p_new = self._apply_bcs(
                ux_new, uy_new, p_new, inlet_velocity, lid_velocity, case
            )
            # Pin pressure gauge at outlet-center cell (same cell as linear-system pin)
            p_new = p_new - p_new[ny // 2, nx - 1]

            with torch.no_grad():
                res = _divergence(ux_new, uy_new, dx, dy).abs().max().item()

            ux, uy, p = ux_new, uy_new, p_new
            if res < self.tol:
                break

            # Anderson acceleration (forward-only, does not affect implicit diff)
            if m > 0:
                x_k = torch.cat([ux.flatten(), uy.flatten(), p.flatten()]).detach()
                if it > 0:
                    g_k = x_k - hist_x[-1]
                    hist_g.append(g_k)
                    hist_x.append(x_k)
                    if len(hist_g) > m:
                        hist_g.pop(0)
                        hist_x.pop(0)
                    if len(hist_g) >= 2:
                        G = torch.stack(list(hist_g), dim=1)  # (N, m_used)
                        try:
                            coeffs = torch.linalg.lstsq(G, g_k).solution  # (m_used,)
                            if torch.isfinite(coeffs).all():
                                X_hist = torch.stack(list(hist_x), dim=1)
                                x_new = x_k + (X_hist - x_k.unsqueeze(1)) @ coeffs
                                n_ux = ny * (nx + 1)
                                n_uy = (ny + 1) * nx
                                ux = x_new[:n_ux].reshape(ny, nx + 1)
                                uy = x_new[n_ux:n_ux + n_uy].reshape(ny + 1, nx)
                                p = x_new[n_ux + n_uy:].reshape(ny, nx)
                        except RuntimeError:
                            pass  # lstsq singular — skip this iteration
                else:
                    hist_x.append(x_k)

        if return_aP:
            return ux, uy, p, a_ux, a_uy
        return ux, uy, p

    def pressure_drop(self, ux: Tensor, uy: Tensor, p: Tensor) -> Tensor:
        """Scalar ΔP = mean(p[:, 0]) − mean(p[:, -1]). Differentiable."""
        return p[:, 0].mean() - p[:, -1].mean()

    def _pack_interior(self, ux: Tensor, uy: Tensor) -> Tensor:
        """Flatten interior velocity components into a single vector.

        Interior ux: rows j=1..ny-2, cols 1..nx-1  → (ny-2)*(nx-1) elements
        Interior uy: rows j=1..ny-1, all cols       → (ny-1)*nx elements
        """
        return torch.cat([ux[1:-1, 1:-1].flatten(), uy[1:-1, :].flatten()])

    def _unpack_interior(self, u_flat: Tensor) -> tuple[Tensor, Tensor]:
        """Inverse of _pack_interior. Returns (ux_int, uy_int) without BCs."""
        nx, ny = self.nx, self.ny
        n_ux = (ny - 2) * (nx - 1)
        ux_int = u_flat[:n_ux].reshape(ny - 2, nx - 1)
        uy_int = u_flat[n_ux:].reshape(ny - 1, nx)
        return ux_int, uy_int

    def _residual_flat(
        self,
        u_flat: Tensor,
        theta: Tensor,
        p: Tensor,
        case: str,
        brinkman: Tensor | None = None,
    ) -> Tensor:
        """Pure-PyTorch unrelaxed NS momentum residual at state u_flat.

        No relaxation, no scipy — compatible with torch.func.vjp for implicit diff.

        Args:
            u_flat: Packed interior velocity state from _pack_interior.
            theta: Design parameter (scalar inlet_velocity or lid_velocity).
            p: Converged pressure field (ny, nx), treated as constant.
            case: "channel" or "cavity".
            brinkman: Optional Brinkman penalization (ny, nx); zeros if None.

        Returns:
            Residual vector, same shape as u_flat.
        """
        nx, ny = self.nx, self.ny
        dx, dy, nu = self.mesh.dx, self.mesh.dy, self._nu
        dev = u_flat.device
        dt = u_flat.dtype

        if brinkman is None:
            brinkman = torch.zeros(ny, nx, device=dev, dtype=dt)

        ux_int, uy_int = self._unpack_interior(u_flat)

        # Build full fields with BCs (no autograd through BC enforcement itself)
        ux = torch.zeros(ny, nx + 1, device=dev, dtype=dt)
        uy = torch.zeros(ny + 1, nx, device=dev, dtype=dt)

        ux[1:-1, 1:-1] = ux_int   # interior x-faces
        uy[1:-1, :] = uy_int      # interior y-faces

        # Boundary values
        if case == "channel":
            ux[:, 0] = theta            # inlet: all rows set to theta
            ux[:, -1] = ux[:, -2]      # outlet zero-gradient (Neumann)
            # top/bottom walls: ux rows 0 and -1 already 0 (zeros_like)
        elif case == "cavity":
            ux[-1, :] = theta           # lid: top row = theta (lid_velocity)
            # top/bottom: ux rows 0 and -1 already 0

        # Wall BCs for uy (all cases): uy[0,:]=0 and uy[-1,:]=0 already 0
        if case == "cavity":
            uy[:, 0] = 0.0             # left/right walls for uy
            uy[:, -1] = 0.0

        # ------------------------------------------------------------------
        # u-momentum residual for interior ux faces: j=1..ny-2, ii=0..nx-2
        # physical col = ii+1, so ux[:, 1:-1] are the interior faces
        # ------------------------------------------------------------------
        D = nu / dx   # diffusion coefficient (same in x)
        Dn = nu / dy

        # Interior ux block: shape (ny-2, nx-1), rows j=1..ny-2, cols ii=0..nx-2
        # u_c = ux[1:-1, 1:-1]  (already = ux_int but through ux tensor for grad)
        u_c = ux[1:-1, 1:-1]    # (ny-2, nx-1)

        # East face velocity (for Pe number): average of u_c and east neighbor
        # East neighbor: for col ii → ux[j, ii+2] = ux[1:-1, 2:]  (shape ny-2, nx-2)
        # At the right wall (ii=nx-2): east neighbor is the outlet face ux[:, nx]
        ux_e_nb = torch.cat([ux[1:-1, 2:-1], ux[1:-1, -1:]], dim=1)  # (ny-2, nx-1)
        u_e_face = 0.5 * (ux_e_nb + u_c)

        # West face velocity: ux[1:-1, 0:-2+1] = ux[1:-1, :-1]  includes col 0 (inlet)
        # West of col ii=0 is inlet face ux[j,0]=theta; west of col ii→ux[j,ii]
        u_w_face = 0.5 * (ux[1:-1, :-2] + u_c)   # (ny-2, nx-1): ux[j,ii] + u_c

        # v at x-face: bilinear average of 4 surrounding y-face values
        # uy has shape (ny+1, nx); interior ux rows are j=1..ny-2
        # SW = uy[j, ii], SE = uy[j, ii+1], NW = uy[j+1, ii], NE = uy[j+1, ii+1]
        # Array: j=1..ny-2 → uy[1:-2, :]; j+1=2..ny-1 → uy[2:-1, :]
        v_sw = uy[1:-2, :-1]    # uy[j,   ii]     (ny-2, nx-1)
        v_se = uy[1:-2, 1:]     # uy[j,   ii+1]   (ny-2, nx-1)
        v_nw = uy[2:-1, :-1]    # uy[j+1, ii]     (ny-2, nx-1)
        v_ne = uy[2:-1, 1:]     # uy[j+1, ii+1]   (ny-2, nx-1)
        v_c = 0.25 * (v_sw + v_se + v_nw + v_ne)

        F_e = u_e_face
        F_w = u_w_face
        F_n = v_c
        F_s = v_c

        def h(F, D_val):
            D_t = torch.as_tensor(D_val, dtype=F.dtype, device=F.device)
            return torch.clamp(D_t - 0.5 * torch.abs(F), min=0.0) + torch.clamp(-F, min=0.0)

        a_e_full = h(F_e, D)    # (ny-2, nx-1)
        a_w_full = h(-F_w, D)   # (ny-2, nx-1)
        a_n_full = h(F_n, Dn)   # (ny-2, nx-1)
        a_s_full = h(-F_s, Dn)  # (ny-2, nx-1)

        # Neumann outlet: zero east coefficient for rightmost interior col (ii=nx-2)
        a_e_full = a_e_full.clone()
        a_e_full[:, -1] = 0.0
        # East: col ii=nx-2 has no east interior neighbor → zero contribution in matrix
        mask_e = torch.ones(nx - 1, dtype=torch.bool, device=dev)
        mask_e[-1] = False   # rightmost interior col: no east interior neighbor
        mask_w = torch.ones(nx - 1, dtype=torch.bool, device=dev)
        mask_w[0] = False    # leftmost interior col: no west interior neighbor (inlet BC source)

        # Brinkman face value
        bk_face = 0.5 * (brinkman[1:-1, :-1] + brinkman[1:-1, 1:])   # (ny-2, nx-1)

        # North/south boundary handling:
        # Row j=ny-2 (j_int=ny-3): north neighbor is top wall (row ny-1) → BC source, no matrix entry
        # Row j=1 (j_int=0):       south neighbor is bottom wall (row 0) → BC source (0), no matrix entry
        # Interior rows: a_n and a_s go into neighbor terms

        # Boundary wall velocities
        u_north_wall = theta if case == "cavity" else torch.zeros(1, device=dev, dtype=dt).squeeze()
        u_south_wall = torch.zeros(1, device=dev, dtype=dt).squeeze()

        # North boundary contribution (row j=ny-2, i.e. last row of u_c):
        # a_n_val * u_north_wall → source
        a_n_bc = a_n_full[-1:, :]     # (1, nx-1): BC row
        src_n_bc = (a_n_bc * u_north_wall).unsqueeze(0)  # → added to RHS for last row

        # South boundary contribution (row j=1, i.e. first row of u_c):
        a_s_bc = a_s_full[:1, :]      # (1, nx-1): BC row
        src_s_bc = (a_s_bc * u_south_wall).unsqueeze(0)

        # West boundary (inlet) contribution (col ii=0):
        # a_w * inlet_velocity → source for leftmost interior col
        # inlet_velocity = theta for channel, 0 for cavity
        if case == "channel":
            inlet_val = theta
        else:
            inlet_val = torch.zeros(1, device=dev, dtype=dt).squeeze()
        a_w_bc = a_w_full[:, :1]   # (ny-2, 1)
        src_w_bc = a_w_bc * inlet_val  # (ny-2, 1)

        # Build a_P0 (sum of all neighbor coefficients, incl. BC ones):
        a_P0_u = a_e_full + a_w_full + a_n_full + a_s_full + bk_face   # (ny-2, nx-1)

        # Residual = a_P0 * u_c  -  sum(a_nb * u_nb)  -  dp/dx  -  bc_sources
        # Neighbor contributions (interior only, mask out boundary cols):
        nb_e = torch.zeros_like(u_c)
        nb_e[:, :-1] = a_e_full[:, :-1] * u_c[:, 1:]   # east: u[j, ii+2] = u_c shifted

        nb_w = torch.zeros_like(u_c)
        nb_w[:, 1:] = a_w_full[:, 1:] * u_c[:, :-1]    # west: u[j, ii] = u_c shifted

        nb_n = torch.zeros_like(u_c)
        nb_n[:-1, :] = a_n_full[:-1, :] * u_c[1:, :]   # north: u[j+1, ...]

        nb_s = torch.zeros_like(u_c)
        nb_s[1:, :] = a_s_full[1:, :] * u_c[:-1, :]    # south: u[j-1, ...]

        # Pressure gradient (using converged p, treated as constant)
        # Momentum source = (p_w - p_e) * dy / dx (area-weighted, divided by dx)
        dp_dx = (p[1:-1, 1:] - p[1:-1, :-1]) / dx   # (ny-2, nx-1): p[j,ii+1]-p[j,ii]

        R_ux = a_P0_u * u_c - nb_e - nb_w - nb_n - nb_s - (-dp_dx * dy)
        # Subtract BC source contributions (already removed from matrix side)
        # For north BC row: a_n_bc * u_north_wall was added to RHS in solve; in residual it's on LHS side
        R_ux[-1:, :] -= a_n_bc * u_north_wall
        R_ux[:1, :]  -= a_s_bc * u_south_wall
        R_ux[:, :1]  -= src_w_bc    # inlet BC: a_w contribution

        # ------------------------------------------------------------------
        # v-momentum residual for interior uy faces: j=1..ny-1 (all nx cols)
        # ------------------------------------------------------------------
        Dx = nu / dx
        Dy = nu / dy

        v_c = uy[1:-1, :]    # (ny-1, nx): interior uy = uy_int

        # North face: uy[j+1, i] — for j=ny-1 (top interior face), j+1=ny is the wall
        uy_n_nb = torch.cat([uy[2:-1, :], torch.zeros(1, nx, device=dev, dtype=dt)], dim=0)  # (ny-1, nx)
        v_n_face = 0.5 * (uy_n_nb + v_c)  # but top wall is 0, already in zeros

        uy_s_nb = torch.cat([torch.zeros(1, nx, device=dev, dtype=dt), uy[1:-2, :]], dim=0)  # (ny-1, nx)
        v_s_face = 0.5 * (uy_s_nb + v_c)

        # u at y-face: bilinear average of surrounding x-face values
        # u_sw[j,i] = ux[j, i], u_se = ux[j, i+1], u_nw = ux[j+1, i], u_ne = ux[j+1, i+1]
        # uy interior rows 1..ny-1 → ux rows j-1 and j = rows 0..ny-2 and 1..ny-1
        # In array coords: row jj (0-indexed interior) → physical j = jj+1
        # ux rows for j-1 = 0..ny-2 → ux[:-1, :], for j = 1..ny-1 → ux[1:, :]
        u_sw = ux[:-1, :-1]    # ux[j-1, i]   (ny-1, nx)
        u_se = ux[:-1, 1:]     # ux[j-1, i+1] (ny-1, nx)
        u_nw = ux[1:, :-1]     # ux[j,   i]   (ny-1, nx)
        u_ne = ux[1:, 1:]      # ux[j,   i+1] (ny-1, nx)
        u_c_v = 0.25 * (u_sw + u_se + u_nw + u_ne)

        # v-momentum fluxes
        # North: top interior face (jj=ny-2) has no north neighbor (wall uy=0)
        F_vn = torch.where(
            torch.arange(ny - 1, device=dev).unsqueeze(1) < ny - 2,
            v_n_face, torch.zeros_like(v_n_face)
        )
        F_vs = v_s_face
        # East/West: lateral walls have uy=0 (for cavity/channel both)
        F_ve_all = u_c_v
        F_vw_all = u_c_v

        D_vn = torch.where(
            torch.arange(ny - 1, device=dev).unsqueeze(1) < ny - 2,
            torch.full((1,), Dy, device=dev, dtype=dt),
            torch.zeros((1,), device=dev, dtype=dt)
        )
        D_vs = torch.full((ny - 1, nx), Dy, device=dev, dtype=dt)

        a_vn_full = torch.where(
            torch.arange(ny - 1, device=dev).unsqueeze(1) < ny - 2,
            h(F_vn, Dy), torch.zeros(1, device=dev, dtype=dt)
        )
        a_vs_full = torch.where(
            torch.arange(ny - 1, device=dev).unsqueeze(1) > 0,
            h(-F_vs, Dy), torch.zeros(1, device=dev, dtype=dt)
        )
        a_ve_full = torch.where(
            torch.arange(nx, device=dev).unsqueeze(0) < nx - 1,
            h(F_ve_all, Dx), torch.zeros(1, device=dev, dtype=dt)
        )
        a_vw_full = torch.where(
            torch.arange(nx, device=dev).unsqueeze(0) > 0,
            h(-F_vw_all, Dx), torch.zeros(1, device=dev, dtype=dt)
        )

        bk_v = 0.5 * (brinkman[:-1, :] + brinkman[1:, :])   # (ny-1, nx): average over j-1, j

        a_P0_v = a_vn_full + a_vs_full + a_ve_full + a_vw_full + bk_v

        # v-neighbor contributions
        nb_vn = torch.zeros_like(v_c)
        nb_vn[:-1, :] = a_vn_full[:-1, :] * v_c[1:, :]

        nb_vs = torch.zeros_like(v_c)
        nb_vs[1:, :] = a_vs_full[1:, :] * v_c[:-1, :]

        nb_ve = torch.zeros_like(v_c)
        nb_ve[:, :-1] = a_ve_full[:, :-1] * v_c[:, 1:]

        nb_vw = torch.zeros_like(v_c)
        nb_vw[:, 1:] = a_vw_full[:, 1:] * v_c[:, :-1]

        # Pressure gradient for v: (p[j,i] - p[j-1,i]) * dx / dy (area-weighted)
        dp_dy = (p[1:, :] - p[:-1, :]) / dy    # (ny-1, nx): p[j,i] - p[j-1,i]

        R_uy = a_P0_v * v_c - nb_vn - nb_vs - nb_ve - nb_vw - (-dp_dy * dx)

        # Subtract BC source terms for v-momentum:
        # North boundary (jj=ny-2, j=ny-1): wall uy=0 → a_vn * 0 = 0 (no effect, but include for generality)
        # South boundary (jj=0, j=1): wall uy=0 → a_vs * 0 = 0
        # East/West boundaries (cavity): wall uy=0 → a_ve/a_vw * 0 = 0
        # For non-zero wall velocities these would contribute; with no-slip walls they are zero.
        v_north_wall = torch.zeros(1, device=dev, dtype=dt).squeeze()
        v_south_wall = torch.zeros(1, device=dev, dtype=dt).squeeze()
        R_uy[-1:, :] -= a_vn_full[-1:, :] * v_north_wall
        R_uy[:1, :]  -= a_vs_full[:1, :] * v_south_wall

        return torch.cat([R_ux.flatten(), R_uy.flatten()])

    def _combined_residual(
        self,
        z: "Tensor",
        theta: "Tensor",
        n_u: int,
        case: str,
        brinkman: "Tensor | None" = None,
    ) -> "Tensor":
        """Combined (u, p) residual for implicit differentiation.

        z = cat([u_flat, p_flat]) where u_flat is packed interior velocity and
        p_flat = p.flatten().  Returns cat([R_u(u,p,theta), div(u)]) with
        a pressure gauge pin at the outlet-center cell.

        Used in _SteadyStateNS.backward to handle pressure-velocity coupling
        in the fixed-point implicit differentiation.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        dev = z.device
        dt = z.dtype

        u_flat = z[:n_u]
        p_flat = z[n_u:]
        p_2d = p_flat.reshape(ny, nx)

        R_u = self._residual_flat(u_flat, theta, p_2d, case, brinkman)

        # Reconstruct full ux/uy to compute divergence
        ux_int, uy_int = self._unpack_interior(u_flat)
        ux = torch.zeros(ny, nx + 1, device=dev, dtype=dt)
        uy = torch.zeros(ny + 1, nx, device=dev, dtype=dt)
        ux[1:-1, 1:-1] = ux_int
        uy[1:-1, :] = uy_int

        if case == "channel":
            ux[1:-1, 0] = theta   # inlet only for interior rows (walls stay 0)
            ux[:, -1] = ux[:, -2]  # Neumann outlet
        elif case == "cavity":
            ux[-1, :] = theta
            uy[:, 0] = 0.0
            uy[:, -1] = 0.0

        # Divergence: (ux[j,i+1]-ux[j,i])/dx + (uy[j+1,i]-uy[j,i])/dy
        div = (ux[:, 1:] - ux[:, :-1]) / dx + (uy[1:, :] - uy[:-1, :]) / dy  # (ny, nx)
        R_p = div.flatten()

        # Pin pressure gauge at outlet-center cell to remove null space
        pin_cell = (ny // 2) * nx + (nx - 1)
        R_p = R_p.clone()
        R_p[pin_cell] = p_flat[pin_cell]

        return torch.cat([R_u, R_p])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_bcs(self, ux, uy, p, inlet_velocity, lid_velocity, case):
        if case == "channel":
            ux = self.bc.apply_inlet(ux, inlet_velocity)
            ux, uy = self.bc.apply_no_slip_walls(ux, uy)
            ux = self.bc.apply_outlet_velocity(ux)
        elif case == "cavity":
            ux, uy = self.bc.apply_no_slip_walls(ux, uy)
            ux = self.bc.apply_lid(ux, lid_velocity)
            uy_c = uy.clone()
            uy_c[:, 0] = 0.0
            uy_c[:, -1] = 0.0
            uy = uy_c
        return ux, uy, p


# ------------------------------------------------------------------
# Solver kernels
# ------------------------------------------------------------------

def _solve_u(
    ux: Tensor, uy: Tensor, p: Tensor,
    nu: float, dx: float, dy: float, alpha_u: float,
    brinkman: Tensor, nx: int, ny: int,
    inlet_velocity, lid_velocity, case: str,
) -> tuple[Tensor, Tensor]:
    """Build and solve the implicit u-momentum equation.

    Solves only interior y-rows (j = 1..ny-2); boundary rows are Dirichlet BCs
    and are treated as source terms for adjacent rows.  Hybrid upwind
    (Patankar 1980) for convection, central for diffusion.

    Returns (u_star, a_P_field) where a_P_field shape is (ny, nx-1).
    Boundary rows of a_P_field are filled with 1.0 (unused by pressure system).
    """
    ux_np = ux.detach().numpy()
    uy_np = uy.detach().numpy()
    p_np  = p.detach().numpy()
    bk_np = brinkman.detach().numpy()

    # Interior y-rows: j = 1..ny-2  (boundary rows j=0 and j=ny-1 are BCs)
    ny_int = ny - 2          # number of interior rows to solve
    nu_int = ny_int * (nx - 1)

    rows, cols, vals = [], [], []
    b = np.zeros(nu_int, dtype=np.float64)
    a_P_arr = np.full((ny, nx - 1), 1e30, dtype=np.float64)  # large → 1/a_P≈0 for BC rows

    def I(j_int, i_int):
        # j_int in [0, ny-3] → physical j = j_int + 1
        return j_int * (nx - 1) + i_int

    # Boundary u-values for source terms
    u_south_wall = 0.0           # bottom no-slip (all cases)
    if case == "cavity":
        u_north_wall = float(lid_velocity) if isinstance(lid_velocity, (int, float)) \
                       else lid_velocity.item()
    else:
        u_north_wall = 0.0       # top no-slip for channel

    def hybrid(F, D):
        return max(abs(D) - 0.5 * abs(F), 0.0) + max(0.0, -F)

    for j_int in range(ny_int):   # j_int in [0, ny-3], physical j = j_int + 1
        j = j_int + 1
        for ii in range(nx - 1):   # ii = interior index, physical col = ii+1
            k = I(j_int, ii)
            u_c = ux_np[j, ii + 1]

            u_e_face = 0.5 * (ux_np[j, ii + 2] if ii + 2 <= nx else ux_np[j, nx]) + 0.5 * u_c
            u_w_face = 0.5 * ux_np[j, ii] + 0.5 * u_c

            v_c = 0.25 * (uy_np[j, ii] + uy_np[j, ii + 1] + uy_np[j + 1, ii] + uy_np[j + 1, ii + 1])

            F_e = u_e_face
            F_w = u_w_face
            F_n = v_c
            F_s = v_c

            D_e = nu / dx
            D_w = nu / dx
            D_n = nu / dy
            D_s = nu / dy

            # East: if at the right boundary, Neumann outlet → zero coefficient, no source
            if ii + 1 >= nx - 1:
                a_e_val = 0.0; a_e_matrix = 0.0; src_e = 0.0
            else:
                a_e_val = hybrid(F_e, D_e); a_e_matrix = a_e_val; src_e = 0.0

            # West: if at the left boundary, Dirichlet inlet BC source
            a_w_val = hybrid(-F_w, D_w)
            if ii == 0:
                u_inlet_val = float(inlet_velocity) if isinstance(inlet_velocity, (int, float)) \
                              else inlet_velocity.item()
                src_w = a_w_val * (u_inlet_val if case == "channel" else 0.0)
                a_w_matrix = 0.0
            else:
                src_w = 0.0
                a_w_matrix = a_w_val

            # North: if j+1 is the lid row (j == ny-2) → Dirichlet BC source, no matrix entry
            # Otherwise: interior row → matrix entry
            if j + 1 == ny - 1:    # next row is boundary (top wall)
                a_n_val = hybrid(F_n, D_n)
                src_n = a_n_val * u_north_wall  # Dirichlet source
                a_n_matrix = 0.0
            elif j_int + 1 < ny_int:
                a_n_val = hybrid(F_n, D_n)
                src_n = 0.0
                a_n_matrix = a_n_val
            else:
                a_n_val = 0.0; src_n = 0.0; a_n_matrix = 0.0

            if j - 1 == 0:         # prev row is boundary (bottom wall)
                a_s_val = hybrid(-F_s, D_s)
                src_s = a_s_val * u_south_wall
                a_s_matrix = 0.0
            elif j_int - 1 >= 0:
                a_s_val = hybrid(-F_s, D_s)
                src_s = 0.0
                a_s_matrix = a_s_val
            else:
                a_s_val = 0.0; src_s = 0.0; a_s_matrix = 0.0

            bk_face = 0.5 * (bk_np[j, ii] + bk_np[j, ii + 1])
            a_P0 = a_e_val + a_w_val + a_n_val + a_s_val + bk_face
            a_P  = a_P0 / alpha_u

            src = -(p_np[j, ii + 1] - p_np[j, ii]) * dy / dx
            src += (1.0 - alpha_u) / alpha_u * a_P0 * u_c
            src += src_n + src_s + src_w + src_e

            b[k] = src
            a_P_arr[j, ii] = a_P

            rows.append(k); cols.append(k); vals.append(a_P)
            if a_e_matrix > 0.0:
                rows.append(k); cols.append(I(j_int, ii + 1)); vals.append(-a_e_matrix)
            if a_w_matrix > 0.0:
                rows.append(k); cols.append(I(j_int, ii - 1)); vals.append(-a_w_matrix)
            if a_n_matrix > 0.0:
                rows.append(k); cols.append(I(j_int + 1, ii)); vals.append(-a_n_matrix)
            if a_s_matrix > 0.0:
                rows.append(k); cols.append(I(j_int - 1, ii)); vals.append(-a_s_matrix)

    A = sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(nu_int, nu_int)
    )
    sol = spla.spsolve(A, b).reshape(ny_int, nx - 1)

    ux_star = ux.clone()
    ux_star[1:ny-1, 1:nx] = torch.tensor(sol, dtype=torch.float32, device=ux.device)
    a_P_t = torch.tensor(a_P_arr, dtype=torch.float32, device=ux.device)
    return ux_star, a_P_t


def _solve_v(
    ux: Tensor, uy: Tensor, p: Tensor,
    nu: float, dx: float, dy: float, alpha_u: float,
    brinkman: Tensor, nx: int, ny: int,
    inlet_velocity, lid_velocity, case: str,
    buoyancy_src: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build and solve the implicit v-momentum equation.

    Returns (uy_star, a_P_field) for interior y-faces (rows 1..ny-1).
    a_P_field shape: (ny-1, nx).
    """
    ux_np = ux.detach().numpy()
    uy_np = uy.detach().numpy()
    p_np  = p.detach().numpy()
    bk_np = brinkman.detach().numpy()
    buoy_np = buoyancy_src.detach().numpy() if buoyancy_src is not None else None

    nv_int = (ny - 1) * nx

    rows, cols, vals = [], [], []
    b = np.zeros(nv_int, dtype=np.float64)
    a_P_arr = np.zeros(nv_int, dtype=np.float64)

    def I(jj, i):   # jj in [0, ny-2], physical row j = jj+1
        return jj * nx + i

    for jj in range(ny - 1):   # physical j = jj+1
        j = jj + 1
        for i in range(nx):
            k = I(jj, i)
            v_c = uy_np[j, i]

            v_n_face = 0.5 * (uy_np[j + 1, i] if j + 1 <= ny - 1 else 0.0) + 0.5 * v_c
            v_s_face = 0.5 * uy_np[j - 1, i] + 0.5 * v_c

            # u at this y-face (bilinear average)
            u_sw = ux_np[j - 1, i    ]
            u_se = ux_np[j - 1, i + 1]
            u_nw = ux_np[j,     i    ]
            u_ne = ux_np[j,     i + 1]
            u_c = 0.25 * (u_sw + u_se + u_nw + u_ne)

            F_n = v_n_face if j + 1 <= ny - 1 else 0.0
            F_s = v_s_face
            F_e = u_c if i < nx - 1 else 0.0
            F_w = u_c if i > 0      else 0.0

            D_n = nu / dy if j + 1 < ny else 0.0
            D_s = nu / dy
            D_e = nu / dx if i < nx - 1 else 0.0
            D_w = nu / dx if i > 0      else 0.0

            def hybrid(F, D):
                return max(abs(D) - 0.5 * abs(F), 0.0) + max(0.0, -F)

            a_n = hybrid( F_n, D_n) if jj + 1 < ny - 1 else 0.0
            a_s = hybrid(-F_s, D_s) if jj - 1 >= 0      else 0.0
            a_e = hybrid( F_e, D_e) if i + 1 < nx       else 0.0
            a_w = hybrid(-F_w, D_w) if i - 1 >= 0       else 0.0

            bk_face = 0.5 * (bk_np[j - 1, i] + bk_np[min(j, ny-1), i])

            a_P0 = a_n + a_s + a_e + a_w + bk_face
            a_P  = a_P0 / alpha_u

            src = -(p_np[j, i] - p_np[j - 1, i]) * dx / dy
            src += (1.0 - alpha_u) / alpha_u * a_P0 * v_c
            if buoy_np is not None:
                src += buoy_np[j, i] * dx * dy

            b[k] = src
            a_P_arr[k] = a_P

            rows.append(k); cols.append(k); vals.append(a_P)
            if jj + 1 < ny - 1:
                rows.append(k); cols.append(I(jj + 1, i)); vals.append(-a_n)
            if jj - 1 >= 0:
                rows.append(k); cols.append(I(jj - 1, i)); vals.append(-a_s)
            if i + 1 < nx:
                rows.append(k); cols.append(I(jj, i + 1)); vals.append(-a_e)
            if i - 1 >= 0:
                rows.append(k); cols.append(I(jj, i - 1)); vals.append(-a_w)

    A = sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(nv_int, nv_int)
    )
    sol = spla.spsolve(A, b).reshape(ny - 1, nx)

    uy_star = uy.clone()
    uy_star[1:ny, :] = torch.tensor(sol, dtype=torch.float32, device=uy.device)
    a_P_t = torch.tensor(a_P_arr.reshape(ny - 1, nx), dtype=torch.float32, device=uy.device)
    return uy_star, a_P_t


def _build_pressure_system(
    a_ux: Tensor, a_uy: Tensor, dx: float, dy: float, nx: int, ny: int
):
    """Build SIMPLE pressure correction Laplacian with a_P-weighted face coefficients.

    SIMPLE: pressure correction equation is
        -∂/∂x[(1/a_P_u) ∂p'/∂x] - ∂/∂y[(1/a_P_v) ∂p'/∂y] = ∇·u*

    The coefficient at interior x-face (j, i+1) is 1/(a_P_u[j, i]).
    At boundary faces (x=0, x=Lx, y=0, y=Ly), coefficient = 0 (Neumann → no flux).
    """
    a_ux_np = a_ux.detach().numpy()   # (ny, nx-1)
    a_uy_np = a_uy.detach().numpy()   # (ny-1, nx)

    n = nx * ny
    # Pin at a cell in the top-right region (away from inlet/corners)
    # to remove the rank-1 null space
    pin_idx = (ny // 2) * nx + (nx - 1)
    rows, cols, vals = [], [], []

    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if k == pin_idx:
                rows.append(k); cols.append(k); vals.append(1.0)
                continue

            d = 0.0

            # East face: x-face at column i+1 (interior x-face ii=i, physical col i+1)
            if i + 1 < nx:
                c_e = dy / (a_ux_np[j, i] * dx ** 2 + 1e-30)
                rows.append(k); cols.append(k + 1); vals.append(-c_e)
                d += c_e
            # West face: x-face at column i (interior x-face ii=i-1)
            if i - 1 >= 0:
                c_w = dy / (a_ux_np[j, i - 1] * dx ** 2 + 1e-30)
                rows.append(k); cols.append(k - 1); vals.append(-c_w)
                d += c_w
            # North face: y-face at row j+1 (interior y-face jj=j)
            if j + 1 < ny:
                c_n = dx / (a_uy_np[j, i] * dy ** 2 + 1e-30)
                rows.append(k); cols.append(k + nx); vals.append(-c_n)
                d += c_n
            # South face: y-face at row j (interior y-face jj=j-1)
            if j - 1 >= 0:
                c_s = dx / (a_uy_np[j - 1, i] * dy ** 2 + 1e-30)
                rows.append(k); cols.append(k - nx); vals.append(-c_s)
                d += c_s

            rows.append(k); cols.append(k); vals.append(d)

    L = sp.csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n)
    )
    return L, pin_idx


def _solve_sparse(
    rhs_np: np.ndarray, L, pin_idx: int, nx: int, ny: int, device: torch.device
) -> Tensor:
    b = rhs_np.flatten().astype(np.float64)
    b[pin_idx] = 0.0
    x = spla.spsolve(L, b)
    return torch.tensor(x.reshape(ny, nx), dtype=torch.float32, device=device)


def _divergence(ux: Tensor, uy: Tensor, dx: float, dy: float) -> Tensor:
    return (ux[:, 1:] - ux[:, :-1]) / dx + (uy[1:, :] - uy[:-1, :]) / dy


def _vcorr_x(p_prime: Tensor, a_ux: Tensor, dx: float, dy: float, nx: int, ny: int) -> Tensor:
    """u-velocity correction: Δu = (dy/dx) * (dp'/dx) / a_P at interior x-faces."""
    out = torch.zeros(ny, nx + 1, device=p_prime.device, dtype=p_prime.dtype)
    dp = (p_prime[:, 1:nx] - p_prime[:, 0:nx-1]) / dx   # (ny, nx-1)
    out[:, 1:nx] = dp * dy / a_ux.clamp(min=1e-10)
    return out


def _vcorr_y(p_prime: Tensor, a_uy: Tensor, dx: float, dy: float, nx: int, ny: int) -> Tensor:
    """v-velocity correction: Δv = (dx/dy) * (dp'/dy) / a_P at interior y-faces."""
    out = torch.zeros(ny + 1, nx, device=p_prime.device, dtype=p_prime.dtype)
    # Only correct interior y-faces (rows 1..ny-1)
    dp = (p_prime[1:ny, :] - p_prime[0:ny-1, :]) / dy   # (ny-1, nx)
    out[1:ny, :] = dp * dx / a_uy.clamp(min=1e-10)
    return out
