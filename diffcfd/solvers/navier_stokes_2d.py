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
    ) -> None:
        self.re = reynolds_number
        self.nx, self.ny = grid
        self.device = torch.device(device)
        self.backward = backward
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p
        self.max_iter = max_iter
        self.tol = tol
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run SIMPLE to steady state.

        Returns: (ux, uy, p).
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

        for _ in range(self.max_iter):
            ux_old, uy_old, p_old = ux, uy, p

            # Step 1: implicit momentum solve
            ux_star, a_ux = _solve_u(
                ux, uy, p, nu, dx, dy, self.alpha_u, brinkman, nx, ny,
                inlet_velocity, lid_velocity, case
            )
            uy_star, a_uy = _solve_v(
                ux, uy, p, nu, dx, dy, self.alpha_u, brinkman, nx, ny,
                inlet_velocity, lid_velocity, case
            )

            ux_star, uy_star, _ = self._apply_bcs(
                ux_star, uy_star, p_old, inlet_velocity, lid_velocity, case
            )

            # Step 2: pressure correction with SIMPLE weighting
            div_star = _divergence(ux_star, uy_star, dx, dy)   # (ny, nx)
            L_p, pin = _build_pressure_system(a_ux, a_uy, dx, dy, nx, ny)
            p_prime = _solve_sparse(-div_star.detach().numpy(), L_p, pin, nx, ny, dev)

            # Step 3: velocity correction  Δu = (1/a_P) * ∇p'
            ux_new = ux_star - _vcorr_x(p_prime, a_ux, dx, nx, ny)
            uy_new = uy_star - _vcorr_y(p_prime, a_uy, dy, nx, ny)

            # Step 4: pressure update with relaxation
            # Velocity under-relaxation is already embedded in the momentum system
            # via a_P = a_P0 / alpha_u; do NOT re-relax the corrected velocity.
            p_new = p_old + self.alpha_p * p_prime

            ux_new, uy_new, p_new = self._apply_bcs(
                ux_new, uy_new, p_new, inlet_velocity, lid_velocity, case
            )
            p_new = self.bc.apply_pressure_reference(p_new)

            with torch.no_grad():
                res = _divergence(ux_new, uy_new, dx, dy).abs().max().item()

            ux, uy, p = ux_new, uy_new, p_new
            if res < self.tol:
                break

        return ux, uy, p

    def pressure_drop(self, ux: Tensor, uy: Tensor, p: Tensor) -> Tensor:
        """Scalar ΔP = mean(p[:, 0]) − mean(p[:, -1]). Differentiable."""
        return p[:, 0].mean() - p[:, -1].mean()

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
    a_P_arr = np.ones((ny, nx - 1), dtype=np.float64)  # default 1 for BC rows

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

            a_e = hybrid( F_e, D_e) if ii + 1 < nx - 1 else 0.0
            a_w = hybrid(-F_w, D_w) if ii - 1 >= 0       else 0.0
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
            a_P0 = a_e + a_w + a_n_val + a_s_val + bk_face
            a_P  = a_P0 / alpha_u

            src = -(p_np[j, ii + 1] - p_np[j, ii]) / dx
            src += (1.0 - alpha_u) / alpha_u * a_P0 * u_c
            src += src_n + src_s

            b[k] = src
            a_P_arr[j, ii] = a_P

            rows.append(k); cols.append(k); vals.append(a_P)
            if ii + 1 < nx - 1:
                rows.append(k); cols.append(I(j_int, ii + 1)); vals.append(-a_e)
            if ii - 1 >= 0:
                rows.append(k); cols.append(I(j_int, ii - 1)); vals.append(-a_w)
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
) -> tuple[Tensor, Tensor]:
    """Build and solve the implicit v-momentum equation.

    Returns (uy_star, a_P_field) for interior y-faces (rows 1..ny-1).
    a_P_field shape: (ny-1, nx).
    """
    ux_np = ux.detach().numpy()
    uy_np = uy.detach().numpy()
    p_np  = p.detach().numpy()
    bk_np = brinkman.detach().numpy()

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

            v_n_face = 0.5 * (uy_np[j + 1, i] if j + 1 <= ny else 0.0) + 0.5 * v_c
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

            src = -(p_np[j, i] - p_np[j - 1, i]) / dy
            src += (1.0 - alpha_u) / alpha_u * a_P0 * v_c

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
    pin_idx = 0
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
                c_e = 1.0 / (a_ux_np[j, i] * dx ** 2 + 1e-30)
                rows.append(k); cols.append(k + 1); vals.append(-c_e)
                d += c_e
            # West face: x-face at column i (interior x-face ii=i-1)
            if i - 1 >= 0:
                c_w = 1.0 / (a_ux_np[j, i - 1] * dx ** 2 + 1e-30)
                rows.append(k); cols.append(k - 1); vals.append(-c_w)
                d += c_w
            # North face: y-face at row j+1 (interior y-face jj=j)
            if j + 1 < ny:
                c_n = 1.0 / (a_uy_np[j, i] * dy ** 2 + 1e-30)
                rows.append(k); cols.append(k + nx); vals.append(-c_n)
                d += c_n
            # South face: y-face at row j (interior y-face jj=j-1)
            if j - 1 >= 0:
                c_s = 1.0 / (a_uy_np[j - 1, i] * dy ** 2 + 1e-30)
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


def _vcorr_x(p_prime: Tensor, a_ux: Tensor, dx: float, nx: int, ny: int) -> Tensor:
    """u-velocity correction: Δu = (1/a_P) * (dp'/dx) at interior x-faces."""
    out = torch.zeros(ny, nx + 1, device=p_prime.device, dtype=p_prime.dtype)
    dp = (p_prime[:, 1:nx] - p_prime[:, 0:nx-1]) / dx   # (ny, nx-1)
    out[:, 1:nx] = dp / a_ux.clamp(min=1e-10)
    return out


def _vcorr_y(p_prime: Tensor, a_uy: Tensor, dy: float, nx: int, ny: int) -> Tensor:
    """v-velocity correction: Δv = (1/a_P) * (dp'/dy) at interior y-faces."""
    out = torch.zeros(ny + 1, nx, device=p_prime.device, dtype=p_prime.dtype)
    dp = (p_prime[1:ny, :] - p_prime[0:ny-1, :]) / dy   # (ny-1, nx)
    out[1:ny, :] = dp / a_uy.clamp(min=1e-10)
    return out
