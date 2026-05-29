"""Conjugate heat transfer solver: energy equation coupled with NS.

Steady-state convection-diffusion for temperature on the same MAC grid:
    ∇·(u T) = ∇·(α ∇T) + S_T

where α = k/(ρ cp) is the thermal diffusivity.

Coupling strategy: sequential (solve NS to steady state, then solve energy;
optionally iterate between the two for buoyancy-driven flows).

Two solve paths:
  - solve(): scipy sparse direct solve (fast, not differentiable)
  - solve_differentiable(): PyTorch-native iterative solve (differentiable)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

from diffcfd.geometry.mesh import CartesianMesh


class HeatTransfer2D:
    """Steady-state energy equation on a structured MAC grid.

    Args:
        mesh: CartesianMesh instance.
        alpha: Thermal diffusivity k/(ρ·cp) [m²/s].
        k: Thermal conductivity.
        rho: Density.
        cp: Specific heat.
    """

    def __init__(
        self,
        mesh: CartesianMesh,
        alpha: float | None = None,
        k: float = 1.0,
        rho: float = 1.0,
        cp: float = 1.0,
    ) -> None:
        self.mesh = mesh
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = k / (rho * cp)
        self.k = k

    def solve(
        self,
        ux: Tensor,
        uy: Tensor,
        T_bc: dict | None = None,
        tol: float = 1e-6,
        max_iter: int = 500,
    ) -> Tensor:
        """Solve steady-state energy equation via scipy sparse direct solve.

        Fast but not differentiable through autograd.

        Args:
            ux: x-velocity on x-faces, shape (ny, nx+1).
            uy: y-velocity on y-faces, shape (ny+1, nx).
            T_bc: Dict with thermal BCs. Keys:
                - 'top': (type, value) — 'dirichlet' or 'neumann'
                - 'bottom': (type, value)
                - 'left': (type, value)
                - 'right': (type, value)
            tol: Convergence tolerance on max residual.
            max_iter: Maximum iterations (for iterative solve; currently direct).

        Returns:
            T: Temperature field (ny, nx).
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        alpha_th = self.alpha

        if T_bc is None:
            T_bc = {
                "bottom": ("dirichlet", 0.0),
                "top": ("dirichlet", 1.0),
                "left": ("neumann", 0.0),
                "right": ("neumann", 0.0),
            }

        ux_np = ux.detach().cpu().numpy()
        uy_np = uy.detach().cpu().numpy()

        n = nx * ny
        rows, cols, vals = [], [], []
        b = np.zeros(n, dtype=np.float64)

        def I(j, i):
            return j * nx + i

        for j in range(ny):
            for i in range(nx):
                k = I(j, i)

                # Cell-center velocity (average of faces)
                u_c = 0.5 * (ux_np[j, i] + ux_np[j, i + 1])
                v_c = 0.5 * (uy_np[j, i] + uy_np[j + 1, i])

                # Face velocities
                u_e = 0.5 * (ux_np[j, i] + ux_np[j, i + 1])  # east face x-vel
                u_w = u_c  # west face
                v_n = 0.5 * (uy_np[j, i] + uy_np[j + 1, i])  # north face y-vel
                v_s = v_c  # south face

                F_e = u_e * dy  # mass flux east
                F_w = u_w * dy
                F_n = v_n * dx
                F_s = v_s * dx

                # Diffusion coefficients — Neumann walls get D=0 (zero flux),
                # Dirichlet walls get double D (half distance to wall).
                bc_r = T_bc.get("right", ("neumann", 0.0))
                bc_l = T_bc.get("left", ("neumann", 0.0))
                bc_t = T_bc.get("top", ("dirichlet", 1.0))
                bc_b = T_bc.get("bottom", ("dirichlet", 0.0))
                if i + 1 < nx:
                    D_e = alpha_th * dy / dx
                elif bc_r[0] == "dirichlet":
                    D_e = alpha_th * dy / (dx / 2)
                else:
                    D_e = 0.0
                if i - 1 >= 0:
                    D_w = alpha_th * dy / dx
                elif bc_l[0] == "dirichlet":
                    D_w = alpha_th * dy / (dx / 2)
                else:
                    D_w = 0.0
                if j + 1 < ny:
                    D_n = alpha_th * dx / dy
                elif bc_t[0] == "dirichlet":
                    D_n = alpha_th * dx / (dy / 2)
                else:
                    D_n = 0.0
                if j - 1 >= 0:
                    D_s = alpha_th * dx / dy
                elif bc_b[0] == "dirichlet":
                    D_s = alpha_th * dx / (dy / 2)
                else:
                    D_s = 0.0

                def hybrid(F, D):
                    return max(abs(D) - 0.5 * abs(F), 0.0) + max(0.0, -F)

                a_e = hybrid(F_e, D_e)
                a_w = hybrid(-F_w, D_w)
                a_n = hybrid(F_n, D_n)
                a_s = hybrid(-F_s, D_s)

                src = 0.0
                diag = 0.0

                # East neighbor
                if i + 1 < nx:
                    rows.append(k); cols.append(I(j, i + 1)); vals.append(-a_e)
                    diag += a_e
                else:
                    bc_type, bc_val = T_bc.get("right", ("neumann", 0.0))
                    if bc_type == "dirichlet":
                        src += a_e * bc_val
                        diag += a_e

                # West neighbor
                if i - 1 >= 0:
                    rows.append(k); cols.append(I(j, i - 1)); vals.append(-a_w)
                    diag += a_w
                else:
                    bc_type, bc_val = T_bc.get("left", ("neumann", 0.0))
                    if bc_type == "dirichlet":
                        src += a_w * bc_val
                        diag += a_w

                # North neighbor
                if j + 1 < ny:
                    rows.append(k); cols.append(I(j + 1, i)); vals.append(-a_n)
                    diag += a_n
                else:
                    bc_type, bc_val = T_bc.get("top", ("dirichlet", 1.0))
                    if bc_type == "dirichlet":
                        src += a_n * bc_val
                        diag += a_n

                # South neighbor
                if j - 1 >= 0:
                    rows.append(k); cols.append(I(j - 1, i)); vals.append(-a_s)
                    diag += a_s
                else:
                    bc_type, bc_val = T_bc.get("bottom", ("dirichlet", 0.0))
                    if bc_type == "dirichlet":
                        src += a_s * bc_val
                        diag += a_s

                rows.append(k); cols.append(k); vals.append(diag)
                b[k] = src

        A = sp.csr_matrix(
            (np.array(vals), (np.array(rows, dtype=np.int32),
                              np.array(cols, dtype=np.int32))),
            shape=(n, n),
        )
        T_flat = spla.spsolve(A, b)
        return torch.tensor(T_flat.reshape(ny, nx), dtype=torch.float32,
                            device=ux.device)

    def solve_differentiable(
        self,
        ux: Tensor,
        uy: Tensor,
        T_bc: dict | None = None,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> Tensor:
        """Solve energy equation with PyTorch-native iterative sweep (differentiable).

        Uses point Gauss-Seidel-like sweeps where each cell update uses the
        hybrid upwind scheme with full autograd tracking. The iteration is
        unrolled so gradients flow through all sweeps.

        Args:
            ux: x-velocity on x-faces, shape (ny, nx+1).
            uy: y-velocity on y-faces, shape (ny+1, nx).
            T_bc: Dict with thermal BCs (same format as solve()).
            tol: Convergence tolerance.
            max_iter: Maximum sweeps.

        Returns:
            T: Temperature field (ny, nx), differentiable w.r.t. ux, uy.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        alpha_th = self.alpha
        dev = ux.device

        if T_bc is None:
            T_bc = {
                "bottom": ("dirichlet", 0.0),
                "top": ("dirichlet", 1.0),
                "left": ("neumann", 0.0),
                "right": ("neumann", 0.0),
            }

        T = torch.zeros(ny, nx, device=dev, dtype=ux.dtype)

        # Initialize with linear interpolation between Dirichlet BCs
        for wall, (bc_type, bc_val) in T_bc.items():
            if bc_type == "dirichlet":
                if wall == "bottom":
                    # Don't set T[0] = bc_val — Dirichlet is at the wall (y=0),
                    # T[0] is at cell center (y=dy/2). Initialize with small value.
                    pass
                elif wall == "top":
                    pass
                elif wall == "left":
                    pass
                elif wall == "right":
                    pass

        # Simple initialization for vertical conduction
        bc_bottom = T_bc.get("bottom", ("neumann", 0.0))
        bc_top = T_bc.get("top", ("neumann", 1.0))
        if bc_bottom[0] == "dirichlet" and bc_top[0] == "dirichlet":
            y_frac = torch.linspace(0, 1, ny + 2, device=dev, dtype=ux.dtype)
            for j in range(ny):
                T[j, :] = bc_bottom[1] + (bc_top[1] - bc_bottom[1]) * y_frac[j + 1]

        # Pre-compute face velocities on staggered MAC grid:
        # ux is on x-faces: shape (ny, nx+1). East face of cell (j,i) = ux[j, i+1], west = ux[j, i].
        # uy is on y-faces: shape (ny+1, nx). North face of cell (j,i) = uy[j+1, i], south = uy[j, i].
        u_east = ux[:, 1:]    # (ny, nx) velocity at east face of each cell
        u_west = ux[:, :-1]   # (ny, nx) velocity at west face of each cell
        v_north = uy[1:, :]   # (ny, nx) velocity at north face of each cell
        v_south = uy[:-1, :]  # (ny, nx) velocity at south face of each cell

        # Boundary condition masks (constant across iterations)
        bc_bottom = T_bc.get("bottom", ("neumann", 0.0))
        bc_top = T_bc.get("top", ("neumann", 1.0))
        bc_left = T_bc.get("left", ("neumann", 0.0))
        bc_right = T_bc.get("right", ("neumann", 0.0))

        for _it in range(max_iter):
            T_old = T.clone()

            # Padded T with boundary ghost cells for BC handling
            T_pad = torch.zeros(ny + 2, nx + 2, device=dev, dtype=ux.dtype)
            T_pad[1:-1, 1:-1] = T

            # Ghost cells: Dirichlet → mirror the wall value through the cell center
            # so that (T_ghost + T_first_cell)/2 = T_wall, i.e. T_ghost = 2*T_wall - T_cell
            # Neumann (zero gradient) → T_ghost = T_cell

            if bc_bottom[0] == "dirichlet":
                T_pad[0, 1:-1] = 2 * bc_bottom[1] - T[0, :]
            else:
                T_pad[0, 1:-1] = T[0, :]

            if bc_top[0] == "dirichlet":
                T_pad[-1, 1:-1] = 2 * bc_top[1] - T[-1, :]
            else:
                T_pad[-1, 1:-1] = T[-1, :]

            if bc_left[0] == "dirichlet":
                T_pad[1:-1, 0] = 2 * bc_left[1] - T[:, 0]
            else:
                T_pad[1:-1, 0] = T[:, 0]

            if bc_right[0] == "dirichlet":
                T_pad[1:-1, -1] = 2 * bc_right[1] - T[:, -1]
            else:
                T_pad[1:-1, -1] = T[:, -1]

            T_e = T_pad[1:-1, 2:]
            T_w = T_pad[1:-1, :-2]
            T_n = T_pad[2:, 1:-1]
            T_s = T_pad[:-2, 1:-1]

            # Face mass fluxes (dimensional) — each face uses its own staggered velocity
            F_e = u_east * dy
            F_w = u_west * dy
            F_n = v_north * dx
            F_s = v_south * dx

            # Diffusion coefficients: interior faces use full spacing;
            # faces at Dirichlet walls use half spacing (wall to cell center).
            # Use broadcastable scalars for uniform grid; multiply by directional
            # wall-correction masks where needed.
            D_base_x = alpha_th * dy / dx
            D_base_y = alpha_th * dx / dy
            # Uniform diffusion coefficients — the ghost cell Dirichlet formula
            # (T_ghost = 2*T_wall - T_cell) already accounts for the half-distance.
            D_e = torch.full((ny, nx), D_base_x, device=dev, dtype=ux.dtype)
            D_w = torch.full((ny, nx), D_base_x, device=dev, dtype=ux.dtype)
            D_n = torch.full((ny, nx), D_base_y, device=dev, dtype=ux.dtype)
            D_s = torch.full((ny, nx), D_base_y, device=dev, dtype=ux.dtype)

            # Hybrid upwind scheme
            a_e = torch.clamp(D_e - 0.5 * F_e.abs(), min=0.0) + torch.clamp(-F_e, min=0.0)
            a_w = torch.clamp(D_w - 0.5 * F_w.abs(), min=0.0) + torch.clamp(F_w, min=0.0)
            a_n = torch.clamp(D_n - 0.5 * F_n.abs(), min=0.0) + torch.clamp(-F_n, min=0.0)
            a_s = torch.clamp(D_s - 0.5 * F_s.abs(), min=0.0) + torch.clamp(F_s, min=0.0)

            a_P = a_e + a_w + a_n + a_s + 1e-30

            # Jacobi update: T_new = (a_e*T_e + a_w*T_w + a_n*T_n + a_s*T_s) / a_P
            T = (a_e * T_e + a_w * T_w + a_n * T_n + a_s * T_s) / a_P

            # Convergence check
            with torch.no_grad():
                res = (T - T_old).abs().max().item()
                if res < tol:
                    break

        return T

    def nusselt_number(
        self,
        T: Tensor,
        T_hot: float,
        T_cold: float,
        L: float,
        wall: str = "bottom",
        T_wall: float | None = None,
    ) -> Tensor:
        """Compute average Nusselt number on a wall (differentiable).

        Nu = h·L/k where h = q_w / (T_hot - T_cold).
        q_w = -k * dT/dn at the wall.

        Args:
            T: Temperature field (ny, nx).
            T_hot: Hot wall temperature (reference).
            T_cold: Cold (reference) temperature.
            L: Characteristic length.
            wall: Which wall to compute Nu on ('top', 'bottom', 'left', 'right').
            T_wall: Temperature at the specified wall. If None, uses T_hot for
                    bottom/left walls and T_cold for top/right walls.

        Returns:
            Nu_avg: Average Nusselt number (scalar tensor, differentiable).
        """
        dT = T_hot - T_cold
        if abs(dT) < 1e-12:
            return torch.tensor(0.0, dtype=T.dtype, device=T.device)

        dx, dy = self.mesh.dx, self.mesh.dy

        # Determine wall temperature
        if T_wall is None:
            T_wall = T_hot if wall in ("bottom", "left") else T_cold

        if wall == "bottom":
            grad_T = (T[0, :] - T_wall) / (dy / 2)
        elif wall == "top":
            grad_T = (T[-1, :] - T_wall) / (dy / 2)
        elif wall == "left":
            grad_T = (T[:, 0] - T_wall) / (dx / 2)
        elif wall == "right":
            grad_T = (T[:, -1] - T_wall) / (dx / 2)
        else:
            raise ValueError(f"Unknown wall: {wall}")

        Nu_avg = L * grad_T.abs().mean() / abs(dT)
        return Nu_avg


def coupled_steady_solve(
    ns_solver,
    heat_solver: HeatTransfer2D,
    T_bc: dict | None = None,
    buoyancy: bool = False,
    Ra: float = 0.0,
    sdf: Tensor | None = None,
    inlet_velocity: float = 1.0,
    lid_velocity: float = 0.0,
    case: str = "channel",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Solve coupled NS + energy to steady state.

    For non-buoyancy cases: sequential (NS first, then energy).
    For buoyancy: iterate between NS (with Boussinesq term) and energy.

    Args:
        ns_solver: NavierStokes2D instance.
        heat_solver: HeatTransfer2D instance.
        T_bc: Thermal boundary conditions.
        buoyancy: Whether to include Boussinesq buoyancy coupling.
        Ra: Rayleigh number (for buoyancy).
        sdf: Optional signed distance field.
        inlet_velocity: Inlet velocity (channel flow).
        lid_velocity: Lid velocity (cavity flow).
        case: Flow case ('channel' or 'cavity').

    Returns:
        (ux, uy, p, T) velocity, pressure, temperature fields.
    """
    ux, uy, p = ns_solver.solve_steady(
        sdf=sdf, inlet_velocity=inlet_velocity,
        lid_velocity=lid_velocity, case=case,
    )

    T = heat_solver.solve(ux, uy, T_bc=T_bc)

    if buoyancy and Ra > 0:
        # Boussinesq approximation: F_b = Ra * alpha * (T - T_ref) in y-direction
        T_ref = 0.5
        for _ in range(3):
            buoyancy_src = Ra * heat_solver.alpha * (T - T_ref)
            ux, uy, p = ns_solver.solve_steady(
                sdf=sdf, inlet_velocity=inlet_velocity,
                lid_velocity=lid_velocity, case=case,
                buoyancy_src=buoyancy_src,
            )
            T = heat_solver.solve(ux, uy, T_bc=T_bc)

    return ux, uy, p, T
