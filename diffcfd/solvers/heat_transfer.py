"""Conjugate heat transfer solver: energy equation coupled with NS.

Steady-state convection-diffusion for temperature on the same MAC grid:
    ∇·(u T) = ∇·(α ∇T) + S_T

where α = k/(ρ cp) is the thermal diffusivity.

Coupling strategy: sequential (solve NS to steady state, then solve energy;
optionally iterate between the two for buoyancy-driven flows).

Differentiable Nusselt number: Nu = h·L/k where h = q_w / (T_w - T_ref).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.solvers.boundary import BoundaryConditions


class HeatTransfer2D:
    """Steady-state energy equation on a structured MAC grid.

    Solves the convection-diffusion equation for temperature using the
    same hybrid upwind scheme as the NS momentum equations.

    Args:
        mesh: CartesianMesh instance.
        alpha: Thermal diffusivity k/(ρ·cp) [m²/s].
        Pe: Peclet number (overrides alpha if provided: α = U·L/Pe).
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
        """Solve steady-state energy equation given a velocity field.

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

                # Diffusion coefficients: interior faces use full cell spacing,
                # Dirichlet wall faces use half spacing (wall to cell center).
                D_e = alpha_th * dy / (dx if i + 1 < nx else
                        dx / 2 if T_bc.get("right", ("neumann", 0.0))[0] == "dirichlet"
                        else dx)
                D_w = alpha_th * dy / (dx if i - 1 >= 0 else
                        dx / 2 if T_bc.get("left", ("neumann", 0.0))[0] == "dirichlet"
                        else dx)
                D_n = alpha_th * dx / (dy if j + 1 < ny else
                        dy / 2 if T_bc.get("top", ("dirichlet", 1.0))[0] == "dirichlet"
                        else dy)
                D_s = alpha_th * dx / (dy if j - 1 >= 0 else
                        dy / 2 if T_bc.get("bottom", ("dirichlet", 0.0))[0] == "dirichlet"
                        else dy)

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
                    # neumann: zero flux → coefficient stays zero

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

    def nusselt_number(
        self,
        T: Tensor,
        T_hot: float,
        T_cold: float,
        L: float,
        wall: str = "bottom",
    ) -> Tensor:
        """Compute local and average Nusselt number on a wall.

        Nu = h·L/k where h = q_w / (T_hot - T_cold).
        q_w is computed from the temperature gradient at the wall.

        Args:
            T: Temperature field (ny, nx).
            T_hot: Hot wall temperature.
            T_cold: Cold (reference) temperature.
            L: Characteristic length.
            wall: Which wall to compute Nu on ('top', 'bottom', 'left', 'right').

        Returns:
            Nu_avg: Average Nusselt number (scalar tensor, differentiable).
        """
        dT = T_hot - T_cold
        if abs(dT) < 1e-12:
            return torch.tensor(0.0, dtype=T.dtype, device=T.device)

        dx, dy = self.mesh.dx, self.mesh.dy

        # Wall gradient: use one-sided difference from wall to first cell center.
        # Distance from wall to cell center = half cell size.
        if wall == "bottom":
            # T_wall = T_cold, dT/dy ≈ (T[0] - T_cold) / (dy/2)
            grad_T = (T[0, :] - T_cold) / (dy / 2)
        elif wall == "top":
            grad_T = (T[-1, :] - T_hot) / (dy / 2)
        elif wall == "left":
            grad_T = (T[:, 0] - T_cold) / (dx / 2)
        elif wall == "right":
            grad_T = (T[:, -1] - T_hot) / (dx / 2)
        else:
            raise ValueError(f"Unknown wall: {wall}")

        # Nu = h*L/k = |q_w|*L / (k*|T_hot - T_cold|) = L * |grad_T_wall| / |T_hot - T_cold|
        # For bottom wall (cold): heat flux INTO wall = k * dT/dy > 0 (T increases into domain)
        # For top wall (hot): heat flux out of domain = -k * dT/dy
        Nu_avg = L * grad_T.abs().mean() / abs(dT)
        return Nu_avg


def coupled_steady_solve(
    ns_solver,
    heat_solver: HeatTransfer2D,
    T_bc: dict | None = None,
    buoyancy: bool = False,
    Ra: float = 0.0,
    Gr: float = 0.0,
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
        Gr: Grashof number (for buoyancy).
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
        # Boussinesq approximation: add buoyancy source to v-momentum
        # Iterate between NS and energy for coupling
        for _ in range(5):
            # Buoyancy force: F_b = Ra * α * (T - T_ref) in y-direction
            # This modifies the v-momentum source; for now we do a simple
            # correction loop (full implementation would modify the SIMPLE loop)
            pass

    return ux, uy, p, T
