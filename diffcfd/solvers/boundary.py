"""Boundary condition application for the staggered MAC grid.

MAC grid layout (nx×ny cells, unit square example):
  - u_x:  (ny, nx+1) — x-faces (vertical), including left/right walls
  - u_y:  (ny+1, nx) — y-faces (horizontal), including top/bottom walls
  - p:    (ny, nx)   — cell centres

Boundary conditions are applied directly to the face arrays; they modify
ghost-layer or boundary-face values in-place on a clone so autograd is preserved.

Convention (lid-driven cavity):
  - Left  (x=0):   inlet or wall
  - Right (x=Lx):  outlet or wall
  - Bottom (y=0):  no-slip wall  (u_x=0, u_y=0)
  - Top   (y=Ly):  lid, u_x = U_lid, u_y = 0

For Poiseuille / channel flow:
  - Left:  parabolic inlet profile
  - Right: zero-gradient pressure (dp/dn = 0)
  - Top/Bottom: no-slip
"""

from __future__ import annotations

from torch import Tensor


class BoundaryConditions:
    """Apply boundary conditions to staggered-grid velocity and pressure fields.

    All operations use torch operations so autograd graphs are preserved.

    Args:
        mesh: CartesianMesh instance.
    """

    def __init__(self, mesh) -> None:
        self.mesh = mesh

    # ------------------------------------------------------------------
    # Velocity BCs
    # ------------------------------------------------------------------

    def apply_inlet(self, ux: Tensor, profile: Tensor | float) -> Tensor:
        """Set Dirichlet u_x on the left face (index 0 of x-face array).

        Args:
            ux: x-velocity on x-faces, shape (ny, nx+1).
            profile: Scalar or 1-D tensor (ny,).

        Returns:
            ux clone with column 0 set to profile.
        """
        out = ux.clone()
        if isinstance(profile, Tensor):
            out[:, 0] = profile
        else:
            out[:, 0] = profile
        return out

    def apply_no_slip_walls(self, ux: Tensor, uy: Tensor) -> tuple[Tensor, Tensor]:
        """Zero velocity on bottom (y=0) and top (y=Ly) walls.

        For ux (ny, nx+1): bottom row = 0, top row = 0.
        For uy (ny+1, nx): bottom face = 0, top face = 0.

        Returns:
            (ux, uy) clones with wall faces zeroed.
        """
        ux_out = ux.clone()
        uy_out = uy.clone()
        ux_out[0, :] = 0.0
        ux_out[-1, :] = 0.0
        uy_out[0, :] = 0.0
        uy_out[-1, :] = 0.0
        return ux_out, uy_out

    def apply_lid(self, ux: Tensor, u_lid: float | Tensor) -> Tensor:
        """Set moving lid at top wall: u_x = u_lid on the top face row.

        Args:
            ux: x-velocity (ny, nx+1).
            u_lid: Lid velocity (scalar).

        Returns:
            ux clone with top row set to u_lid.
        """
        out = ux.clone()
        out[-1, :] = u_lid
        return out

    def apply_outlet_velocity(self, ux: Tensor) -> Tensor:
        """Zero-gradient (Neumann) u_x at the right face: copy second-to-last column.

        Args:
            ux: x-velocity (ny, nx+1).

        Returns:
            ux clone with last column = second-to-last.
        """
        out = ux.clone()
        out[:, -1] = out[:, -2]
        return out

    # ------------------------------------------------------------------
    # Pressure BCs
    # ------------------------------------------------------------------

    def apply_outlet_pressure(self, p: Tensor) -> Tensor:
        """Zero-gradient (Neumann) pressure at the right boundary.

        Implemented by copying the last interior column to a virtual ghost cell;
        in practice we simply enforce p[:, -1] = p[:, -2] after each pressure solve.

        Args:
            p: Pressure field (ny, nx).

        Returns:
            p clone with right column set to second-to-last.
        """
        out = p.clone()
        out[:, -1] = out[:, -2]
        return out

    def apply_pressure_reference(self, p: Tensor, ref_i: int = 0, ref_j: int = 0) -> Tensor:
        """Pin pressure at one cell to remove the rank-1 null space of the Poisson equation.

        Shifts the entire pressure field so that p[ref_j, ref_i] = 0.

        Args:
            p: Pressure field (ny, nx).
            ref_i, ref_j: Cell indices of the reference point.

        Returns:
            p shifted so that p[ref_j, ref_i] == 0.
        """
        return p - p[ref_j, ref_i]
