"""Boundary condition application for the staggered MAC grid.

All BC parameters are differentiable: inlet velocity profile shape, wall temperature,
etc. are tensors that can carry gradients.
"""

from __future__ import annotations

import torch
from torch import Tensor


class BoundaryConditions:
    """Apply boundary conditions to velocity and pressure fields.

    Args:
        mesh: CartesianMesh describing grid dimensions and cell size.
    """

    def __init__(self, mesh) -> None:
        self.mesh = mesh

    def apply_inlet(self, u: Tensor, profile: Tensor | float) -> Tensor:
        """Set Dirichlet u_x on the left (x=0) face.

        Args:
            u: Velocity field (2, ny, nx).
            profile: Scalar or 1-D tensor of length ny.

        Returns:
            u with inlet column overwritten (in-place clone).
        """
        raise NotImplementedError

    def apply_outlet(self, p: Tensor) -> Tensor:
        """Set zero-gradient pressure (Neumann) on the right (x=Lx) face."""
        raise NotImplementedError

    def apply_no_slip(self, u: Tensor) -> Tensor:
        """Set u=0 on solid walls (top and bottom faces, or Brinkman-penalized cells)."""
        raise NotImplementedError

    def apply_symmetry(self, u: Tensor) -> Tensor:
        """Set normal velocity component to zero on symmetry planes."""
        raise NotImplementedError
