"""Differentiable Helmholtz filter for manufacturing constraints (C3 claim).

Implements the PDE-based Helmholtz filter:
    (I - r²·∇²) ρ_filtered = ρ

where r is the minimum length scale radius. This smooths the density field
before Heaviside projection and Brinkman penalization, ensuring minimum
feature size in the optimized geometry.

The filter is implemented as a differentiable sparse linear solve; its gradient
flows through the entire shape → filter → Heaviside → Brinkman → NS → objective
chain within a single PyTorch autograd computational graph.

Reference: Lazarov & Sigmund 2016 for Helmholtz filtering in topology optimization.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

from diffcfd.geometry.mesh import CartesianMesh


class HelmholtzFilter:
    """Differentiable Helmholtz PDE filter for density-based topology optimization.

    Solves (I - r²·∇²)ρ_filtered = ρ on a structured Cartesian grid.
    The minimum length scale is controlled by r: features smaller than ~2r
    are filtered out.

    Args:
        mesh: CartesianMesh instance.
        radius: Minimum length scale radius r.
    """

    def __init__(self, mesh: CartesianMesh, radius: float = 0.05) -> None:
        self.mesh = mesh
        self.radius = radius
        self._L = self._build_helmholtz_matrix()

    def _build_helmholtz_matrix(self) -> sp.csr_matrix:
        """Build the Helmholtz operator (I - r²·∇²) as a sparse matrix.

        Uses second-order central differences for the Laplacian.
        Boundary conditions: Neumann (zero gradient) on all walls.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        r = self.radius
        n = nx * ny

        rows, cols, vals = [], [], []

        for j in range(ny):
            for i in range(nx):
                k = j * nx + i
                diag = 1.0  # Identity contribution
                r2_over_dx2 = r**2 / dx**2
                r2_over_dy2 = r**2 / dy**2

                # East neighbor
                if i + 1 < nx:
                    rows.append(k)
                    cols.append(k + 1)
                    vals.append(-r2_over_dx2)
                    diag += r2_over_dx2
                else:
                    # Neumann BC: ghost cell = interior cell → off-diagonal and diagonal cancel
                    diag += 0

                # West neighbor
                if i - 1 >= 0:
                    rows.append(k)
                    cols.append(k - 1)
                    vals.append(-r2_over_dx2)
                    diag += r2_over_dx2

                # North neighbor
                if j + 1 < ny:
                    rows.append(k)
                    cols.append(k + nx)
                    vals.append(-r2_over_dy2)
                    diag += r2_over_dy2

                # South neighbor
                if j - 1 >= 0:
                    rows.append(k)
                    cols.append(k - nx)
                    vals.append(-r2_over_dy2)
                    diag += r2_over_dy2

                rows.append(k)
                cols.append(k)
                vals.append(diag)

        return sp.csr_matrix(
            (
                np.array(vals),
                (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
            ),
            shape=(n, n),
        )

    def apply(self, rho: Tensor) -> Tensor:
        """Apply the Helmholtz filter to a density field.

        Solves (I - r²·∇²)ρ_filtered = ρ using scipy sparse direct solve,
        then returns the result as a differentiable PyTorch tensor.

        The gradient w.r.t. ρ flows through via the implicit function theorem:
        d(ρ_filtered)/d(ρ) = (I - r²·∇²)⁻¹.

        For fully differentiable chains, use apply_differentiable() instead.

        Args:
            rho: Unfiltered density field (ny, nx).

        Returns:
            rho_filtered: Filtered density field (ny, nx).
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        rho_np = rho.detach().cpu().numpy().flatten().astype(np.float64)
        rho_filtered_np = spla.spsolve(self._L, rho_np)
        return torch.tensor(
            rho_filtered_np.reshape(ny, nx),
            dtype=rho.dtype,
            device=rho.device,
        )

    def apply_differentiable(self, rho: Tensor, n_iter: int = 50) -> Tensor:
        """Apply Helmholtz filter with PyTorch-native Jacobi iteration (differentiable).

        All operations are PyTorch tensors with autograd tracking.

        Args:
            rho: Unfiltered density field (ny, nx).
            n_iter: Number of Jacobi iterations.

        Returns:
            rho_filtered: Filtered density field (ny, nx), differentiable w.r.t. rho.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        r = self.radius
        dev = rho.device
        dt = rho.dtype

        r2_dx2 = r**2 / dx**2
        r2_dy2 = r**2 / dy**2
        diag = 1.0 + 2 * r2_dx2 + 2 * r2_dy2

        rho_f = rho.clone()

        for _ in range(n_iter):
            rho_pad = torch.zeros(ny + 2, nx + 2, device=dev, dtype=dt)
            rho_pad[1:-1, 1:-1] = rho_f
            # Neumann BC: ghost = interior
            rho_pad[0, 1:-1] = rho_f[0, :]
            rho_pad[-1, 1:-1] = rho_f[-1, :]
            rho_pad[1:-1, 0] = rho_f[:, 0]
            rho_pad[1:-1, -1] = rho_f[:, -1]
            # Corner ghost cells: Neumann from both directions
            rho_pad[0, 0] = rho_f[0, 0]
            rho_pad[0, -1] = rho_f[0, -1]
            rho_pad[-1, 0] = rho_f[-1, 0]
            rho_pad[-1, -1] = rho_f[-1, -1]

            laplacian = r2_dx2 * (
                rho_pad[1:-1, 2:] + rho_pad[1:-1, :-2] - 2 * rho_f
            ) + r2_dy2 * (rho_pad[2:, 1:-1] + rho_pad[:-2, 1:-1] - 2 * rho_f)

            # (I - r²∇²) rho_f = rho  →  rho_f = (rho + r²∇² rho_f) / diag
            # where diag = 1 + 2*r²/dx² + 2*r²/dy² (interior cells)
            rho_f = (
                rho
                + r2_dx2 * (rho_pad[1:-1, 2:] + rho_pad[1:-1, :-2])
                + r2_dy2 * (rho_pad[2:, 1:-1] + rho_pad[:-2, 1:-1])
            ) / diag

        return rho_f
