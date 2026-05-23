"""Structured Cartesian MAC grid with SDF-based Brinkman immersed boundary.

Brinkman penalization adds (1 - χ(φ)) · u / ε to the momentum equation,
where φ is a signed distance field and χ is a smooth Heaviside of φ.
This keeps geometry gradients well-defined everywhere (no step-function cut-cell).

Reference: Lazarov & Sigmund 2016 for β-continuation Heaviside in topology optimization.
"""

from __future__ import annotations

import torch
from torch import Tensor


class CartesianMesh:
    """Uniform Cartesian grid for MAC (staggered) velocity-pressure layout.

    Cell centers at (i+0.5)*dx, (j+0.5)*dy.
    u_x lives on vertical cell faces; u_y on horizontal faces; p at cell centers.

    Args:
        nx: Number of cells in x direction.
        ny: Number of cells in y direction.
        lx: Domain length in x.
        ly: Domain length in y.
        device: PyTorch device.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float = 1.0,
        ly: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny
        self.device = torch.device(device)

    def cell_centers(self) -> tuple[Tensor, Tensor]:
        """Return (x, y) meshgrid tensors at cell centers, shape (ny, nx)."""
        x = torch.linspace(self.dx / 2, self.lx - self.dx / 2, self.nx, device=self.device)
        y = torch.linspace(self.dy / 2, self.ly - self.dy / 2, self.ny, device=self.device)
        return torch.meshgrid(x, y, indexing="xy")

    def sdf_to_mask(self, sdf: Tensor, epsilon: float = 1e-3, beta: float = 32.0) -> Tensor:
        """Convert a signed distance field to a smooth Brinkman fluid mask χ(φ).

        χ = 1 in fluid (φ > 0), χ = 0 in solid (φ < 0).
        Uses the smooth Heaviside from Lazarov & Sigmund 2016:
            χ(φ) = 0.5 + 0.5 * tanh(β * φ / (2 * max_sdf))
        where β controls interface sharpness (β-continuation: start at 1, increase).

        Args:
            sdf: Signed distance field tensor (ny, nx). Positive inside fluid.
            epsilon: Brinkman penalization coefficient.  Not used here (passed to
                     the momentum equation externally); kept for API consistency.
            beta: Heaviside sharpness.  β=1 → very smooth; β=32 → near step function.

        Returns:
            mask: Fluid volume fraction χ ∈ [0, 1], shape (ny, nx).
        """
        # Normalise by max |sdf| so β is dimensionless and mesh-size independent.
        scale = sdf.abs().max().clamp(min=1e-8)
        return 0.5 + 0.5 * torch.tanh(beta * sdf / (2.0 * scale))
