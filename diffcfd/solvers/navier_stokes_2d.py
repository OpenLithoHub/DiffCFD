"""2D incompressible Navier-Stokes solver on staggered MAC grid.

v0.05 target: unrolled SIMPLE with full autograd through all iterations.
v0.1 target: replace autograd unrolling with matrix-free GMRES implicit diff (C1 claim).
"""

from __future__ import annotations

import torch
from torch import Tensor


class NavierStokes2D:
    """Steady-state 2D incompressible NS via SIMPLE pressure-velocity coupling.

    Args:
        reynolds_number: Re = U·L / ν (reference velocity and length set per case).
        grid: (nx, ny) cell counts for the Cartesian MAC grid.
        device: PyTorch device string, e.g. "cpu" or "cuda".
        backward: "unrolled" (v0.05, O(N·K) memory) or "implicit_diff" (v0.1, O(N)).
    """

    def __init__(
        self,
        reynolds_number: float,
        grid: tuple[int, int],
        device: str = "cpu",
        backward: str = "unrolled",
    ) -> None:
        self.re = reynolds_number
        self.nx, self.ny = grid
        self.device = torch.device(device)
        self.backward = backward

    def solve_steady(
        self,
        sdf: Tensor | None = None,
        inlet_velocity: float | Tensor = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Run SIMPLE to convergence and return (u, p) on the MAC grid.

        Args:
            sdf: Signed distance field (ny, nx) for Brinkman immersed boundary.
                 None → pure fluid domain.
            inlet_velocity: Scalar or 1-D tensor (ny,) inlet u-velocity profile.

        Returns:
            u: Velocity field tensor (2, ny, nx) — [u_x, u_y] components.
            p: Pressure field tensor (ny, nx).
        """
        raise NotImplementedError("Implement in v0.05 — unrolled SIMPLE forward pass.")

    def pressure_drop(self, u: Tensor, p: Tensor) -> Tensor:
        """Return scalar pressure drop ΔP = p_inlet_mean - p_outlet_mean.

        Differentiable: gradient flows back through p via autograd.
        """
        raise NotImplementedError("Implement alongside solve_steady in v0.05.")
