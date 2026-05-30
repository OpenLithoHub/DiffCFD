"""Solvers: SIMPLE NS, boundary conditions, implicit differentiation."""

from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
from diffcfd.solvers.heat_transfer import HeatTransfer2D
from diffcfd.solvers.turbulence import FrozenEddyViscosity
from diffcfd.solvers.implicit_diff import fixed_point_gradient

__all__ = [
    "NavierStokes2D",
    "HeatTransfer2D",
    "FrozenEddyViscosity",
    "fixed_point_gradient",
]
