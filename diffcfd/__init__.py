"""DiffCFD — Differentiable Computational Fluid Dynamics."""

__version__ = "0.4.1"

from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.geometry.shapes import cylinder_sdf, rectangle_sdf, naca0012_sdf
from diffcfd.geometry.airfoil import NACA4Digit, BSplineAirfoil
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
from diffcfd.solvers.heat_transfer import HeatTransfer2D
from diffcfd.solvers.turbulence import FrozenEddyViscosity
from diffcfd.solvers.implicit_diff import fixed_point_gradient
from diffcfd.envs.base import DiffCFDEnv
from diffcfd.envs.cylinder_wake import CylinderWakeEnv
from diffcfd.envs.heat_exchanger import HeatExchangerEnv
from diffcfd.props.ideal_gas import ConstantProps
