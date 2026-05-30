"""DiffCFD — Differentiable Computational Fluid Dynamics."""

__version__ = "0.7.0"

from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.geometry.shapes import cylinder_sdf, rectangle_sdf, naca0012_sdf
from diffcfd.geometry.airfoil import NACA4Digit, BSplineAirfoil
from diffcfd.geometry.filters import HelmholtzFilter
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
from diffcfd.solvers.heat_transfer import HeatTransfer2D
from diffcfd.solvers.turbulence import FrozenEddyViscosity
from diffcfd.solvers.implicit_diff import fixed_point_gradient
from diffcfd.envs.base import DiffCFDEnv
from diffcfd.envs.cylinder_wake import CylinderWakeEnv
from diffcfd.envs.heat_exchanger import HeatExchangerEnv
from diffcfd.props.ideal_gas import ConstantProps, ThermophysicalProps
from diffcfd.props.sco2 import SCO2Surrogate, train_sco2_surrogate
from diffcfd.workflows.aero import optimize_airfoil
from diffcfd.workflows.topology import optimize_topology, smooth_heaviside
from diffcfd.workflows.pche import optimize_pche
from diffcfd.solvers.spin_coating import MeyerhoferSolver, RadialThinFilmSolver
from diffcfd.workflows.spin_coat_opt import optimize_spin_profile
from diffcfd.surrogates.fno import FNO2D, train_fno

__all__ = [
    "CartesianMesh",
    "cylinder_sdf",
    "rectangle_sdf",
    "naca0012_sdf",
    "NACA4Digit",
    "BSplineAirfoil",
    "HelmholtzFilter",
    "NavierStokes2D",
    "HeatTransfer2D",
    "FrozenEddyViscosity",
    "fixed_point_gradient",
    "DiffCFDEnv",
    "CylinderWakeEnv",
    "HeatExchangerEnv",
    "ConstantProps",
    "ThermophysicalProps",
    "SCO2Surrogate",
    "train_sco2_surrogate",
    "optimize_airfoil",
    "optimize_topology",
    "smooth_heaviside",
    "optimize_pche",
    "MeyerhoferSolver",
    "RadialThinFilmSolver",
    "optimize_spin_profile",
    "FNO2D",
    "train_fno",
]
