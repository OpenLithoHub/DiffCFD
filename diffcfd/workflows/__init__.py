"""Optimization workflows: aerodynamic, topology, spin coating."""

from diffcfd.workflows.aero import optimize_airfoil
from diffcfd.workflows.topology import optimize_topology, multi_corner_optimize
from diffcfd.workflows.spin_coat_opt import optimize_spin_profile

__all__ = [
    "optimize_airfoil",
    "optimize_topology",
    "multi_corner_optimize",
    "optimize_spin_profile",
]
