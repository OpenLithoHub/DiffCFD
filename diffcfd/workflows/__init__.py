"""Optimization workflows: aerodynamic, topology, spin coating, litho."""

from diffcfd.workflows.aero import optimize_airfoil
from diffcfd.workflows.topology import optimize_topology, multi_corner_optimize
from diffcfd.workflows.spin_coat_opt import optimize_spin_profile
from diffcfd.workflows.joint_litho_opt import optimize_joint_process

__all__ = [
    "optimize_airfoil",
    "optimize_topology",
    "multi_corner_optimize",
    "optimize_spin_profile",
    "optimize_joint_process",
]
