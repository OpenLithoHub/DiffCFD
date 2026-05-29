"""Optimization workflows: aerodynamic, topology, convergence monitoring."""

from diffcfd.workflows.aero import optimize_airfoil
from diffcfd.workflows.topology import optimize_topology, multi_corner_optimize

__all__ = ["optimize_airfoil", "optimize_topology", "multi_corner_optimize"]
