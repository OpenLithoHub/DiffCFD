"""Optimization workflows: aerodynamic, topology."""

from diffcfd.workflows.aero import optimize_airfoil
from diffcfd.workflows.topology import optimize_topology

__all__ = ["optimize_airfoil", "optimize_topology"]
