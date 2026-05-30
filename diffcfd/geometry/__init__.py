"""Geometry: Cartesian mesh, SDF shapes, airfoils, filters."""

from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.geometry.shapes import cylinder_sdf, rectangle_sdf, naca0012_sdf
from diffcfd.geometry.airfoil import NACA4Digit, BSplineAirfoil
from diffcfd.geometry.filters import HelmholtzFilter

__all__ = [
    "CartesianMesh",
    "cylinder_sdf",
    "rectangle_sdf",
    "naca0012_sdf",
    "NACA4Digit",
    "BSplineAirfoil",
    "HelmholtzFilter",
]
