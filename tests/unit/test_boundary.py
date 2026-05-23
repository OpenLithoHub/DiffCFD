"""Unit tests for boundary condition stubs."""

import pytest


def test_boundary_import():
    from diffcfd.solvers.boundary import BoundaryConditions
    assert BoundaryConditions is not None


def test_boundary_inlet_not_implemented(small_mesh):
    from diffcfd.solvers.boundary import BoundaryConditions
    import torch
    bc = BoundaryConditions(small_mesh)
    u = torch.zeros(2, small_mesh.ny, small_mesh.nx)
    with pytest.raises(NotImplementedError):
        bc.apply_inlet(u, profile=1.0)
