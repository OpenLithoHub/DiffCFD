"""Unit tests for boundary condition implementations."""

import pytest
import torch


def test_boundary_import():
    from diffcfd.solvers.boundary import BoundaryConditions
    assert BoundaryConditions is not None


def test_boundary_apply_inlet(small_mesh):
    from diffcfd.solvers.boundary import BoundaryConditions
    bc = BoundaryConditions(small_mesh)
    ny, nx = small_mesh.ny, small_mesh.nx
    ux = torch.zeros(ny, nx + 1)
    ux_out = bc.apply_inlet(ux, profile=1.0)
    assert ux_out[:, 0].allclose(torch.ones(ny))
    assert ux_out[:, 1:].allclose(torch.zeros(ny, nx))


def test_boundary_no_slip(small_mesh):
    from diffcfd.solvers.boundary import BoundaryConditions
    bc = BoundaryConditions(small_mesh)
    ny, nx = small_mesh.ny, small_mesh.nx
    ux = torch.ones(ny, nx + 1)
    uy = torch.ones(ny + 1, nx)
    ux_out, uy_out = bc.apply_no_slip_walls(ux, uy)
    assert ux_out[0, :].allclose(torch.zeros(nx + 1))
    assert ux_out[-1, :].allclose(torch.zeros(nx + 1))
    assert uy_out[0, :].allclose(torch.zeros(nx))
    assert uy_out[-1, :].allclose(torch.zeros(nx))
