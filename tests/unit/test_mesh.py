"""Unit tests for CartesianMesh."""

import torch


def test_mesh_dimensions(small_mesh):
    assert small_mesh.nx == 8
    assert small_mesh.ny == 8


def test_mesh_cell_size(small_mesh):
    assert abs(small_mesh.dx - 1.0 / 8) < 1e-10
    assert abs(small_mesh.dy - 1.0 / 8) < 1e-10


def test_mesh_cell_centers_shape(small_mesh):
    x, y = small_mesh.cell_centers()
    assert x.shape == (8, 8)
    assert y.shape == (8, 8)


def test_mesh_cell_centers_range(small_mesh):
    x, y = small_mesh.cell_centers()
    dx = 1.0 / 8
    assert abs(x.min().item() - dx / 2) < 1e-6
    assert abs(x.max().item() - (1.0 - dx / 2)) < 1e-6
    assert abs(y.min().item() - dx / 2) < 1e-6
    assert abs(y.max().item() - (1.0 - dx / 2)) < 1e-6


def test_mesh_device(small_mesh):
    x, _ = small_mesh.cell_centers()
    assert x.device.type == "cpu"
