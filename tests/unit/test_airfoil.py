"""Tests for airfoil geometry and aerodynamic optimization."""

import torch


def test_airfoil_import():
    from diffcfd.geometry.airfoil import NACA4Digit

    assert NACA4Digit is not None


def test_naca0012_sdf():
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.airfoil import NACA4Digit

    mesh = CartesianMesh(nx=64, ny=64, lx=2.0, ly=2.0)
    naca = NACA4Digit(chord=0.5, leading_edge_x=0.5, center_y=1.0)
    sdf = naca.sdf(mesh, thickness=0.12)

    assert sdf.shape == (64, 64)
    assert sdf.min() < 0, "SDF must be negative inside airfoil"
    assert sdf.max() > 0, "SDF must be positive outside airfoil"


def test_naca_cambered():
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.airfoil import NACA4Digit

    mesh = CartesianMesh(nx=64, ny=64, lx=2.0, ly=2.0)
    naca = NACA4Digit(chord=0.5, leading_edge_x=0.5, center_y=1.0, angle_deg=5.0)
    sdf = naca.sdf(mesh, thickness=0.12, camber=0.02, camber_pos=0.4)

    assert sdf.shape == (64, 64)
    assert sdf.min() < 0


def test_bspline_airfoil():
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.airfoil import BSplineAirfoil

    mesh = CartesianMesh(nx=64, ny=64, lx=2.0, ly=2.0)
    bspline = BSplineAirfoil(
        n_control_points=6, chord=0.8, leading_edge_x=0.5, center_y=1.0
    )
    cp = bspline.initial_control_points()

    assert cp.shape == (12, 2)
    sdf = bspline.sdf(mesh, cp)
    assert sdf.shape == (64, 64)
    assert sdf.min() < 0, "SDF must be negative inside B-spline airfoil"


def test_compute_forces():
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.airfoil import compute_forces

    mesh = CartesianMesh(nx=16, ny=16, lx=1.0, ly=1.0)
    p = torch.zeros(16, 16)
    ux = torch.ones(16, 17) * 0.1
    uy = torch.zeros(17, 16)
    sdf = torch.ones(16, 16)  # No body

    drag, lift = compute_forces(p, ux, uy, mesh, sdf, mu=0.01)
    assert drag.shape == ()
    assert lift.shape == ()
