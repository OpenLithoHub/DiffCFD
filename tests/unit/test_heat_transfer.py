"""Unit and validation tests for conjugate heat transfer solver."""

import pytest
import torch


def test_heat_transfer_import():
    from diffcfd.solvers.heat_transfer import HeatTransfer2D, coupled_steady_solve
    assert HeatTransfer2D is not None


def test_pure_conduction_linear():
    """Pure conduction with T=0 bottom, T=1 top gives Nu=1.0 exactly."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.solvers.heat_transfer import HeatTransfer2D

    mesh = CartesianMesh(nx=32, ny=32, lx=1.0, ly=1.0)
    ht = HeatTransfer2D(mesh, alpha=0.1)
    ux = torch.zeros(32, 33)
    uy = torch.zeros(33, 32)
    T_bc = {
        "bottom": ("dirichlet", 0.0),
        "top": ("dirichlet", 1.0),
        "left": ("neumann", 0.0),
        "right": ("neumann", 0.0),
    }
    T = ht.solve(ux, uy, T_bc=T_bc)

    # Linear profile check at center column
    dy = 1.0 / 32
    y = torch.arange(32) * dy + dy / 2
    T_center = T[:, 16]
    l2_err = (torch.norm(T_center - y) / torch.norm(y)).item()
    assert l2_err < 1e-5, f"Pure conduction L2 error {l2_err:.2e}"

    # Nusselt number
    Nu = ht.nusselt_number(T, T_hot=1.0, T_cold=0.0, L=1.0, wall="bottom")
    assert abs(Nu.item() - 1.0) < 1e-3, f"Nu={Nu:.4f}, expected 1.0"


def test_conduction_with_lateral_dirichlet():
    """All four walls Dirichlet: check solution is within bounds."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.solvers.heat_transfer import HeatTransfer2D

    mesh = CartesianMesh(nx=16, ny=16, lx=1.0, ly=1.0)
    ht = HeatTransfer2D(mesh, alpha=0.1)
    ux = torch.zeros(16, 17)
    uy = torch.zeros(17, 16)
    T_bc = {
        "bottom": ("dirichlet", 0.0),
        "top": ("dirichlet", 1.0),
        "left": ("dirichlet", 0.0),
        "right": ("dirichlet", 0.0),
    }
    T = ht.solve(ux, uy, T_bc=T_bc)
    assert T.min() >= -0.01 and T.max() <= 1.01
    assert T.mean() > 0.2, "Mean temperature should be above 0.2"


def test_vtk_export(tmp_path):
    """VTK export creates a valid file."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.export.vtk import save_vtk

    mesh = CartesianMesh(nx=8, ny=8, lx=1.0, ly=1.0)
    ux = torch.zeros(8, 9)
    uy = torch.zeros(9, 8)
    p = torch.zeros(8, 8)

    path = tmp_path / "test_output.vtk"
    save_vtk(ux, uy, p, mesh, path)
    assert path.exists()
    content = path.read_text()
    assert "DATASET RECTILINEAR_GRID" in content
    assert "VECTORS velocity" in content
    assert "SCALARS pressure" in content


@pytest.mark.slow
@pytest.mark.validation
def test_coupled_channel_flow_heat():
    """Coupled Poiseuille + heated walls: temperature field is bounded."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
    from diffcfd.solvers.heat_transfer import HeatTransfer2D, coupled_steady_solve

    ns = NavierStokes2D(reynolds_number=10.0, grid=(32, 16), lx=4.0, ly=1.0, tol=1e-6)
    mesh = ns.mesh
    ht = HeatTransfer2D(mesh, alpha=0.01)
    T_bc = {
        "bottom": ("dirichlet", 0.0),
        "top": ("dirichlet", 1.0),
        "left": ("neumann", 0.0),
        "right": ("neumann", 0.0),
    }
    ux, uy, p, T = coupled_steady_solve(
        ns, ht, T_bc=T_bc, inlet_velocity=1.0, case="channel"
    )
    assert T.min() >= -0.01, f"T.min={T.min():.4f}"
    assert T.max() <= 1.01, f"T.max={T.max():.4f}"
    assert ux.mean() > 0, "Flow should have positive velocity"
