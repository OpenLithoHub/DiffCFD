"""Edge case and integration tests for DiffCFD."""

import torch
import numpy as np


# ── Geometry edge cases ──────────────────────────────────────────────


def test_sdf_cylinder_at_boundary():
    """Cylinder SDF near domain boundary doesn't produce NaN."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.shapes import cylinder_sdf

    mesh = CartesianMesh(16, 16)
    sdf = cylinder_sdf(mesh, 0.05, 0.05, 0.1)
    assert not torch.any(torch.isnan(sdf))
    assert not torch.any(torch.isinf(sdf))


def test_sdf_naca0012_symmetric():
    """NACA 0012 SDF is symmetric about the camber line (y=0.5)."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.shapes import naca0012_sdf

    mesh = CartesianMesh(64, 64, lx=1.0, ly=1.0)
    sdf = naca0012_sdf(mesh, chord=0.4, leading_edge_x=0.3, center_y=0.5, angle_deg=0.0)

    ny = 64
    # Upper half vs lower half should be symmetric
    diff = (sdf[: ny // 2, :] - sdf[ny // 2 :, :].flip(0)).abs()
    assert diff.max().item() < 0.05, f"SDF asymmetry: {diff.max().item():.4f}"


def test_mesh_sdf_mask_brinkman():
    """SDF-to-mask produces values in [0, 1] for smooth Heaviside."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.shapes import cylinder_sdf

    mesh = CartesianMesh(32, 32)
    sdf = cylinder_sdf(mesh, 0.5, 0.5, 0.2)
    mask = mesh.sdf_to_mask(sdf, epsilon=1e-3)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    # Interior of cylinder should be ~0, exterior ~1
    assert mask[16, 16].item() < 0.5  # center of cylinder


def test_mesh_sdf_mask_sharp_interface():
    """Very small epsilon approaches step function."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.shapes import cylinder_sdf

    mesh = CartesianMesh(64, 64)
    sdf = cylinder_sdf(mesh, 0.5, 0.5, 0.2)
    mask_sharp = mesh.sdf_to_mask(sdf, epsilon=1e-6)
    # Most values should be near 0 or 1
    mid = (mask_sharp > 0.01) & (mask_sharp < 0.99)
    assert mid.float().mean().item() < 0.25


# ── Solver edge cases ────────────────────────────────────────────────


def test_channel_very_low_re():
    """Channel flow at very low Re (Re=1) converges without NaN."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    solver = NavierStokes2D(
        reynolds_number=1.0,
        grid=(16, 8),
        max_iter=500,
        tol=1e-4,
    )
    ux, uy, p = solver.solve_steady(inlet_velocity=1.0, case="channel")
    assert not torch.any(torch.isnan(ux))
    assert not torch.any(torch.isnan(p))
    # At Re=1, flow should be nearly fully developed Poiseuille
    dp = solver.pressure_drop(ux, uy, p)
    assert dp.item() > 0


def test_channel_sdf_obstruction():
    """Channel with cylinder obstruction has higher pressure drop."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
    from diffcfd.geometry.shapes import cylinder_sdf

    solver_clean = NavierStokes2D(
        reynolds_number=50,
        grid=(32, 16),
        lx=2.0,
        ly=1.0,
        max_iter=500,
        tol=1e-4,
    )
    ux_c, uy_c, p_c = solver_clean.solve_steady(inlet_velocity=1.0, case="channel")
    dp_clean = solver_clean.pressure_drop(ux_c, uy_c, p_c).item()

    solver_obs = NavierStokes2D(
        reynolds_number=50,
        grid=(32, 16),
        lx=2.0,
        ly=1.0,
        max_iter=500,
        tol=1e-4,
    )
    sdf = cylinder_sdf(solver_obs.mesh, 1.0, 0.5, 0.15)
    ux_o, uy_o, p_o = solver_obs.solve_steady(
        sdf=sdf, inlet_velocity=1.0, case="channel"
    )
    dp_obstructed = solver_obs.pressure_drop(ux_o, uy_o, p_o).item()

    assert dp_obstructed > dp_clean, (
        f"Obstructed dp ({dp_obstructed:.4f}) should exceed clean ({dp_clean:.4f})"
    )


def test_implicit_diff_gradient_finite():
    """Implicit differentiation produces finite gradients."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    solver = NavierStokes2D(
        reynolds_number=50,
        grid=(16, 8),
        lx=2.0,
        ly=1.0,
        backward="implicit_diff",
        max_iter=500,
        tol=1e-4,
    )
    inlet = torch.tensor(1.0, requires_grad=True)
    ux, uy, p = solver.solve_steady(inlet_velocity=inlet, case="channel")
    dp = solver.pressure_drop(ux, uy, p)
    dp.backward()
    assert inlet.grad is not None
    assert torch.isfinite(inlet.grad).all()
    # dΔP/dU should be positive (higher velocity → higher pressure drop)
    assert inlet.grad.item() > 0


# ── Heat transfer edge cases ─────────────────────────────────────────


def test_pure_conduction_nusselt():
    """Pure conduction (zero velocity) gives Nu=1.0 for parallel plates."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.solvers.heat_transfer import HeatTransfer2D

    mesh = CartesianMesh(16, 16, lx=1.0, ly=1.0)
    ht = HeatTransfer2D(mesh, alpha=1.0)

    ux = torch.zeros(16, 17)
    uy = torch.zeros(17, 16)

    T_bc = {
        "bottom": ("dirichlet", 1.0),
        "top": ("dirichlet", 0.0),
        "left": ("neumann", 0.0),
        "right": ("neumann", 0.0),
    }
    T = ht.solve(ux, uy, T_bc=T_bc)
    Nu = ht.nusselt_number(T, T_hot=1.0, T_cold=0.0, L=1.0, wall="bottom")
    assert abs(Nu.item() - 1.0) < 0.05, f"Pure conduction Nu={Nu.item():.3f} != 1.0"


def test_heat_transfer_differentiable_gradient():
    """Gradient flows through differentiable heat solve."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.solvers.heat_transfer import HeatTransfer2D

    mesh = CartesianMesh(8, 8)
    ht = HeatTransfer2D(mesh, alpha=0.1)

    ux = torch.randn(8, 9) * 0.1
    ux.requires_grad_(True)
    uy = torch.randn(9, 8) * 0.1
    uy.requires_grad_(True)

    T_bc = {
        "bottom": ("dirichlet", 1.0),
        "top": ("dirichlet", 0.0),
        "left": ("neumann", 0.0),
        "right": ("neumann", 0.0),
    }
    T = ht.solve_differentiable(ux, uy, T_bc=T_bc, max_iter=50)
    loss = T.sum()
    loss.backward()
    assert ux.grad is not None
    assert ux.grad.norm().item() > 0


# ── Turbulence edge cases ────────────────────────────────────────────


def test_blasius_from_re_range():
    """Blasius model works across a range of Re values."""
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    for Re in [5000, 10000, 50000]:
        fev = FrozenEddyViscosity.from_blasius(Re=Re, ny=20, nx=10, ly=1.0)
        assert fev.mu_t.max().item() > 0, f"Re={Re}: μ_t should be non-zero"

    # Effective viscosity ratio mu_t/nu should increase with Re
    # (turbulent contribution grows relative to molecular viscosity)
    fev_low = FrozenEddyViscosity.from_blasius(Re=5000, ny=20, nx=10, ly=1.0)
    fev_high = FrozenEddyViscosity.from_blasius(Re=50000, ny=20, nx=10, ly=1.0)
    nu_low = 1.0 / 5000
    nu_high = 1.0 / 50000
    ratio_low = (fev_low.mu_t.max() / nu_low).item()
    ratio_high = (fev_high.mu_t.max() / nu_high).item()
    assert ratio_high > ratio_low, (
        f"μ_t/nu ratio should increase with Re: {ratio_high:.1f} vs {ratio_low:.1f}"
    )


# ── Environment edge cases ───────────────────────────────────────────


def test_env_reset_idempotent():
    """Resetting environment twice gives same initial observation."""
    from diffcfd.envs.heat_exchanger import HeatExchangerEnv

    env = HeatExchangerEnv(re=50.0, grid=(16, 8))
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    np.testing.assert_allclose(obs1, obs2, atol=1e-6)
    env.close()


def test_cylinder_env_zero_action():
    """Zero rotation action doesn't produce NaN."""
    from diffcfd.envs.cylinder_wake import CylinderWakeEnv

    env = CylinderWakeEnv(
        re=100.0,
        grid=(32, 16),
        max_steps=2,
        lx=2.5,
        ly=1.0,
        cylinder_radius=0.05,
        cylinder_center=(0.5, 0.5),
    )
    obs, _ = env.reset()
    obs2, reward, done, truncated, info = env.step([0.0])
    assert not np.any(np.isnan(obs2))
    assert np.isfinite(reward)
    env.close()


# ── VTK export edge cases ────────────────────────────────────────────


def test_vtk_export_basic(tmp_path):
    """VTK export produces a non-empty file."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.export.vtk import save_vtk

    mesh = CartesianMesh(8, 8)
    ux = torch.ones(8, 9) * 0.5
    uy = torch.zeros(9, 8)
    p = torch.randn(8, 8)
    T = torch.randn(8, 8)

    path = tmp_path / "test.vtk"
    save_vtk(ux, uy, p, mesh, path, T=T)

    content = path.read_text()
    assert "velocity" in content
    assert "pressure" in content
    assert "temperature" in content
    assert "vorticity" in content
    assert "velocity_magnitude" in content


def test_vtk_export_extra_scalars(tmp_path):
    """VTK export handles extra scalar fields."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.export.vtk import save_vtk

    mesh = CartesianMesh(4, 4)
    ux = torch.zeros(4, 5)
    uy = torch.zeros(5, 4)
    p = torch.zeros(4, 4)

    path = tmp_path / "extra.vtk"
    save_vtk(
        ux,
        uy,
        p,
        mesh,
        path,
        extra_scalars={
            "density": torch.ones(4, 4),
            "viscosity": torch.ones(4, 4) * 0.5,
        },
    )

    content = path.read_text()
    assert "density" in content
    assert "viscosity" in content
