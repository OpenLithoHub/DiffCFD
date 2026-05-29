"""Tests for frozen eddy viscosity turbulence model."""

import torch


def test_turbulence_import():
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    assert FrozenEddyViscosity is not None


def test_mixing_length_channel():
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    mu_t = FrozenEddyViscosity.mixing_length_channel(
        ny=64, nx=32, ly=1.0, u_tau=0.05, nu=1e-4
    )
    assert mu_t.mu_t.shape == (64, 32)
    assert mu_t.mu_t.min() >= 0, "Eddy viscosity must be non-negative"


def test_effective_viscosity():
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    mu_t = FrozenEddyViscosity.mixing_length_channel(
        ny=32, nx=16, ly=1.0, u_tau=0.05, nu=1e-4
    )
    mu_eff = mu_t.effective_viscosity(mu=1e-4)
    assert mu_eff.min() >= 1e-4, "Effective viscosity >= molecular viscosity"


def test_from_tensor():
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    mu_t_field = torch.ones(16, 16) * 0.01
    fev = FrozenEddyViscosity.from_tensor(mu_t_field)
    assert fev.mu_t.shape == (16, 16)
    assert not fev.mu_t.requires_grad, "Frozen mu_t must not require grad"


def test_perturbation_check():
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    mu_t1 = FrozenEddyViscosity.mixing_length_channel(
        ny=32, nx=16, ly=1.0, u_tau=0.05, nu=1e-4
    )
    mu_t2 = FrozenEddyViscosity.mixing_length_channel(
        ny=32, nx=16, ly=1.0, u_tau=0.05, nu=1e-4
    )
    cos_sim = mu_t1.perturbation_validity_check(mu_t2.mu_t)
    assert abs(cos_sim - 1.0) < 1e-5, "Same μ_t should have cosine similarity 1.0"


def test_turbulence_in_ns_solver():
    """Test that FrozenEddyViscosity can be passed to NavierStokes2D."""
    from diffcfd.solvers.turbulence import FrozenEddyViscosity
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    mu_t = FrozenEddyViscosity.mixing_length_channel(
        ny=16, nx=32, ly=1.0, u_tau=0.05, nu=1e-3
    )
    solver = NavierStokes2D(
        reynolds_number=1000,
        grid=(32, 16),
        lx=4.0,
        ly=1.0,
        turbulence=mu_t,
    )
    assert solver._nu_field is not None
    assert solver._nu_field.shape == (16, 32)
    assert solver._nu_field.min() >= 1e-3

    # Run solver — should converge with turbulent viscosity
    ux, uy, p = solver.solve_steady(inlet_velocity=1.0, case="channel")
    assert ux.shape == (16, 33)
    assert torch.isfinite(ux).all()
