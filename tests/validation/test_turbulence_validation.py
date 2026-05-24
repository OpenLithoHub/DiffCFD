"""Turbulence model validation: duct flow at Re=10000 vs Dittus-Boelter."""

import pytest
import torch


@pytest.mark.slow
@pytest.mark.validation
def test_turbulent_duct_nusselt():
    """Duct flow at Re=10000: Nu should be within 50% of Dittus-Boelter.

    Dittus-Boelter correlation: Nu = 0.023 * Re^0.8 * Pr^0.4

    Uses Blasius friction velocity and effective thermal diffusivity with
    turbulent Prandtl number Pr_t = 0.9.
    """
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
    from diffcfd.solvers.turbulence import FrozenEddyViscosity
    from diffcfd.solvers.heat_transfer import HeatTransfer2D

    Re = 10000.0
    Pr = 0.71
    nx, ny = 48, 24
    lx, ly = 4.0, 1.0
    nu = 1.0 / Re

    # Generate frozen eddy viscosity from Blasius friction velocity
    fev = FrozenEddyViscosity.from_blasius(
        Re=Re, ny=ny, nx=nx, ly=ly, U_bulk=1.0,
    )

    # Run NS solver with effective viscosity (spatially averaged)
    mu_eff = fev.effective_viscosity(nu)
    nu_eff_avg = mu_eff.mean().item()
    Re_eff = 1.0 / nu_eff_avg

    solver = NavierStokes2D(
        reynolds_number=Re_eff,
        grid=(nx, ny),
        lx=lx,
        ly=ly,
        max_iter=2000,
        tol=1e-5,
    )

    ux, uy, p = solver.solve_steady(inlet_velocity=1.0, case="channel")

    # Solve heat transfer with effective thermal diffusivity
    alpha_mol = nu / Pr
    alpha_eff = fev.effective_thermal_diffusivity(alpha_mol, Pr_t=0.9)
    alpha_eff_avg = alpha_eff.mean().item()

    mesh = solver.mesh
    ht = HeatTransfer2D(mesh, alpha=alpha_eff_avg)

    T_bc = {
        "bottom": ("dirichlet", 1.0),
        "top": ("dirichlet", 0.0),
        "left": ("neumann", 0.0),
        "right": ("neumann", 0.0),
    }
    T = ht.solve(ux, uy, T_bc=T_bc)
    Nu = ht.nusselt_number(T, T_hot=1.0, T_cold=0.0, L=ly, wall="bottom")

    # Dittus-Boelter correlation
    Nu_db = 0.023 * Re ** 0.8 * Pr ** 0.4

    rel_error = abs(Nu.item() - Nu_db) / Nu_db
    print(f"Computed Nu = {Nu.item():.2f}, Dittus-Boelter Nu = {Nu_db:.2f}, "
          f"error = {rel_error * 100:.1f}%")

    # Allow generous tolerance since frozen μ_t is approximate
    assert rel_error < 0.50, (
        f"Nu error {rel_error * 100:.1f}% exceeds 50% threshold"
    )


def test_mixing_length_produces_nonzero_mut():
    """Mixing-length model produces non-zero eddy viscosity."""
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    fev = FrozenEddyViscosity.from_blasius(
        Re=10000, ny=50, nx=20, ly=1.0, U_bulk=1.0,
    )
    mu_eff = fev.effective_viscosity(1e-4)
    assert fev.mu_t.max().item() > 0, "Eddy viscosity should be non-zero"
    assert mu_eff.min().item() >= 1e-4, "Effective viscosity should be >= molecular"


def test_effective_thermal_diffusivity():
    """Effective thermal diffusivity includes turbulent contribution."""
    from diffcfd.solvers.turbulence import FrozenEddyViscosity

    fev = FrozenEddyViscosity.from_blasius(
        Re=10000, ny=24, nx=48, ly=1.0, U_bulk=1.0,
    )
    alpha_mol = 1e-4 / 0.71
    alpha_eff = fev.effective_thermal_diffusivity(alpha_mol, Pr_t=0.9)

    assert alpha_eff.min().item() >= alpha_mol, \
        "Effective alpha should be >= molecular"
    assert alpha_eff.max().item() > alpha_mol, \
        "Turbulent contribution should increase effective alpha"
