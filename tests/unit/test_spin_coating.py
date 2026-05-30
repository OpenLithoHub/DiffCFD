"""Unit and gradient tests for differentiable spin coating solvers."""

import math

import torch

from diffcfd.solvers.spin_coating import MeyerhoferSolver, RadialThinFilmSolver

RPM_TO_RAD = 2.0 * math.pi / 60.0


def test_meyerhofer_import():
    assert MeyerhoferSolver is not None


def test_radial_import():
    assert RadialThinFilmSolver is not None


def test_meyerhofer_forward():
    """MeyerhoferSolver outputs physically valid positive thickness."""
    solver = MeyerhoferSolver()
    omega = torch.full((100,), 2000.0 * RPM_TO_RAD)
    dt = 0.05

    h_hist, c_hist = solver(omega, dt, h0=8e-6, c0=0.85)

    assert h_hist.shape == (100,)
    assert c_hist.shape == (100,)
    assert (h_hist > 0).all()
    assert (c_hist >= 0).all() and (c_hist <= 1.0).all()
    assert h_hist[-1] < h_hist[0]


def test_spin_coating_gradients_flow():
    """Autograd gradients flow back to the RPM profile."""
    solver = MeyerhoferSolver()
    omega = torch.full((50,), 1500.0 * RPM_TO_RAD, requires_grad=True)
    dt = 0.005

    h_hist, _ = solver(omega, dt, h0=5e-6, c0=0.8)
    loss = (h_hist[-1] - 100e-9) ** 2
    loss.backward()

    assert omega.grad is not None
    assert torch.isfinite(omega.grad).all()
    assert (omega.grad < 0).all()


def test_radial_pde_no_spin():
    """With zero rotation, film still thins from evaporation."""
    solver = RadialThinFilmSolver(nr=10)
    omega = torch.zeros(50)
    dt = 0.1
    h0 = torch.full((10,), 5e-6)
    c0 = torch.full((10,), 0.8)

    h_final = solver(omega, dt, h0, c0)
    assert (h_final > 0).all()
    assert h_final.mean() < h0.mean()


def test_radial_pde_gradient_flows():
    """Gradients flow through the radial PDE solver."""
    solver = RadialThinFilmSolver(nr=10)
    omega = torch.full((50,), 2000.0 * RPM_TO_RAD, requires_grad=True)
    h0 = torch.full((10,), 5e-6)
    c0 = torch.full((10,), 0.8)

    h_final = solver(omega, 0.05, h0, c0)
    loss = h_final.mean()
    loss.backward()

    assert omega.grad is not None
    assert torch.isfinite(omega.grad).all()


def test_meyerhofer_higher_spin_thinner_film():
    """Higher RPM must produce thinner final film."""
    solver = MeyerhoferSolver()
    dt = 0.005
    n = 10

    omega_low = torch.full((n,), 1000.0 * RPM_TO_RAD)
    omega_high = torch.full((n,), 4000.0 * RPM_TO_RAD)

    h_low, _ = solver(omega_low, dt, h0=8e-6, c0=0.85)
    h_high, _ = solver(omega_high, dt, h0=8e-6, c0=0.85)

    assert h_high[-1] < h_low[-1]
