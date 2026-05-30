"""Tests for the coupled Spin-Lithography optimization pipeline."""

import math

import torch

from diffcfd.solvers.litho import LithoSolver
from diffcfd.solvers.spin_coating import MeyerhoferSolver

RPM_TO_RAD = 2.0 * math.pi / 60.0


def test_litho_import():
    assert LithoSolver is not None


def test_litho_solver_forward():
    """LithoSolver returns physically valid remaining thickness."""
    solver = LithoSolver()

    thickness = torch.tensor(1000e-9)
    solvent = torch.tensor(0.08)
    dose = torch.tensor(80.0)

    h_final = solver(thickness, solvent, dose, dev_time=30.0)

    assert h_final.item() >= 0.0
    assert h_final.item() <= thickness.item()


def test_litho_solver_gradients():
    """Gradients flow from developed thickness back to exposure dose and solvent."""
    solver = LithoSolver()

    thickness = torch.tensor(1000e-9, requires_grad=True)
    solvent = torch.tensor(0.08, requires_grad=True)
    dose = torch.tensor(30.0, requires_grad=True)

    h_final = solver(thickness, solvent, dose, dev_time=30.0)

    loss = h_final.sum()
    loss.backward()

    assert thickness.grad is not None
    assert dose.grad is not None
    assert solvent.grad is not None
    assert torch.isfinite(dose.grad).all()


def test_litho_higher_dose_thinner_resist():
    """Higher exposure dose must produce thinner remaining resist."""
    solver = LithoSolver()

    dose_low = torch.tensor(20.0)
    dose_high = torch.tensor(100.0)
    thickness = torch.tensor(1000e-9)
    solvent = torch.tensor(0.08)

    h_low = solver(thickness, solvent, dose_low, dev_time=30.0)
    h_high = solver(thickness, solvent, dose_high, dev_time=30.0)

    assert h_high.item() < h_low.item()


def test_coupled_spin_litho_gradient_path():
    """Gradient path is unbroken through the combined spin + litho stack."""
    spin_s = MeyerhoferSolver()
    litho_s = LithoSolver()

    omega = torch.full((20,), 3000.0 * RPM_TO_RAD, requires_grad=True)
    dose = torch.tensor(30.0, requires_grad=True)

    h_hist, c_hist = spin_s(omega, 0.005, h0=6e-6, c0=0.85)
    h_final = litho_s(h_hist[-1], c_hist[-1], dose, dev_time=30.0)

    loss = (h_final - 50e-9) ** 2
    loss.backward()

    assert omega.grad is not None
    assert dose.grad is not None
    assert torch.isfinite(omega.grad).all()
    assert torch.isfinite(dose.grad).all()
