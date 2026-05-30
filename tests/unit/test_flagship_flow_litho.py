"""Validation tests for the flagship flow-litho co-optimization pipeline.

Runs short versions of both joint and decoupled optimization and
verifies correctness of the optimization, gradient flow, and process window.

Note: spin time is kept short (0.2s) so the film stays fluid and gradients
flow through the centrifugal-thinning term.  Once solvent fraction drops
below c_solid (0.15), the flow_mask sigmoid kills omega gradients.
"""

from __future__ import annotations

import math

import torch

from diffcfd.solvers.litho import LithoSolver
from diffcfd.solvers.spin_coating import MeyerhoferSolver
from diffcfd.workflows.joint_litho_opt import (
    optimize_decoupled_process,
    optimize_joint_process,
    process_window_analysis,
)

RPM_TO_RAD = 2.0 * math.pi / 60.0

TARGET_NM = 50.0
# Short spin time keeps solvent fraction high enough for gradients to flow.
SPIN_DT = 0.005
TOTAL_TIME = 0.2  # 40 steps -- film still fluid at end
N_STEPS = int(TOTAL_TIME / SPIN_DT)
N_EPOCHS = 5
N_EPOCHS_LONG = 20


def test_joint_returns_finite_nonnan():
    """Joint optimization produces finite, non-NaN losses throughout."""
    res = optimize_joint_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_TIME,
        spin_dt=SPIN_DT,
        n_epochs=N_EPOCHS,
        verbose=False,
    )
    for loss_val in res["loss_history"]:
        assert math.isfinite(loss_val), f"Non-finite loss: {loss_val}"
    assert math.isfinite(res["final_developed_nm"])
    assert res["final_developed_nm"] >= 0.0


def test_decoupled_returns_finite_nonnan():
    """Decoupled optimization produces finite, non-NaN losses throughout."""
    res = optimize_decoupled_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_TIME,
        spin_dt=SPIN_DT,
        n_spin_epochs=N_EPOCHS,
        n_dose_epochs=N_EPOCHS,
        verbose=False,
    )
    for loss_val in res["loss_history"]:
        assert math.isfinite(loss_val), f"Non-finite loss: {loss_val}"
    assert math.isfinite(res["final_developed_nm"])
    assert res["final_developed_nm"] >= 0.0


def test_gradient_flows_through_full_chain():
    """Gradients propagate from developed thickness back to omega and dose."""
    spin_solver = MeyerhoferSolver()
    litho_solver = LithoSolver()

    omega = torch.full((N_STEPS,), 2500.0 * RPM_TO_RAD, requires_grad=True)
    dose = torch.tensor(80.0, requires_grad=True)

    h_hist, c_hist = spin_solver(omega, SPIN_DT, h0=8e-6, c0=0.85)
    h_dev = litho_solver(h_hist[-1], c_hist[-1], dose, dev_time=30.0)

    loss = ((h_dev - TARGET_NM * 1e-9) ** 2) / (TARGET_NM * 1e-9) ** 2
    loss.backward()

    assert omega.grad is not None, "No gradient on omega"
    assert dose.grad is not None, "No gradient on dose"
    assert torch.isfinite(omega.grad).all(), "Non-finite gradient on omega"
    assert torch.isfinite(dose.grad).all(), "Non-finite gradient on dose"
    assert (omega.grad.abs() > 0).any(), "Zero gradient on omega"
    assert (dose.grad.abs() > 0).any(), "Zero gradient on dose"


def test_joint_differs_from_decoupled():
    """Joint and decoupled produce different results (different design space)."""
    joint = optimize_joint_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_TIME,
        spin_dt=SPIN_DT,
        n_epochs=N_EPOCHS_LONG,
        verbose=False,
    )
    decoupled = optimize_decoupled_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_TIME,
        spin_dt=SPIN_DT,
        n_spin_epochs=N_EPOCHS_LONG // 2,
        n_dose_epochs=N_EPOCHS_LONG // 2,
        verbose=False,
    )
    # With enough epochs, the two strategies should produce different outcomes
    assert joint["opt_dose_mj"] != decoupled["opt_dose_mj"] or \
           joint["final_developed_nm"] != decoupled["final_developed_nm"], \
        "Joint and decoupled produced identical results unexpectedly"


def test_process_window_analysis():
    """Process window analysis returns valid sweep with sensible bounds."""
    omega = torch.full((N_STEPS,), 2500.0 * RPM_TO_RAD)
    dose = torch.tensor(80.0)

    pw = process_window_analysis(
        omega_profile=omega,
        nominal_dose=dose,
        spin_dt=SPIN_DT,
        target_developed_nm=TARGET_NM,
        n_sweep=11,
        dose_range_frac=0.10,
        tolerance_nm=20.0,
    )

    assert len(pw["sweep_results"]) == 11
    for entry in pw["sweep_results"]:
        assert math.isfinite(entry["dose_mj"])
        assert math.isfinite(entry["developed_nm"])
        assert entry["developed_nm"] >= 0.0
    assert pw["process_window_width_mj"] >= 0.0


def test_joint_produces_different_omega_than_decoupled():
    """Joint optimization explores a different spin profile than decoupled."""
    joint = optimize_joint_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_TIME,
        spin_dt=SPIN_DT,
        n_epochs=N_EPOCHS_LONG,
        verbose=False,
    )
    decoupled = optimize_decoupled_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_TIME,
        spin_dt=SPIN_DT,
        n_spin_epochs=N_EPOCHS_LONG // 2,
        n_dose_epochs=N_EPOCHS_LONG // 2,
        verbose=False,
    )
    j_mean = joint["omega_profile"].mean().item()
    d_mean = decoupled["omega_profile"].mean().item()
    assert abs(j_mean - d_mean) > 1e-3, \
        f"Joint and decoupled spin profiles are identical: {j_mean:.6f} vs {d_mean:.6f}"
