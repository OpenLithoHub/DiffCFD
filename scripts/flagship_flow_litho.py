#!/usr/bin/env python
"""Flagship demo: flow-litho co-optimization (joint vs decoupled).

Runs both approaches with a small problem size, compares results, and
writes a JSON report with loss histories and process-window analysis.

Usage:
    python scripts/flagship_flow_litho.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from diffcfd.workflows.joint_litho_opt import (
    optimize_decoupled_process,
    optimize_joint_process,
    process_window_analysis,
)

TARGET_NM = 50.0
TOTAL_SPIN_TIME = 0.5
SPIN_DT = 0.005
N_EPOCHS = 30
# Higher initial wetness and thickness keep the film fluid so spin-profile
# gradients flow meaningfully through the co-design loop.
H0 = 12e-6
C0 = 0.90
INIT_OMEGA_RPM = 1500.0


def _run_joint() -> dict:
    print("=" * 64)
    print("  JOINT CO-DESIGN: spin profile + exposure dose optimized together")
    print("=" * 64)
    t0 = time.perf_counter()
    res = optimize_joint_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_SPIN_TIME,
        spin_dt=SPIN_DT,
        n_epochs=N_EPOCHS,
        lr=0.2,
        lr_dose=1.0,
        verbose=True,
        h0=H0,
        c0=C0,
        init_omega_rpm=INIT_OMEGA_RPM,
    )
    elapsed = time.perf_counter() - t0
    res["wall_time_s"] = elapsed
    print(f"  => Completed in {elapsed:.2f}s\n")
    return res


def _run_decoupled() -> dict:
    print("=" * 64)
    print("  DECOUPLED BASELINE: spin first, then dose sweep")
    print("=" * 64)
    t0 = time.perf_counter()
    res = optimize_decoupled_process(
        target_developed_h_nm=TARGET_NM,
        total_spin_time=TOTAL_SPIN_TIME,
        spin_dt=SPIN_DT,
        n_spin_epochs=N_EPOCHS // 2,
        n_dose_epochs=N_EPOCHS // 2,
        lr_spin=0.2,
        lr_dose=1.0,
        verbose=True,
        h0=H0,
        c0=C0,
        init_omega_rpm=INIT_OMEGA_RPM,
    )
    elapsed = time.perf_counter() - t0
    res["wall_time_s"] = elapsed
    print(f"  => Completed in {elapsed:.2f}s\n")
    return res


def _process_window(omega: torch.Tensor, dose: torch.Tensor, label: str) -> dict:
    print(f"Process window analysis ({label})...")
    pw = process_window_analysis(
        omega_profile=omega,
        nominal_dose=dose,
        spin_dt=SPIN_DT,
        target_developed_nm=TARGET_NM,
        n_sweep=21,
        dose_range_frac=0.10,
        tolerance_nm=10.0,
        h0=H0,
        c0=C0,
    )
    print(
        f"  {label}: window [{pw['process_window_low_mj']:.2f}, "
        f"{pw['process_window_high_mj']:.2f}] mJ/cm2  "
        f"(width={pw['process_window_width_mj']:.2f} mJ/cm2)\n"
    )
    return pw


def _summary(joint: dict, decoupled: dict, pw_joint: dict, pw_decoupled: dict) -> None:
    sep = "-" * 64
    print("=" * 64)
    print("  SUMMARY TABLE: Joint Co-Design vs Decoupled Baseline")
    print("=" * 64)
    print(f"{'Metric':<32s} {'Joint':>14s} {'Decoupled':>14s}")
    print(sep)
    print(
        f"{'Final Developed (nm)':<32s} "
        f"{joint['final_developed_nm']:>14.2f} "
        f"{decoupled['final_developed_nm']:>14.2f}"
    )
    print(
        f"{'Optimal Dose (mJ/cm2)':<32s} "
        f"{joint['opt_dose_mj']:>14.2f} "
        f"{decoupled['opt_dose_mj']:>14.2f}"
    )

    j_final_loss = joint["loss_history"][-1] if joint["loss_history"] else float("nan")
    d_final_loss = decoupled["loss_history"][-1] if decoupled["loss_history"] else float("nan")
    print(
        f"{'Final Loss':<32s} "
        f"{j_final_loss:>14.4e} "
        f"{d_final_loss:>14.4e}"
    )
    print(
        f"{'Process Window Width (mJ/cm2)':<32s} "
        f"{pw_joint['process_window_width_mj']:>14.2f} "
        f"{pw_decoupled['process_window_width_mj']:>14.2f}"
    )
    print(
        f"{'Wall Time (s)':<32s} "
        f"{joint['wall_time_s']:>14.2f} "
        f"{decoupled['wall_time_s']:>14.2f}"
    )
    print(sep)

    j_err = abs(joint["final_developed_nm"] - TARGET_NM)
    d_err = abs(decoupled["final_developed_nm"] - TARGET_NM)
    winner = "Joint" if j_err < d_err else "Decoupled"
    pw_winner = "Joint" if pw_joint["process_window_width_mj"] >= pw_decoupled["process_window_width_mj"] else "Decoupled"
    print(f"  Closer to target: {winner}")
    print(f"  Wider process window: {pw_winner}")
    print()


def _save_report(
    joint: dict, decoupled: dict, pw_joint: dict, pw_decoupled: dict
) -> Path:
    report = {
        "target_nm": TARGET_NM,
        "total_spin_time": TOTAL_SPIN_TIME,
        "spin_dt": SPIN_DT,
        "n_epochs": N_EPOCHS,
        "joint": {
            "final_developed_nm": joint["final_developed_nm"],
            "opt_dose_mj": joint["opt_dose_mj"],
            "loss_history": joint["loss_history"],
            "wall_time_s": joint["wall_time_s"],
            "process_window": {
                "low_mj": pw_joint["process_window_low_mj"],
                "high_mj": pw_joint["process_window_high_mj"],
                "width_mj": pw_joint["process_window_width_mj"],
            },
        },
        "decoupled": {
            "final_developed_nm": decoupled["final_developed_nm"],
            "opt_dose_mj": decoupled["opt_dose_mj"],
            "loss_history": decoupled["loss_history"],
            "wall_time_s": decoupled["wall_time_s"],
            "process_window": {
                "low_mj": pw_decoupled["process_window_low_mj"],
                "high_mj": pw_decoupled["process_window_high_mj"],
                "width_mj": pw_decoupled["process_window_width_mj"],
            },
        },
        "process_window_comparison": {
            "sweep_dose_range_frac": 0.10,
            "tolerance_nm": 10.0,
            "joint_sweep": [
                {"dose_mj": s["dose_mj"], "developed_nm": s["developed_nm"]}
                for s in pw_joint["sweep_results"]
            ],
            "decoupled_sweep": [
                {"dose_mj": s["dose_mj"], "developed_nm": s["developed_nm"]}
                for s in pw_decoupled["sweep_results"]
            ],
        },
    }

    out = Path(__file__).resolve().parent.parent / "flagship_flow_litho_results.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to {out}")
    return out


def main() -> None:
    torch.manual_seed(42)

    joint = _run_joint()
    decoupled = _run_decoupled()

    pw_joint = _process_window(joint["omega_profile"], joint["dose_tensor"], "Joint")
    pw_decoupled = _process_window(decoupled["omega_profile"], decoupled["dose_tensor"], "Decoupled")

    _summary(joint, decoupled, pw_joint, pw_decoupled)
    out_path = _save_report(joint, decoupled, pw_joint, pw_decoupled)

    print(f"Done. Report: {out_path}")


if __name__ == "__main__":
    main()
