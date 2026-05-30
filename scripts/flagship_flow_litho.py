#!/usr/bin/env python
"""Flagship demo: flow-litho co-optimization (joint vs decoupled).

Supports two modes:

1. Single-seed (default): identical behaviour to the original script,
   runs seed=42 once and writes a full JSON report.

2. Seed sweep (``--seed-sweep``): runs 10 seeds (42..51), varies
   initial conditions (h0, c0, init_omega_rpm) per seed via torch RNG,
   aggregates results as mean +/- std, and performs a Wilcoxon
   signed-rank test for joint vs decoupled.

Usage:
    python scripts/flagship_flow_litho.py
    python scripts/flagship_flow_litho.py --seed-sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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

# Nominal initial conditions
H0_NOMINAL = 12e-6
C0_NOMINAL = 0.90
INIT_OMEGA_RPM_NOMINAL = 1500.0

# Perturbation ranges for seed sweep (+/- relative)
H0_PERTURB = 0.08       # +/- 8%
C0_PERTURB = 0.05       # +/- 5%
OMEGA_PERTURB = 0.10    # +/- 10%

SEED_START = 42
SEED_END = 51  # inclusive


class _ResultsEncoder(json.JSONEncoder):
    """Handle numpy/torch types that the default encoder chokes on."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Helpers to perturb initial conditions from a seed
# ---------------------------------------------------------------------------

def _sample_initial_conditions(seed: int) -> dict[str, float]:
    """Draw perturbed h0, c0, init_omega_rpm from a seeded RNG."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    u = torch.rand(3, generator=rng)  # uniform [0, 1)

    h0 = H0_NOMINAL * (1.0 - H0_PERTURB + 2.0 * H0_PERTURB * u[0].item())
    c0 = C0_NOMINAL * (1.0 - C0_PERTURB + 2.0 * C0_PERTURB * u[1].item())
    omega = INIT_OMEGA_RPM_NOMINAL * (
        1.0 - OMEGA_PERTURB + 2.0 * OMEGA_PERTURB * u[2].item()
    )

    return {"h0": h0, "c0": c0, "init_omega_rpm": omega}


# ---------------------------------------------------------------------------
# Single-seed run helpers (accept initial conditions as parameters)
# ---------------------------------------------------------------------------

def _run_joint(
    h0: float = H0_NOMINAL,
    c0: float = C0_NOMINAL,
    init_omega_rpm: float = INIT_OMEGA_RPM_NOMINAL,
) -> dict:
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
        h0=h0,
        c0=c0,
        init_omega_rpm=init_omega_rpm,
    )
    elapsed = time.perf_counter() - t0
    res["wall_time_s"] = elapsed
    print(f"  => Completed in {elapsed:.2f}s\n")
    return res


def _run_decoupled(
    h0: float = H0_NOMINAL,
    c0: float = C0_NOMINAL,
    init_omega_rpm: float = INIT_OMEGA_RPM_NOMINAL,
) -> dict:
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
        h0=h0,
        c0=c0,
        init_omega_rpm=init_omega_rpm,
    )
    elapsed = time.perf_counter() - t0
    res["wall_time_s"] = elapsed
    print(f"  => Completed in {elapsed:.2f}s\n")
    return res


def _process_window(
    omega: torch.Tensor,
    dose: torch.Tensor,
    label: str,
    h0: float = H0_NOMINAL,
    c0: float = C0_NOMINAL,
) -> dict:
    print(f"Process window analysis ({label})...")
    pw = process_window_analysis(
        omega_profile=omega,
        nominal_dose=dose,
        spin_dt=SPIN_DT,
        n_sweep=21,
        dose_range_frac=0.10,
        tolerance_frac=0.02,
        h0=h0,
        c0=c0,
    )
    print(
        f"  {label}: window [{pw['process_window_low_mj']:.2f}, "
        f"{pw['process_window_high_mj']:.2f}] mJ/cm2  "
        f"(width={pw['process_window_width_mj']:.2f} mJ/cm2)\n"
    )
    return pw


def _summary(
    joint: dict, decoupled: dict, pw_joint: dict, pw_decoupled: dict
) -> None:
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
    d_final_loss = (
        decoupled["loss_history"][-1] if decoupled["loss_history"] else float("nan")
    )
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
    pw_winner = (
        "Joint"
        if pw_joint["process_window_width_mj"] >= pw_decoupled["process_window_width_mj"]
        else "Decoupled"
    )
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
        "seed_sweep": False,
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
        json.dump(report, f, indent=2, cls=_ResultsEncoder)
    print(f"Results saved to {out}")
    return out


# ---------------------------------------------------------------------------
# Single-seed trial (with perturbed initial conditions)
# ---------------------------------------------------------------------------

def _single_seed_run(seed: int, verbose: bool = True) -> dict[str, Any]:
    """Run one joint + decoupled trial with seed-perturbed initial conditions."""
    ic = _sample_initial_conditions(seed)

    if verbose:
        print(f"\n{'#' * 64}")
        print(f"  SEED {seed}")
        print(
            f"  h0={ic['h0']*1e6:.2f} um  "
            f"c0={ic['c0']:.4f}  "
            f"omega0={ic['init_omega_rpm']:.0f} RPM"
        )
        print(f"{'#' * 64}")

    torch.manual_seed(seed)

    try:
        joint = _run_joint(h0=ic["h0"], c0=ic["c0"], init_omega_rpm=ic["init_omega_rpm"])
    except Exception as exc:
        if verbose:
            print(f"  WARNING: joint optimization failed for seed {seed}: {exc}")
        joint = None

    try:
        decoupled = _run_decoupled(
            h0=ic["h0"], c0=ic["c0"], init_omega_rpm=ic["init_omega_rpm"]
        )
    except Exception as exc:
        if verbose:
            print(f"  WARNING: decoupled optimization failed for seed {seed}: {exc}")
        decoupled = None

    if joint is None or decoupled is None:
        return {
            "seed": seed,
            "initial_conditions": {
                "h0_um": ic["h0"] * 1e6,
                "c0": ic["c0"],
                "init_omega_rpm": ic["init_omega_rpm"],
            },
            "joint": {
                "final_developed_nm": float("nan"),
                "opt_dose_mj": float("nan"),
                "final_loss": float("nan"),
                "wall_time_s": float("nan"),
                "process_window_width_mj": float("nan"),
                "process_window_low_mj": float("nan"),
                "process_window_high_mj": float("nan"),
            },
            "decoupled": {
                "final_developed_nm": float("nan"),
                "opt_dose_mj": float("nan"),
                "final_loss": float("nan"),
                "wall_time_s": float("nan"),
                "process_window_width_mj": float("nan"),
                "process_window_low_mj": float("nan"),
                "process_window_high_mj": float("nan"),
            },
        }

    pw_joint = _process_window(
        joint["omega_profile"], joint["dose_tensor"], "Joint",
        h0=ic["h0"], c0=ic["c0"],
    )
    pw_decoupled = _process_window(
        decoupled["omega_profile"], decoupled["dose_tensor"], "Decoupled",
        h0=ic["h0"], c0=ic["c0"],
    )

    j_final_loss = joint["loss_history"][-1] if joint["loss_history"] else float("nan")
    d_final_loss = (
        decoupled["loss_history"][-1] if decoupled["loss_history"] else float("nan")
    )

    if verbose:
        _summary(joint, decoupled, pw_joint, pw_decoupled)

    return {
        "seed": seed,
        "initial_conditions": {
            "h0_um": ic["h0"] * 1e6,
            "c0": ic["c0"],
            "init_omega_rpm": ic["init_omega_rpm"],
        },
        "joint": {
            "final_developed_nm": joint["final_developed_nm"],
            "opt_dose_mj": joint["opt_dose_mj"],
            "final_loss": j_final_loss,
            "wall_time_s": joint["wall_time_s"],
            "process_window_width_mj": pw_joint["process_window_width_mj"],
            "process_window_low_mj": pw_joint["process_window_low_mj"],
            "process_window_high_mj": pw_joint["process_window_high_mj"],
        },
        "decoupled": {
            "final_developed_nm": decoupled["final_developed_nm"],
            "opt_dose_mj": decoupled["opt_dose_mj"],
            "final_loss": d_final_loss,
            "wall_time_s": decoupled["wall_time_s"],
            "process_window_width_mj": pw_decoupled["process_window_width_mj"],
            "process_window_low_mj": pw_decoupled["process_window_low_mj"],
            "process_window_high_mj": pw_decoupled["process_window_high_mj"],
        },
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def _std(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    if len(valid) < 2:
        return 0.0
    m = _mean(valid)
    return math.sqrt(sum((v - m) ** 2 for v in valid) / (len(valid) - 1))


def _wilcoxon(x: list[float], y: list[float], label: str) -> dict[str, Any]:
    """Wilcoxon signed-rank test on paired samples (x - y), NaN-filtered."""
    from scipy.stats import wilcoxon

    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    if len(pairs) < 3:
        return {
            "test": "Wilcoxon signed-rank",
            "metric": label,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant_at_005": False,
            "note": f"Only {len(pairs)} valid pairs (need >=3)",
        }

    x_clean, y_clean = zip(*pairs)
    diffs = [a - b for a, b in zip(x_clean, y_clean)]
    if all(d == 0 for d in diffs):
        return {
            "test": "Wilcoxon signed-rank",
            "metric": label,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_005": False,
            "note": "All differences are zero",
        }

    result = wilcoxon(list(x_clean), list(y_clean), alternative="two-sided")
    return {
        "test": "Wilcoxon signed-rank",
        "metric": label,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_005": bool(result.pvalue < 0.05),
    }


# ---------------------------------------------------------------------------
# Seed-sweep mode
# ---------------------------------------------------------------------------

def _run_seed_sweep(seed_start: int, seed_end: int) -> dict[str, Any]:
    seeds = list(range(seed_start, seed_end + 1))
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        trial = _single_seed_run(seed, verbose=True)
        per_seed.append(trial)

    metrics = [
        "final_loss",
        "final_developed_nm",
        "opt_dose_mj",
        "wall_time_s",
        "process_window_width_mj",
    ]
    agg: dict[str, Any] = {}
    for mode in ("joint", "decoupled"):
        agg[mode] = {}
        for m in metrics:
            vals = [t[mode][m] for t in per_seed]
            agg[mode][m] = {
                "values": vals,
                "mean": _mean(vals),
                "std": _std(vals),
            }

    # Wilcoxon signed-rank tests
    wilcoxon_results = []
    for m in metrics:
        j_vals = [t["joint"][m] for t in per_seed]
        d_vals = [t["decoupled"][m] for t in per_seed]
        w = _wilcoxon(j_vals, d_vals, m)
        wilcoxon_results.append(w)

    # Print aggregated summary
    print("\n" + "=" * 80)
    print(
        "  AGGREGATED RESULTS ACROSS %d SEEDS (%d..%d)"
        % (len(seeds), seed_start, seed_end)
    )
    print("=" * 80)

    header = (
        f"{'Metric':<36s} {'Joint (mean +/- std)':>22s} "
        f"{'Decoupled (mean +/- std)':>24s} {'p-value':>10s}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        jm = agg["joint"][m]
        dm = agg["decoupled"][m]
        w = next(wr for wr in wilcoxon_results if wr["metric"] == m)
        sig = "*" if w["significant_at_005"] else " "
        print(
            f"{m:<36s} "
            f"{jm['mean']:>10.4e} +/- {jm['std']:.4e} "
            f"{dm['mean']:>10.4e} +/- {dm['std']:.4e} "
            f"{w['p_value']:>9.4f} {sig}"
        )
    print("-" * len(header))
    print("  * = significant at p < 0.05 (Wilcoxon signed-rank, two-sided)")
    print()

    report = {
        "target_nm": TARGET_NM,
        "total_spin_time": TOTAL_SPIN_TIME,
        "spin_dt": SPIN_DT,
        "n_epochs": N_EPOCHS,
        "seed_sweep": True,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "n_seeds": len(seeds),
        "initial_condition_perturbation": {
            "h0_nominal_um": H0_NOMINAL * 1e6,
            "h0_perturbation_frac": H0_PERTURB,
            "c0_nominal": C0_NOMINAL,
            "c0_perturbation_frac": C0_PERTURB,
            "omega_nominal_rpm": INIT_OMEGA_RPM_NOMINAL,
            "omega_perturbation_frac": OMEGA_PERTURB,
        },
        "aggregated": agg,
        "wilcoxon_tests": wilcoxon_results,
        "per_seed": per_seed,
    }

    return report


def _save_sweep_report(report: dict[str, Any]) -> Path:
    out = Path(__file__).resolve().parent.parent / "flagship_flow_litho_results.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, cls=_ResultsEncoder)
    print(f"Results saved to {out}")
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flagship flow-litho benchmark: joint vs decoupled co-optimization."
    )
    parser.add_argument(
        "--seed-sweep",
        action="store_true",
        help=(
            f"Run a multi-seed sweep (seeds {SEED_START}..{SEED_END}) with "
            "perturbed initial conditions and report aggregated mean +/- std "
            "with Wilcoxon signed-rank tests."
        ),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=SEED_START,
        help=f"First seed in sweep (default: {SEED_START})",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=SEED_END,
        help=f"Last seed in sweep, inclusive (default: {SEED_END})",
    )
    args = parser.parse_args()

    if args.seed_sweep:
        report = _run_seed_sweep(args.seed_start, args.seed_end)
        out_path = _save_sweep_report(report)
        print(f"Done. Seed-sweep report: {out_path}")
    else:
        # Original single-seed behaviour (nominal initial conditions)
        torch.manual_seed(42)
        joint = _run_joint()
        decoupled = _run_decoupled()
        pw_joint = _process_window(
            joint["omega_profile"], joint["dose_tensor"], "Joint"
        )
        pw_decoupled = _process_window(
            decoupled["omega_profile"], decoupled["dose_tensor"], "Decoupled"
        )
        _summary(joint, decoupled, pw_joint, pw_decoupled)
        out_path = _save_report(joint, decoupled, pw_joint, pw_decoupled)
        print(f"Done. Report: {out_path}")


if __name__ == "__main__":
    main()
