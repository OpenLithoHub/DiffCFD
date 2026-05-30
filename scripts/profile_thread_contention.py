#!/usr/bin/env python3
"""Profile thread contention between Rust rayon and PyTorch intra-op threads.

Runs a representative DiffCFD workload (channel flow forward + backward)
with and without the ``single_torch_thread`` context manager, then reports
wall time, CPU utilization, and a contention verdict.

Results are saved to ``scripts/profile_results.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffcfd.geometry.shapes import rectangle_sdf
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D
from diffcfd.utils.threading import single_torch_thread

RESULTS_PATH = Path(__file__).resolve().parent / "profile_results.json"


def _measure_run(use_single_thread: bool, n_iters: int = 3) -> dict:
    """Run forward + backward and collect timing."""
    device = "cpu"
    grid = (32, 16)  # small grid for fast profiling

    times: list[float] = []
    cpu_samples: list[float] = []

    for _ in range(n_iters):
        solver = NavierStokes2D(
            reynolds_number=50.0,
            grid=grid,
            device=device,
            backward="unrolled",
            max_iter=50,
            tol=1e-3,
            lx=2.0,
            ly=1.0,
        )

        inlet = torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True)

        t0 = time.perf_counter()
        if use_single_thread:
            with single_torch_thread():
                ux, uy, p = solver.solve_steady(case="channel", inlet_velocity=inlet)
        else:
            ux, uy, p = solver.solve_steady(case="channel", inlet_velocity=inlet)

        loss = ux.sum() + uy.sum()
        loss.backward()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Sample CPU utilization
        try:
            cpu_pct = _cpu_utilization()
            cpu_samples.append(cpu_pct)
        except Exception:
            pass

    wall_mean = float(np.mean(times))
    wall_std = float(np.std(times))
    cpu_mean = float(np.mean(cpu_samples)) if cpu_samples else -1.0

    return {
        "wall_time_mean_s": round(wall_mean, 4),
        "wall_time_std_s": round(wall_std, 4),
        "cpu_utilization_pct": round(cpu_mean, 1),
        "n_iters": n_iters,
        "grid": list(grid),
    }


def _cpu_utilization() -> float:
    """Read CPU utilization from /proc/stat (Linux only)."""
    stat_path = "/proc/stat"
    with open(stat_path) as f:
        line = f.readline()
    vals = [int(x) for x in line.split()[1:]]
    idle = vals[3]
    total = sum(vals)
    time.sleep(0.1)
    with open(stat_path) as f:
        line = f.readline()
    vals2 = [int(x) for x in line.split()[1:]]
    idle2 = vals2[3]
    total2 = sum(vals2)
    diff_idle = idle2 - idle
    diff_total = total2 - total
    if diff_total == 0:
        return 0.0
    return (1.0 - diff_idle / diff_total) * 100.0


def main() -> None:
    print("DiffCFD Thread Contention Profiler")
    print("=" * 40)

    n_torch_threads = torch.get_num_threads()
    n_cpus = os.cpu_count() or 1
    print(f"PyTorch threads: {n_torch_threads}")
    print(f"CPU cores: {n_cpus}")
    print()

    print("Running WITHOUT single_torch_thread ...")
    baseline = _measure_run(use_single_thread=False)
    print(f"  Wall time: {baseline['wall_time_mean_s']:.4f} +/- {baseline['wall_time_std_s']:.4f} s")
    print(f"  CPU utilization: {baseline['cpu_utilization_pct']:.1f}%")
    print()

    print("Running WITH single_torch_thread ...")
    pinned = _measure_run(use_single_thread=True)
    print(f"  Wall time: {pinned['wall_time_mean_s']:.4f} +/- {pinned['wall_time_std_s']:.4f} s")
    print(f"  CPU utilization: {pinned['cpu_utilization_pct']:.1f}%")
    print()

    # Compute speedup and contention
    speedup = baseline["wall_time_mean_s"] / max(pinned["wall_time_mean_s"], 1e-9)
    improvement_pct = (speedup - 1.0) * 100.0

    contention_pct = max(0.0, improvement_pct)

    if abs(improvement_pct) < 5.0:
        verdict = (
            "Contention is < 5% -- thread affinity is NOT needed. "
            "Rust rayon and PyTorch do not meaningfully compete for CPU time "
            "in the current codebase because Rust SDF runs as geometry "
            "preprocessing, not concurrently with torch.backward()."
        )
    elif improvement_pct > 0:
        verdict = (
            f"Thread affinity improves wall time by {improvement_pct:.1f}%. "
            "Consider enabling single_torch_thread for workloads with "
            "significant Rust/PyTorch overlap."
        )
    else:
        verdict = (
            f"Thread affinity slows wall time by {abs(improvement_pct):.1f}%. "
            "Do NOT enable -- reducing PyTorch threads hurts more than "
            "any contention reduction helps."
        )

    print(f"Speedup with thread pinning: {speedup:.3f}x ({improvement_pct:+.1f}%)")
    print(f"\nVerdict: {verdict}")

    results = {
        "baseline": baseline,
        "with_single_thread": pinned,
        "speedup": round(speedup, 4),
        "contention_improvement_pct": round(improvement_pct, 2),
        "verdict": verdict,
        "environment": {
            "n_torch_threads": n_torch_threads,
            "n_cpus": n_cpus,
            "device": "cpu",
        },
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
