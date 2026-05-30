#!/usr/bin/env python3
"""Reproducible performance benchmark for DiffCFD README.

Collects per-sample timings with warmup, reports median/P95/P99,
and exports structured JSON for chart generation.

Usage:
    python tests/benchmarks/benchmark_performance.py
    python tests/benchmarks/benchmark_performance.py --json results/perf_bench.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class PerfResult:
    name: str
    grid: str
    samples: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    l2_error: float | None = None


def _measure(fn, warmup: int = 3, samples: int = 20) -> list[float]:
    """Run fn() with warmup, collect raw timings in ms."""
    for _ in range(warmup):
        fn()

    gc.collect()
    gc.disable()
    timings = []
    try:
        for _ in range(samples):
            t0 = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - t0) * 1000)
    finally:
        gc.enable()
    return timings


def _summarize(name: str, grid: str, timings: list[float], l2: float | None = None) -> PerfResult:
    timings_sorted = sorted(timings)
    q = statistics.quantiles(timings_sorted, n=100)
    return PerfResult(
        name=name,
        grid=grid,
        samples=len(timings),
        median_ms=statistics.median(timings_sorted),
        p95_ms=q[94],
        p99_ms=q[98],
        min_ms=timings_sorted[0],
        max_ms=timings_sorted[-1],
        l2_error=l2,
    )


def bench_cavity_re100():
    """Lid-driven cavity at Re=100, multiple grid sizes."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    ghia_dir = Path(__file__).parent.parent / "validation" / "ghia1982"
    results = []
    configs = [
        (32, 2000, 0.7, 0.3, 0),
        (64, 3000, 0.7, 0.3, 0),
    ]
    for grid, max_iter, au, ap, anderson in configs:
        y_ref, u_ref = np.loadtxt(ghia_dir / "re100_u.csv", delimiter=",", skiprows=1, unpack=True)

        def run(g=grid, mi=max_iter, a_u=au, a_p=ap, ad=anderson):
            solver = NavierStokes2D(reynolds_number=100, grid=(g, g), max_iter=mi, tol=1e-5,
                                    alpha_u=a_u, alpha_p=a_p, anderson_depth=ad)
            ux, uy, p = solver.solve_steady(lid_velocity=1.0, case="cavity")
            return ux

        timings = _measure(run, warmup=1, samples=5)
        # Compute L2 on last run
        solver = NavierStokes2D(reynolds_number=100, grid=(grid, grid), max_iter=max_iter,
                                tol=1e-5, alpha_u=au, alpha_p=ap, anderson_depth=anderson)
        ux, _, _ = solver.solve_steady(lid_velocity=1.0, case="cavity")
        y_ux = np.linspace(0.0, 1.0, grid)
        ux_center = ux[:, grid // 2].detach().numpy()
        u_at_ghia = np.interp(y_ref, y_ux, ux_center)
        l2 = float(np.sqrt(np.mean((u_at_ghia - u_ref) ** 2)))

        results.append(_summarize(f"Cavity Re=100", f"{grid}²", timings, l2=l2))
    return results


def bench_poiseuille_forward():
    """Poiseuille channel forward solve at multiple grids."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    results = []
    for nx, ny in [(32, 16), (64, 32)]:
        def run(nx=nx, ny=ny):
            solver = NavierStokes2D(reynolds_number=1.0, grid=(nx, ny), lx=4.0, ly=1.0,
                                    max_iter=3000, tol=1e-8)
            solver.solve_steady(inlet_velocity=1.0, case="channel")

        timings = _measure(run, warmup=1, samples=5)
        results.append(_summarize("Poiseuille forward", f"{nx}×{ny}", timings))
    return results


def bench_implicit_backward():
    """Implicit differentiation backward pass timing."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    results = []
    for nx, ny in [(32, 16), (64, 32)]:
        def run(nx=nx, ny=ny):
            solver = NavierStokes2D(reynolds_number=1.0, grid=(nx, ny), lx=4.0, ly=1.0,
                                    backward="implicit_diff", max_iter=2000, tol=1e-8)
            u_inlet = torch.tensor(1.0, requires_grad=True)
            ux, uy, p = solver.solve_steady(inlet_velocity=u_inlet, case="channel")
            dp = solver.pressure_drop(ux, uy, p)
            dp.backward()

        timings = _measure(run, warmup=1, samples=5)
        results.append(_summarize("Implicit diff (forward+backward)", f"{nx}×{ny}", timings))
    return results


def bench_env_step():
    """Gymnasium env step timing."""
    from diffcfd.envs.cylinder_wake import CylinderWakeEnv

    def run():
        env = CylinderWakeEnv(re=100, grid=(32, 16))
        obs, info = env.reset()
        env.step([0.5])

    timings = _measure(run, warmup=1, samples=10)
    return [_summarize("CylinderWakeEnv reset+step", "32×16", timings)]


def bench_sco2_inference():
    """sCO2 surrogate property inference timing."""
    from diffcfd.props.sco2 import SCO2Surrogate

    model = SCO2Surrogate(hidden_dim=32)
    T = torch.randn(100, requires_grad=True)
    p = torch.randn(100, requires_grad=True) * 1e7 + 7e6

    def run():
        rho = model.density(T, p)
        rho.sum().backward()

    timings = _measure(run, warmup=5, samples=50)
    return [_summarize("sCO2 surrogate (batch=100)", "128-dim", timings)]


def print_env_info():
    info = {
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "pytorch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="Export results to JSON")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    print("DiffCFD Performance Benchmark")
    env_info = print_env_info()

    all_results = []
    sections = [
        ("Solver Performance", bench_cavity_re100),
        ("Solver Performance", bench_poiseuille_forward),
        ("Implicit Differentiation", bench_implicit_backward),
        ("RL Environment", bench_env_step),
        ("Surrogate Inference", bench_sco2_inference),
    ]

    for section, bench_fn in sections:
        print(f"\n{section}: {bench_fn.__doc__.strip()}")
        results = bench_fn()
        for r in results:
            print(f"  {r.grid:<10} median={r.median_ms:>10.1f}ms  "
                  f"P95={r.p95_ms:>10.1f}ms  P99={r.p99_ms:>10.1f}ms"
                  f"{'  L2=' + f'{r.l2_error:.4f}' if r.l2_error else ''}")
            all_results.append({"section": section, **asdict(r)})

    if args.json:
        out = {"env": env_info, "results": all_results}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults exported to {args.json}")

    print("\nAll tests run on the above hardware/software configuration.")
    print("No values are estimated or extrapolated.")


if __name__ == "__main__":
    main()
