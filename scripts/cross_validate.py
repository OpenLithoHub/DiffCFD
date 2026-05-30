#!/usr/bin/env python
"""Cross-validation of DiffCFD solvers against analytical solutions.

Compares solver output to known closed-form results:
1. Lid-driven cavity (Re=100) vs Ghia et al. 1982 reference data
2. Poiseuille flow gradient vs analytical dDP/dU_inlet

Outputs a JSON table of pass/fail results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

GHIA_DIR = Path(__file__).resolve().parent.parent / "tests" / "validation" / "ghia1982"


def _load_ghia(filename: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(GHIA_DIR / filename, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def validate_lid_driven_cavity_re100() -> dict:
    """Lid-driven cavity Re=100 on 64x64 grid vs Ghia 1982."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    y_ref, u_ref = _load_ghia("re100_u.csv")
    nx = ny = 64

    t0 = time.perf_counter()
    solver = NavierStokes2D(
        reynolds_number=100,
        grid=(nx, ny),
        device="cpu",
        max_iter=2000,
        tol=1e-5,
        alpha_u=0.7,
        alpha_p=0.3,
    )
    ux, _uy, _p = solver.solve_steady(
        sdf=None, inlet_velocity=0.0, lid_velocity=1.0, case="cavity"
    )
    elapsed = time.perf_counter() - t0

    ux_center = ux[:, ny // 2].detach().numpy()
    y_ux = np.linspace(0.0, 1.0, ny)
    u_at_ghia = interp1d(y_ux, ux_center, kind="linear")(y_ref)
    l2 = float(np.sqrt(np.mean((u_at_ghia - u_ref) ** 2)))

    gate = 0.01
    passed = l2 < gate

    return {
        "test": "lid_driven_cavity_re100",
        "reference": "Ghia et al. 1982",
        "grid": f"{nx}x{ny}",
        "l2_error": round(l2, 6),
        "l2_error_pct": f"{l2 * 100:.3f}%",
        "gate_pct": f"{gate * 100:.1f}%",
        "passed": passed,
        "wall_time_s": round(elapsed, 2),
    }


def validate_poiseuille_forward() -> dict:
    """Poiseuille channel Re=1 outlet profile vs analytical parabolic."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    grid = (32, 16)
    lx, ly = 4.0, 1.0

    t0 = time.perf_counter()
    solver = NavierStokes2D(reynolds_number=1.0, grid=grid, lx=lx, ly=ly)
    ux, _uy, _pf = solver.solve_steady(
        sdf=None, inlet_velocity=1.0, case="channel"
    )
    elapsed = time.perf_counter() - t0

    ny = grid[1]
    dy = ly / ny
    h_eff = (ny - 1) * dy
    y = torch.arange(ny, dtype=torch.float32) * dy

    u_outlet = ux[:, -1]
    q = (u_outlet[1:-1].sum() * dy).item()
    u_mean = q / h_eff

    u_analytical = 6.0 * u_mean * (y / h_eff) * (1.0 - y / h_eff)
    l2_err = (torch.norm(u_outlet - u_analytical) / torch.norm(u_analytical)).item()

    gate = 0.01
    passed = l2_err < gate

    return {
        "test": "poiseuille_forward_re1",
        "reference": "Analytical parabolic profile",
        "grid": f"{grid[0]}x{grid[1]}",
        "l2_error": round(l2_err, 6),
        "l2_error_pct": f"{l2_err * 100:.3f}%",
        "gate_pct": f"{gate * 100:.1f}%",
        "passed": passed,
        "wall_time_s": round(elapsed, 2),
    }


def validate_poiseuille_gradient() -> dict:
    """Poiseuille dDP/dU_inlet: implicit-diff vs finite difference."""
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    grid = (32, 16)
    lx, ly = 4.0, 1.0
    re = 1.0
    u_inlet_val = 1.0
    eps = 0.01

    t0 = time.perf_counter()

    solver_fd = NavierStokes2D(
        reynolds_number=re, grid=grid, lx=lx, ly=ly, tol=1e-8
    )
    ux_p, uy_p, p_p = solver_fd._run_simple(
        None, inlet_velocity=u_inlet_val + eps, case="channel"
    )
    ux_m, uy_m, p_m = solver_fd._run_simple(
        None, inlet_velocity=u_inlet_val - eps, case="channel"
    )
    dp_p = solver_fd.pressure_drop(ux_p, uy_p, p_p)
    dp_m = solver_fd.pressure_drop(ux_m, uy_m, p_m)
    fd_grad = float((dp_p - dp_m) / (2 * eps))

    u_inlet = torch.tensor(u_inlet_val, dtype=torch.float64, requires_grad=True)
    solver_id = NavierStokes2D(
        reynolds_number=re, grid=grid, lx=lx, ly=ly,
        backward="implicit_diff", tol=1e-8,
    )
    ux, uy, pf = solver_id.solve_steady(
        sdf=None, inlet_velocity=u_inlet, case="channel"
    )
    dp = solver_id.pressure_drop(ux, uy, pf)
    dp.backward()

    computed_grad = u_inlet.grad.item()
    rel_err = abs(computed_grad - fd_grad) / abs(fd_grad)
    elapsed = time.perf_counter() - t0

    gate = 1e-4
    passed = rel_err < gate

    return {
        "test": "poiseuille_gradient_implicit_diff",
        "reference": f"Finite difference (eps={eps})",
        "grid": f"{grid[0]}x{grid[1]}",
        "fd_gradient": round(fd_grad, 6),
        "ad_gradient": round(computed_grad, 6),
        "relative_error": f"{rel_err:.2e}",
        "gate": f"{gate:.0e}",
        "passed": passed,
        "wall_time_s": round(elapsed, 2),
    }


def main() -> None:
    results: list[dict] = []
    all_passed = True

    validators = [
        ("Lid-driven cavity Re=100 vs Ghia 1982", validate_lid_driven_cavity_re100),
        ("Poiseuille forward vs analytical", validate_poiseuille_forward),
        ("Poiseuille gradient implicit-diff vs FD", validate_poiseuille_gradient),
    ]

    print("=" * 72)
    print("  DiffCFD Cross-Validation Against Analytical Solutions")
    print("=" * 72)

    for label, fn in validators:
        print(f"\nRunning: {label} ...", end=" ", flush=True)
        try:
            result = fn()
        except Exception as exc:
            print(f"FAIL ({exc})")
            results.append({
                "test": label,
                "passed": False,
                "error": str(exc),
            })
            all_passed = False
            continue

        status = "PASS" if result["passed"] else "FAIL"
        print(status)
        results.append(result)
        if not result["passed"]:
            all_passed = False

        for key in ("l2_error_pct", "gate_pct", "relative_error", "fd_gradient", "ad_gradient"):
            if key in result:
                print(f"  {key}: {result[key]}")

    out_path = Path(__file__).resolve().parent.parent / "cross_validation_results.json"
    report = {
        "all_passed": all_passed,
        "n_tests": len(results),
        "n_passed": sum(1 for r in results if r["passed"]),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"  {report['n_passed']}/{report['n_tests']} tests passed")
    print(f"  Results saved to {out_path}")
    print(f"{'=' * 72}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
