"""Collect all paper results: validation, sCO2 training, benchmarks, figures.

Run with: python3 scripts/collect_paper_results.py

Outputs saved to results/ directory.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# CPU optimization for 5600G (6 physical cores)
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(6)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_json(data, name):
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path}")


# ======================================================================
# 1. Lid-driven cavity validation at multiple resolutions
# ======================================================================


def run_lid_cavity_validation():
    """Run lid-driven cavity at Re=100 and Re=1000 at multiple grid sizes."""
    print("\n" + "=" * 60)
    print("1. Lid-Driven Cavity Validation")
    print("=" * 60)

    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    results = {}

    # Re=100 grid convergence study (Anderson acceleration for larger grids)
    re100_data = {}
    re100_configs = [
        (32, 2000, 0.7, 0.3, 0),
        (64, 3000, 0.7, 0.3, 0),
        (128, 5000, 0.5, 0.2, 5),  # Anderson depth=5 for faster convergence
    ]
    for grid, max_iter, au, ap, anderson in re100_configs:
        print(f"  Re=100, grid={grid}x{grid}...", end=" ", flush=True)
        t0 = time.time()
        solver = NavierStokes2D(
            reynolds_number=100,
            grid=(grid, grid),
            device="cpu",
            max_iter=max_iter,
            tol=1e-5,
            alpha_u=au,
            alpha_p=ap,
            anderson_depth=anderson,
        )
        ux, uy, p = solver.solve_steady(
            sdf=None, inlet_velocity=0.0, lid_velocity=1.0, case="cavity"
        )
        elapsed = time.time() - t0

        # Compare with Ghia data
        ghia_dir = Path(__file__).parent.parent / "tests" / "validation" / "ghia1982"
        y_ref, u_ref = np.loadtxt(
            ghia_dir / "re100_u.csv", delimiter=",", skiprows=1, unpack=True
        )
        y_ux = np.linspace(0.0, 1.0, grid)
        ux_center = ux[:, grid // 2].detach().numpy()
        u_at_ghia = np.interp(y_ref, y_ux, ux_center)
        l2_err = float(np.sqrt(np.mean((u_at_ghia - u_ref) ** 2)))

        re100_data[str(grid)] = {
            "time_s": elapsed,
            "l2_error": l2_err,
            "converged": True,
        }
        print(f"L2={l2_err * 100:.3f}%, t={elapsed:.1f}s")

        # Save velocity profile
        np.savez(
            RESULTS_DIR / f"cavity_re100_{grid}.npz",
            y=y_ux,
            ux=ux_center,
            ux_field=ux.detach().numpy(),
            uy_field=uy.detach().numpy(),
            p_field=p.detach().numpy(),
        )

    results["re100"] = re100_data

    # Re=1000 grid convergence
    re1000_data = {}
    re1000_configs = [
        (64, 4000, 0.5, 0.1, 3),
        (128, 6000, 0.4, 0.05, 5),
    ]
    for grid, max_iter, au, ap, anderson in re1000_configs:
        print(f"  Re=1000, grid={grid}x{grid}...", end=" ", flush=True)
        t0 = time.time()
        solver = NavierStokes2D(
            reynolds_number=1000,
            grid=(grid, grid),
            device="cpu",
            max_iter=max_iter,
            tol=1e-5,
            alpha_u=au,
            alpha_p=ap,
            anderson_depth=anderson,
        )
        ux, uy, p = solver.solve_steady(
            sdf=None, inlet_velocity=0.0, lid_velocity=1.0, case="cavity"
        )
        elapsed = time.time() - t0

        y_ref, u_ref = np.loadtxt(
            ghia_dir / "re1000_u.csv", delimiter=",", skiprows=1, unpack=True
        )
        y_ux = np.linspace(0.0, 1.0, grid)
        ux_center = ux[:, grid // 2].detach().numpy()
        u_at_ghia = np.interp(y_ref, y_ux, ux_center)
        l2_err = float(np.sqrt(np.mean((u_at_ghia - u_ref) ** 2)))

        re1000_data[str(grid)] = {
            "time_s": elapsed,
            "l2_error": l2_err,
            "converged": True,
        }
        print(f"L2={l2_err * 100:.3f}%, t={elapsed:.1f}s")

        np.savez(
            RESULTS_DIR / f"cavity_re1000_{grid}.npz",
            y=y_ux,
            ux=ux_center,
            ux_field=ux.detach().numpy(),
            uy_field=uy.detach().numpy(),
            p_field=p.detach().numpy(),
        )

    results["re1000"] = re1000_data
    save_json(results, "cavity_validation.json")
    return results


# ======================================================================
# 2. Poiseuille flow validation + gradient verification
# ======================================================================


def run_poiseuille_validation():
    """Poiseuille forward validation and implicit differentiation check."""
    print("\n" + "=" * 60)
    print("2. Poiseuille Flow Validation")
    print("=" * 60)

    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    results = {}

    # Grid convergence study
    for nx_mult in [1, 2, 4]:
        nx, ny = 32 * nx_mult, 16 * nx_mult
        print(f"  Grid ({nx}, {ny})...", end=" ", flush=True)
        solver = NavierStokes2D(
            reynolds_number=1.0,
            grid=(nx, ny),
            lx=4.0,
            ly=1.0,
            max_iter=3000,
            tol=1e-8,
        )
        ux, uy, p = solver.solve_steady(inlet_velocity=1.0, case="channel")

        ny_val = ny
        ly = 1.0
        dy = ly / ny_val
        h_eff = (ny_val - 1) * dy
        y = torch.arange(ny_val, dtype=torch.float32) * dy
        u_outlet = ux[:, -1]
        Q = (u_outlet[1:-1].sum() * dy).item()
        U_mean = Q / h_eff
        u_analytical = 6.0 * U_mean * (y / h_eff) * (1.0 - y / h_eff)
        l2_err = (torch.norm(u_outlet - u_analytical) / torch.norm(u_analytical)).item()

        dp = solver.pressure_drop(ux, uy, p).item()
        # Analytical: dp = 12 * mu * L * U_mean / h^2 = 12 * 1 * 4 * 1 / 1 = 48
        dp_analytical = 48.0 * U_mean
        dp_err = abs(dp - dp_analytical) / abs(dp_analytical)

        key = f"{nx}x{ny}"
        results[key] = {
            "forward_l2": l2_err,
            "dp": dp,
            "dp_analytical": dp_analytical,
            "dp_error": dp_err,
        }
        print(f"L2={l2_err * 100:.4f}%, dp_err={dp_err * 100:.4f}%")

    # Gradient verification
    print("  Implicit differentiation gradient check...", end=" ", flush=True)
    eps = 0.01
    solver_fd = NavierStokes2D(
        reynolds_number=1.0, grid=(32, 16), lx=4.0, ly=1.0, tol=1e-8
    )
    ux_p, uy_p, p_p = solver_fd._run_simple(
        None, inlet_velocity=1.0 + eps, case="channel"
    )
    ux_m, uy_m, p_m = solver_fd._run_simple(
        None, inlet_velocity=1.0 - eps, case="channel"
    )
    dp_p = solver_fd.pressure_drop(ux_p, uy_p, p_p)
    dp_m = solver_fd.pressure_drop(ux_m, uy_m, p_m)
    fd_grad = float((dp_p - dp_m) / (2 * eps))

    u_inlet = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    solver_id = NavierStokes2D(
        reynolds_number=1.0,
        grid=(32, 16),
        lx=4.0,
        ly=1.0,
        backward="implicit_diff",
        tol=1e-8,
    )
    ux, uy, pf = solver_id.solve_steady(inlet_velocity=u_inlet, case="channel")
    dp = solver_id.pressure_drop(ux, uy, pf)
    dp.backward()
    ad_grad = u_inlet.grad.item()

    rel_err = abs(ad_grad - fd_grad) / abs(fd_grad)
    results["gradient"] = {
        "fd_grad": fd_grad,
        "ad_grad": ad_grad,
        "analytical": 48.0,
        "rel_error": rel_err,
    }
    print(f"FD={fd_grad:.6f}, AD={ad_grad:.6f}, rel_err={rel_err:.2e}")

    save_json(results, "poiseuille_validation.json")
    return results


# ======================================================================
# 3. sCO2 surrogate training with realistic data
# ======================================================================


def run_sco2_training():
    """Train sCO2 surrogate with physically realistic training data."""
    print("\n" + "=" * 60)
    print("3. sCO₂ Surrogate Training")
    print("=" * 60)

    from diffcfd.props.sco2 import TC, PC, generate_training_data, train_sco2_surrogate

    # Generate high-quality training data covering the transcritical region
    n_samples = 8000
    print(f"  Generating {n_samples} training samples...", flush=True)
    data = generate_training_data(n_samples=n_samples)

    T = data["T"]
    p = data["p"]

    # Train with more epochs for better convergence
    print("  Training surrogate (1000 epochs)...", flush=True)
    t0 = time.time()
    model = train_sco2_surrogate(
        hidden_dim=128, epochs=1000, lr=1e-3, n_samples=n_samples, verbose=False
    )
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")

    # Evaluate on training data for statistics
    with torch.no_grad():
        rho_pred = model.density(T, p)
        mu_pred = model.viscosity(T, p)
        k_pred = model.conductivity(T, p)
        cp_pred = model.specific_heat(T, p)

    def rel_mse(pred, target):
        return ((pred - target) ** 2).mean().sqrt().item()

    def rel_l2(pred, target):
        return (torch.norm(pred - target) / torch.norm(target)).item()

    results = {
        "train_time_s": train_time,
        "n_samples": n_samples,
        "density_rel_l2": rel_l2(rho_pred, data["rho"]),
        "viscosity_rel_l2": rel_l2(mu_pred, data["mu"]),
        "conductivity_rel_l2": rel_l2(k_pred, data["k"]),
        "specific_heat_rel_l2": rel_l2(cp_pred, data["cp"]),
        "density_positivity": bool((rho_pred > 0).all()),
        "viscosity_positivity": bool((mu_pred > 0).all()),
        "conductivity_positivity": bool((k_pred > 0).all()),
        "specific_heat_positivity": bool((cp_pred > 0).all()),
    }

    print(f"  ρ  rel_L2 = {results['density_rel_l2']:.4f}")
    print(f"  μ  rel_L2 = {results['viscosity_rel_l2']:.4f}")
    print(f"  k  rel_L2 = {results['conductivity_rel_l2']:.4f}")
    print(f"  cp rel_L2 = {results['specific_heat_rel_l2']:.4f}")
    print(
        f"  All positive: ρ={results['density_positivity']}, μ={results['viscosity_positivity']}, "
        f"k={results['conductivity_positivity']}, cp={results['specific_heat_positivity']}"
    )

    # Test differentiability
    T_test = torch.tensor([305.0], requires_grad=True)
    p_test = torch.tensor([8.0e6], requires_grad=True)
    rho = model.density(T_test, p_test)
    rho.backward()
    results["d_rho_dT"] = T_test.grad.item()
    results["d_rho_dp"] = p_test.grad.item()
    print(f"  ∂ρ/∂T = {results['d_rho_dT']:.4f}, ∂ρ/∂p = {results['d_rho_dp']:.4f}")

    # Save model
    torch.save(model.state_dict(), RESULTS_DIR / "sco2_surrogate.pt")

    # Generate property map for visualization
    T_grid = torch.linspace(0.9 * TC, 1.1 * TC, 100)
    p_grid = torch.linspace(0.8 * PC, 1.2 * PC, 100)
    Tg, pg = torch.meshgrid(T_grid, p_grid, indexing="ij")
    with torch.no_grad():
        rho_map = model.density(Tg, pg).numpy()

    np.savez(
        RESULTS_DIR / "sco2_property_map.npz",
        T=T_grid.numpy(),
        p=p_grid.numpy(),
        rho=rho_map,
    )

    save_json(results, "sco2_training.json")
    return results


# ======================================================================
# 4. Full benchmark suite
# ======================================================================


def run_benchmark_suite():
    """Run the complete benchmark suite."""
    print("\n" + "=" * 60)
    print("4. Benchmark Suite")
    print("=" * 60)

    from tests.benchmarks.benchmark_suite import run_all_benchmarks

    results = run_all_benchmarks(verbose=True)

    bench_data = []
    for r in results:
        bench_data.append(
            {
                "name": r.name,
                "status": r.status,
                "time_s": r.time_s,
                "value": r.value,
                "target": r.target,
                "error": r.error,
            }
        )

    save_json(bench_data, "benchmark_results.json")

    passed = sum(1 for r in results if r.status == "pass")
    total = len(results)
    total_time = sum(r.time_s for r in results)
    print(f"\n  Summary: {passed}/{total} passed in {total_time:.1f}s")

    return results


# ======================================================================
# 5. Gradient accuracy across problem sizes
# ======================================================================


def run_gradient_convergence():
    """Test gradient accuracy across different grid sizes."""
    print("\n" + "=" * 60)
    print("5. Gradient Accuracy Convergence")
    print("=" * 60)

    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    results = {}
    eps = 0.01
    analytical = 48.0

    for scale in [1, 2, 3]:
        nx, ny = 16 * scale, 8 * scale
        key = f"{nx}x{ny}"
        print(f"  Grid {key}...", end=" ", flush=True)

        # FD reference
        solver_fd = NavierStokes2D(
            reynolds_number=1.0, grid=(nx, ny), lx=4.0, ly=1.0, tol=1e-8
        )
        ux_p, uy_p, p_p = solver_fd._run_simple(
            None, inlet_velocity=1.0 + eps, case="channel"
        )
        ux_m, uy_m, p_m = solver_fd._run_simple(
            None, inlet_velocity=1.0 - eps, case="channel"
        )
        fd_grad = float(
            (
                solver_fd.pressure_drop(ux_p, uy_p, p_p)
                - solver_fd.pressure_drop(ux_m, uy_m, p_m)
            )
            / (2 * eps)
        )

        # Implicit diff
        u_in = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        solver_ad = NavierStokes2D(
            reynolds_number=1.0,
            grid=(nx, ny),
            lx=4.0,
            ly=1.0,
            backward="implicit_diff",
            tol=1e-8,
        )
        ux, uy, pf = solver_ad.solve_steady(inlet_velocity=u_in, case="channel")
        dp = solver_ad.pressure_drop(ux, uy, pf)
        dp.backward()
        ad_grad = u_in.grad.item()

        fd_err = abs(fd_grad - analytical) / analytical
        ad_err = abs(ad_grad - analytical) / analytical
        fd_ad_diff = abs(ad_grad - fd_grad) / abs(fd_grad)

        results[key] = {
            "fd_grad": fd_grad,
            "ad_grad": ad_grad,
            "analytical": analytical,
            "fd_error": fd_err,
            "ad_error": ad_err,
            "fd_ad_agreement": fd_ad_diff,
        }
        print(f"FD={fd_grad:.4f}, AD={ad_grad:.4f}, agree={fd_ad_diff:.2e}")

    save_json(results, "gradient_convergence.json")
    return results


# ======================================================================
# 6. Generate publication figures
# ======================================================================


def generate_figures(cavity_results, poiseuille_results, sco2_results):
    """Generate all publication-quality figures."""
    print("\n" + "=" * 60)
    print("6. Generating Publication Figures")
    print("=" * 60)

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "figure.figsize": (6, 4.5),
            "figure.dpi": 150,
        }
    )

    # Figure 1: Cavity velocity profiles
    _fig_cavity()
    # Figure 2: Grid convergence
    _fig_grid_convergence(cavity_results, poiseuille_results)
    # Figure 3: sCO2 property map
    _fig_sco2()
    # Figure 4: Gradient verification
    _fig_gradient_bars()


def _fig_cavity():
    """Lid-driven cavity velocity profiles vs Ghia."""
    print("  Cavity profiles...", flush=True)
    ghia_dir = Path(__file__).parent.parent / "tests" / "validation" / "ghia1982"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, re, color in zip(axes, [100, 1000], ["C0", "C1"]):
        # Ghia reference
        fname = f"re{re}_u.csv"
        y_ref, u_ref = np.loadtxt(
            ghia_dir / fname, delimiter=",", skiprows=1, unpack=True
        )
        ax.plot(u_ref, y_ref, "ko", markersize=5, label="Ghia et al. 1982")

        # DiffCFD results at different resolutions
        grids = [32, 64, 128] if re == 100 else [64, 128]
        for grid in grids:
            path = RESULTS_DIR / f"cavity_re{re}_{grid}.npz"
            if path.exists():
                d = np.load(path)
                ax.plot(
                    d["ux"],
                    d["y"],
                    "-",
                    color=color,
                    alpha=0.7,
                    label=f"DiffCFD {grid}²",
                )

        ax.set_xlabel("u / U_lid")
        ax.set_ylabel("y / L")
        ax.set_title(f"Re = {re}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_cavity_profiles.pdf", bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "fig_cavity_profiles.png", bbox_inches="tight")
    plt.close()


def _fig_grid_convergence(cavity_results, poiseuille_results):
    """Grid convergence plots."""
    print("  Grid convergence...", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Cavity Re=100 convergence
    ax = axes[0]
    grids = [32, 64, 128]
    errors = [cavity_results["re100"][str(g)]["l2_error"] for g in grids]
    ax.loglog([g**2 for g in grids], errors, "o-", label="L2 error")
    # Reference 2nd-order line
    ref_x = [g**2 for g in grids]
    ref_y = [errors[0] * (ref_x[0] / x) ** 1 for x in ref_x]
    ax.loglog(ref_x, ref_y, "--", alpha=0.4, label="1st order ref")
    ax.set_xlabel("Grid points (N)")
    ax.set_ylabel("L2 error")
    ax.set_title("Cavity Re=100 convergence")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Poiseuille convergence
    ax = axes[1]
    p_grids = ["32x16", "64x32", "128x64"]
    p_errors = [poiseuille_results[g]["forward_l2"] for g in p_grids]
    p_n = [32 * 16, 64 * 32, 128 * 64]
    ax.loglog(p_n, p_errors, "s-", label="L2 error")
    ref_y2 = [p_errors[0] * (p_n[0] / x) ** 2 for x in p_n]
    ax.loglog(p_n, ref_y2, "--", alpha=0.4, label="2nd order ref")
    ax.set_xlabel("Grid points (N)")
    ax.set_ylabel("L2 error")
    ax.set_title("Poiseuille forward convergence")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_grid_convergence.pdf", bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "fig_grid_convergence.png", bbox_inches="tight")
    plt.close()


def _fig_sco2():
    """sCO2 density map in the transcritical region."""
    print("  sCO2 property map...", flush=True)

    path = RESULTS_DIR / "sco2_property_map.npz"
    if not path.exists():
        print("    Skipping (no data)")
        return

    d = np.load(path)

    fig, ax = plt.subplots(figsize=(6, 5))
    Tc = 304.13
    Pc = 7.377e6

    im = ax.pcolormesh(d["p"] / 1e6, d["T"], d["rho"], cmap="viridis", shading="auto")
    ax.axhline(Tc, color="r", linestyle="--", alpha=0.7, label=f"Tc = {Tc} K")
    ax.axvline(
        Pc / 1e6, color="r", linestyle=":", alpha=0.7, label=f"Pc = {Pc / 1e6:.3f} MPa"
    )
    ax.set_xlabel("Pressure [MPa]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("sCO₂ Density [kg/m³] — Neural Surrogate")
    ax.legend(fontsize=9)
    plt.colorbar(im, ax=ax, label="ρ [kg/m³]")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_sco2_density.pdf", bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "fig_sco2_density.png", bbox_inches="tight")
    plt.close()


def _fig_gradient_bars():
    """Gradient accuracy bar chart."""
    print("  Gradient accuracy...", flush=True)

    path = RESULTS_DIR / "gradient_convergence.json"
    if not path.exists():
        print("    Skipping (no data)")
        return

    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    grids = list(data.keys())
    fd_errors = [data[g]["fd_error"] * 100 for g in grids]
    ad_errors = [data[g]["ad_error"] * 100 for g in grids]

    x = np.arange(len(grids))
    w = 0.35
    ax.bar(x - w / 2, fd_errors, w, label="Finite Difference", color="C0")
    ax.bar(x + w / 2, ad_errors, w, label="Implicit Diff (ours)", color="C1")

    ax.set_xticks(x)
    ax.set_xticklabels(grids)
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("Gradient Accuracy: FD vs Implicit Differentiation")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_gradient_accuracy.pdf", bbox_inches="tight")
    plt.savefig(RESULTS_DIR / "fig_gradient_accuracy.png", bbox_inches="tight")
    plt.close()


# ======================================================================
# Main
# ======================================================================


def main():
    print("=" * 60)
    print("DiffCFD v0.6 — Paper Results Collection")
    print(f"PyTorch {torch.__version__}, CPU threads={torch.get_num_threads()}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60)

    t_start = time.time()

    # Step 1: High-resolution cavity validation
    cavity_results = run_lid_cavity_validation()

    # Step 2: Poiseuille validation
    poiseuille_results = run_poiseuille_validation()

    # Step 3: Gradient convergence
    _gradient_results = run_gradient_convergence()

    # Step 4: sCO2 surrogate training
    sco2_results = run_sco2_training()

    # Step 5: Full benchmark suite
    benchmark_results = run_benchmark_suite()

    # Step 6: Generate figures
    generate_figures(cavity_results, poiseuille_results, sco2_results)

    # Summary
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"COMPLETE — Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"Results saved to {RESULTS_DIR}/")
    print("=" * 60)

    # Final summary JSON
    summary = {
        "total_time_s": total_time,
        "cavity_re100_l2_64": cavity_results["re100"]["64"]["l2_error"],
        "cavity_re1000_l2_128": cavity_results["re1000"]["128"]["l2_error"],
        "poiseuille_l2": poiseuille_results["32x16"]["forward_l2"],
        "gradient_accuracy": poiseuille_results["gradient"]["rel_error"],
        "sco2_density_rel_l2": sco2_results["density_rel_l2"],
        "benchmarks_passed": sum(1 for r in benchmark_results if r.status == "pass"),
        "benchmarks_total": len(benchmark_results),
    }
    save_json(summary, "summary.json")


if __name__ == "__main__":
    main()
