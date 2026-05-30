#!/usr/bin/env python3
"""Generate benchmark charts for README Performance & Benchmarks section.

Outputs SVG files with transparent backgrounds, compatible with both
light and dark GitHub themes.

Usage:
    python docs/benchmark_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
DOCS_DIR = Path(__file__).parent

# Theme-aware colors: neutral gray axes/text work on both light/dark backgrounds
STYLE = {
    "text": "#888888",
    "grid": "#444444",
    "bg": "none",
    "diffcfd_blue": "#4C78A8",
    "diffcfd_orange": "#E45756",
    "diffcfd_green": "#54A24B",
    "diffcfd_purple": "#B279A2",
}


def _apply_theme(ax: plt.Axes):
    ax.tick_params(colors=STYLE["text"])
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.title.set_color(STYLE["text"])
    for spine in ax.spines.values():
        spine.set_color(STYLE["text"])
    ax.grid(True, alpha=0.2, color=STYLE["grid"])


def chart_solver_time():
    """Bar chart: SIMPLE solver wall-clock time vs grid size."""
    data = {
        "Cavity Re=100": {
            "32²": 5.6,
            "64²": 54.6,
            "128²": 129.3,
        },
        "Cavity Re=1000": {
            "64²": None,  # [待填充]
            "128²": 1316.6,
        },
        "Channel Re=1": {
            "32×16": None,  # [待填充: 单次 forward 耗时]
            "64×32": None,
            "128×64": None,
        },
    }
    # Use cavity Re=100 as representative
    grids = ["32²", "64²", "128²"]
    times = [5.6, 54.6, 129.3]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(grids, times, color=STYLE["diffcfd_blue"], width=0.5, edgecolor="none")

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{t:.1f}s",
            ha="center",
            va="bottom",
            color=STYLE["text"],
            fontsize=10,
        )

    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("Wall-clock time [s]")
    ax.set_title("DiffCFD SIMPLE Solver — Cavity Re=100")
    ax.set_yscale("log")
    _apply_theme(ax)
    fig.patch.set_facecolor(STYLE["bg"])

    out = DOCS_DIR / "images" / "benchmark_solver_time.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  {out}")


def chart_grid_convergence():
    """Log-log plot: L2 error vs grid size for cavity and channel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Cavity Re=100 convergence
    grids_cav = np.array([1024, 4096, 16384])  # 32², 64², 128²
    errors_cav = np.array([0.01964, 0.00848, None])  # 128² [待填充]
    mask = errors_cav != None
    if mask.any():
        errors_cav_arr = np.array(errors_cav[mask], dtype=float)
        grids_cav_arr = grids_cav[mask]
        ax1.loglog(grids_cav_arr, errors_cav_arr, "o-", color=STYLE["diffcfd_blue"], label="DiffCFD")
        # 1st-order reference
        ref_y = errors_cav_arr[0] * (grids_cav_arr[0] / grids_cav_arr) ** 1
        ax1.loglog(grids_cav_arr, ref_y, "--", alpha=0.35, color=STYLE["text"], label="1st-order ref")
    ax1.set_xlabel("Grid points N")
    ax1.set_ylabel("L2 error")
    ax1.set_title("Cavity Re=100 convergence")
    ax1.legend(fontsize=8)
    _apply_theme(ax1)

    # Poiseuille convergence
    grids_poi = np.array([32 * 16, 64 * 32, 128 * 64])
    errors_poi = np.array([0.004464, 0.001041, 0.000251])
    ax2.loglog(grids_poi, errors_poi, "s-", color=STYLE["diffcfd_orange"], label="DiffCFD")
    ref_y2 = errors_poi[0] * (grids_poi[0] / grids_poi) ** 2
    ax2.loglog(grids_poi, ref_y2, "--", alpha=0.35, color=STYLE["text"], label="2nd-order ref")
    ax2.set_xlabel("Grid points N")
    ax2.set_ylabel("L2 error")
    ax2.set_title("Poiseuille forward convergence")
    ax2.legend(fontsize=8)
    _apply_theme(ax2)

    fig.patch.set_facecolor(STYLE["bg"])
    plt.tight_layout()
    out = DOCS_DIR / "images" / "benchmark_grid_convergence.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  {out}")


def chart_gradient_accuracy():
    """Bar chart: FD vs AD gradient agreement across grid sizes."""
    with open(RESULTS_DIR / "gradient_convergence.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))

    grids = list(data.keys())
    fd_errors = [data[g]["fd_ad_agreement"] for g in grids]

    bars = ax.bar(grids, fd_errors, color=STYLE["diffcfd_green"], width=0.5, edgecolor="none")

    for bar, v in zip(bars, fd_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            f"{v:.1e}",
            ha="center",
            va="bottom",
            color=STYLE["text"],
            fontsize=9,
        )

    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("|AD − FD| / |FD|")
    ax.set_title("Implicit Differentiation vs Finite Difference")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e-3)
    _apply_theme(ax)
    fig.patch.set_facecolor(STYLE["bg"])

    out = DOCS_DIR / "images" / "benchmark_gradient_accuracy.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  {out}")


def chart_memory_backward():
    """Conceptual chart: O(N) backward memory scaling.

    Uses synthetic data based on the O(N) claim for restarted GMRES.
    The y-axis is peak memory per backward pass.
    """
    # This is a conceptual illustration, not measured data
    fig, ax = plt.subplots(figsize=(6, 4))

    N = np.array([512, 2048, 8192, 32768])
    # O(N) — restarted GMRES with restart=30
    implicit_mem = N * 8 * 30 / 1e6  # bytes to MB (rough)
    # O(N²) — full Jacobian storage (what unrolling would need)
    full_jacobian = N**2 * 8 / 1e6
    # O(N) — forward pass only (no backward)
    forward_only = N * 8 / 1e6

    ax.loglog(N, forward_only, "^-", color=STYLE["diffcfd_green"], label="Forward only O(N)")
    ax.loglog(N, implicit_mem, "o-", color=STYLE["diffcfd_blue"], label="Implicit diff O(N·k)")
    ax.loglog(N, full_jacobian, "s--", color=STYLE["diffcfd_orange"], label="Full Jacobian O(N²)", alpha=0.6)

    ax.set_xlabel("Degrees of freedom N")
    ax.set_ylabel("Peak memory [MB] (estimated)")
    ax.set_title("Backward Pass Memory Scaling (Conceptual)")
    ax.legend(fontsize=8)
    _apply_theme(ax)
    fig.patch.set_facecolor(STYLE["bg"])

    out = DOCS_DIR / "images" / "benchmark_memory_scaling.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  {out}")


def main():
    print("Generating benchmark charts...")
    chart_solver_time()
    chart_grid_convergence()
    chart_gradient_accuracy()
    chart_memory_backward()
    print("Done. Charts saved to docs/images/")


if __name__ == "__main__":
    main()
