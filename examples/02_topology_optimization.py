"""Example: topology optimization for minimum pressure drop.

Demonstrates the C3 patent embodiment:
    design variables → Helmholtz filter → smooth Heaviside →
    Brinkman penalization → SIMPLE solve → pressure drop → implicit diff

All within a single PyTorch autograd computational graph.
"""

import torch
from diffcfd import (
    NavierStokes2D,
    HelmholtzFilter,
    optimize_topology,
    smooth_heaviside,
)

if __name__ == "__main__":
    result = optimize_topology(
        objective="pressure_drop",
        grid=(32, 16),
        lx=2.0,
        ly=1.0,
        re=50.0,
        n_steps=15,
        lr=0.03,
        filter_radius=0.1,
        beta=16.0,
        inlet_velocity=1.0,
        verbose=True,
    )
    print(f"\nOptimization complete.")
    print(f"Final |ΔP|: {result['history']['objective'][-1]:.4f}")
    print(f"Fluid fraction: {result['history']['fluid_fraction'][-1]:.3f}")
    print(f"Design shape: {result['chi'].shape}")
