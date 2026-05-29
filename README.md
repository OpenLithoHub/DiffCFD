<div align="center">

# DiffCFD

**Differentiable Computational Fluid Dynamics for Steady-State Inverse Design and Reinforcement Learning**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

PyTorch-native differentiable fluid dynamics — **matrix-free implicit differentiation** through SIMPLE-converged steady states with **O(N) memory**, plus gradient-attached `gymnasium.Env` for RL.

> **Status:** Early-stage personal research project. Core solver and implicit differentiation verified against analytical solutions. No external users or third-party validation yet.

</div>

---

## Why DiffCFD?

Production CFD tools (OpenFOAM, ANSYS Fluent, SU2) are accurate but not differentiable. Existing differentiable CFD frameworks each have a structural gap:

| Framework | Gap |
|:----------|:----|
| PhiFlow / JAX-Fluids | Transient time-stepping only — no steady-state implicit diff |
| HydroGym | Differentiable backend uses `gymnax` (not standard gymnasium) |
| FluidGym | Gymnasium-compatible mode calls `.detach()` — gradients disabled |

**DiffCFD targets the empty intersection:**

```
PyTorch-native × incompressible FV/SIMPLE × steady-state implicit diff × standard gymnasium.Env
```

Use cases:
- **Shape optimization** — geometry → SIMPLE → drag/Nusselt → `loss.backward()` with O(N) memory
- **Contextual-bandit RL** — design parameters as actions, steady-state physics as environment
- **Quasi-steady flow control** — sequential MDP where each step is a steady-state solve
- **Coupled optimization** — fluid + heat + geometry jointly through one autograd graph

---

## Quick Start

```python
import torch
from diffcfd import NavierStokes2D, CylinderWakeEnv, HeatTransfer2D

# Steady-state SIMPLE solve — lid-driven cavity Re=100
solver = NavierStokes2D(reynolds_number=100, grid=(64, 64))
ux, uy, p = solver.solve_steady(lid_velocity=1.0, case="cavity")

# Implicit differentiation — O(N) memory backward
solver_diff = NavierStokes2D(
    reynolds_number=1.0, grid=(32, 16), lx=4.0, ly=1.0,
    backward="implicit_diff",
)
u_inlet = torch.tensor(1.0, requires_grad=True)
ux, uy, p = solver_diff.solve_steady(inlet_velocity=u_inlet, case="channel")
dp = solver_diff.pressure_drop(ux, uy, p)
dp.backward()  # Exact gradient via matrix-free GMRES

# Gymnasium environment
env = CylinderWakeEnv(re=100, grid=(64, 32))
obs, info = env.reset()
obs, reward, done, truncated, info = env.step([0.5])
```

---

## Installation

```bash
pip install torch numpy scipy gymnasium
pip install -e .

# Optional
pip install pytest pyamg matplotlib
```

---

## Validation (Verified)

| Case | Re | Target | Result | Status |
|:-----|:---|:-------|:-------|:-------|
| Lid-driven cavity u-velocity (64²) | 100 | L2 < 1% | < 1% | Pass |
| Lid-driven cavity u-velocity (128²) | 1000 | L2 < 2% | < 2% | Pass |
| Poiseuille ∂ΔP/∂U_inlet | 1 | < 0.01% vs analytical | < 0.01% | Pass |
| `torch.autograd.gradcheck` (Poiseuille) | 1 | passes | passes | Pass |
| Pure conduction Nusselt number | — | Nu = 1.0 | 1.0000 | Pass |
| Backward-facing step (Brinkman) | 100 | bounded, recirculating | pass | Pass |

---

## Design Philosophy

DiffCFD is intentionally **not** a full-featured CFD code:

| DiffCFD | Production CFD (OpenFOAM, Fluent) |
|:--------|:----------------------------------|
| Differentiable end-to-end | Not differentiable |
| **CPU-first**, GPU-capable | CPU-first, MPI-parallel |
| 2D incompressible NS + heat | Full compressible, complex turbulence |
| Structured Cartesian + Brinkman IB | Unstructured, body-fitted meshes |
| O(N) memory backward | N/A |
| **Single-laptop at 64²–128²** | Cluster-scale meshes |

Use DiffCFD for **optimization loops and ML training**. Use OpenFOAM for **final validation and production runs**.

| Config | Hardware |
|:-------|:---------|
| 64² grid, 2D, CPU | Any modern laptop (~8 GB RAM) |
| 128² grid, 2D, CPU | 16+ GB RAM |
| 256² grid, 2D | GPU recommended |
| 3D | Out of scope for v0.x |

---

## Architecture

```
diffcfd/
├── solvers/
│   ├── navier_stokes_2d.py    # 2D incompressible NS + SIMPLE
│   ├── heat_transfer.py       # Conjugate heat transfer
│   ├── turbulence.py          # Frozen eddy viscosity (Re > 5000)
│   └── implicit_diff.py       # Matrix-free GMRES backward
├── envs/
│   ├── cylinder_wake.py       # Cylinder wake RL (Mode B)
│   ├── heat_exchanger.py      # Heat exchanger fin (Mode A)
│   └── base.py
├── geometry/
│   ├── mesh.py                # Cartesian mesh generation
│   ├── shapes.py              # SDFs (cylinder, rectangle, NACA)
│   └── airfoil.py             # NACA 4-digit + B-spline
└── workflows/
    ├── aero.py                # Aerodynamic shape optimization
    └── topology.py            # Topology optimization + Helmholtz filter
```

---

## Roadmap

| Milestone | Scope | Status |
|:----------|:------|:-------|
| v0.1 | 2D NS + matrix-free implicit diff + validation | Done |
| v0.2 | Conjugate heat transfer + sCO₂ surrogate | Heat done, sCO₂ pending (no timeline) |
| v0.3 | Gymnasium environments (CylinderWake + HeatExchanger) | Done |
| v0.35 | Frozen eddy viscosity for Re > 5000 | Done |
| v0.4 | NACA + B-spline aerodynamic shape optimization | Done |
| v0.4.1 | Helmholtz filter + topology optimization | Done |
| v0.5 | FNO/DeepONet surrogate-in-the-loop | Planned |
| v0.6 | sCO₂ PCHE optimization + sCO2-TMSR-Toolkit integration | Planned |
| v1.0 | Full benchmark suite + arXiv paper | Planned |

---

## Contributing

This repository is currently in a patent-sensitive phase. Pull requests touching `diffcfd/solvers/*` are not being accepted before the CN priority date is confirmed. Discussion issues and benchmark proposals are welcome.

---

## License

Apache License 2.0
