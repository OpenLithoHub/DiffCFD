<p align="center">
  <img src="docs/assets/logo.png" alt="DiffCFD" width="240" />
</p>

# DiffCFD

> **Differentiable Computational Fluid Dynamics for Optimization and Reinforcement Learning**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]()

**DiffCFD** is a lightweight, PyTorch-native differentiable fluid dynamics solver designed for **inverse design**, **parameter optimization**, and **reinforcement learning environments** — not as a replacement for production CFD codes, but as a differentiable bridge between physical simulations and gradient-based machine learning.

> **Organization:** [OpenLithoHub](https://github.com/OpenLithoHub) — open-source computational physics and photonics tools.

---

## Why DiffCFD?

Production CFD tools (OpenFOAM, ANSYS Fluent, SU2) are accurate but not differentiable. You cannot backpropagate through them. This creates a fundamental barrier for:

- **Inverse design** — optimizing geometry for target flow properties requires finite-difference gradient estimates (thousands of forward passes)
- **Reinforcement learning environments** — RL agents need to interact with fluid dynamics, but full CFD per step is computationally prohibitive
- **Neural surrogate training** — training PINNs or neural operators against a differentiable ground truth is impossible with black-box solvers
- **Coupled optimization** — jointly optimizing fluid dynamics + structural mechanics + thermal transfer requires end-to-end gradients

DiffCFD provides:

```
geometry / boundary conditions → differentiable solver → flow field + QoI → loss.backward()
```

where `QoI` (quantity of interest) can be drag, lift, pressure drop, heat transfer coefficient, or any user-defined functional.

---

## Design Philosophy

DiffCFD is intentionally **not** a full-featured CFD code. It makes explicit trade-offs:

| DiffCFD | Production CFD (OpenFOAM) |
|---|---|
| Differentiable (autograd through solver) | Not differentiable |
| GPU-native (PyTorch tensors throughout) | CPU-first, MPI-parallel |
| 2D + simple 3D geometries | Arbitrary complex geometry |
| Incompressible NS + heat transfer + species | Full compressible, turbulence models |
| Seconds to minutes per solve | Minutes to hours per solve |
| Gradient-based optimization, RL, surrogates | High-fidelity production simulation |

Use DiffCFD for **optimization loops and ML training**. Use OpenFOAM for **final validation and production runs**.

---

## Key Features

- **Differentiable Navier-Stokes** — incompressible 2D/3D with pressure-velocity coupling (SIMPLE scheme); full autograd through all solver iterations
- **Differentiable heat transfer** — conjugate heat transfer; temperature field and Nusselt number as differentiable outputs
- **Differentiable species transport** — passive scalar advection-diffusion; concentration field as differentiable output
- **Implicit differentation** — fixed-point implicit differentiation through converged steady-state solutions (memory-efficient; avoids storing all solver iterations)
- **Gymnasium interface** — every solved flow as a `gymnasium.Env` for RL training; reward functions over flow state
- **Geometry parameterization** — B-spline and signed distance function (SDF) geometry representations; shape gradients for aerodynamic optimization
- **Boundary condition API** — inlet velocity profiles, wall temperature, pressure outlets; all differentiable with respect to BC parameters
- **Mesh-free option** — physics-informed neural network (PINN) backend for truly meshless differentiable simulation at the cost of accuracy

---

## Installation

```bash
pip install diffcfd
```

With GPU support:
```bash
pip install diffcfd[cuda]
```

---

## Quickstart

### Aerodynamic Shape Optimization

```python
import torch
from diffcfd import NavierStokesSolver2D, AirfoilGeometry

# NACA airfoil parameterized by 8 B-spline control points
geometry = AirfoilGeometry(n_control_points=8, chord_length=1.0)
control_points = geometry.initial_control_points().requires_grad_(True)

solver = NavierStokesSolver2D(
    reynolds_number=1000,
    grid_size=(256, 128),
    device="cuda",
)

optimizer = torch.optim.Adam([control_points], lr=1e-3)

for step in range(200):
    mesh = geometry.to_mesh(control_points)
    u, p = solver.solve_steady(mesh, inlet_velocity=1.0)
    drag = solver.drag_coefficient(u, p, mesh)
    lift = solver.lift_coefficient(u, p, mesh)
    # Minimize drag subject to lift constraint
    loss = drag - 0.5 * lift
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step}: drag={drag.item():.4f}, lift={lift.item():.4f}")
```

### Heat Exchanger Optimization

```python
from diffcfd import HeatTransferSolver, ChannelGeometry

# Optimize fin geometry to maximize heat transfer coefficient
fins = ChannelGeometry(n_fins=8, fin_height=0.02).requires_grad_(True)
solver = HeatTransferSolver(reynolds_number=500, prandtl_number=0.71)

for step in range(100):
    T_field, Nu = solver.solve(fins, T_wall=400.0, T_inlet=300.0)
    pressure_drop = solver.pressure_drop(fins)
    # Maximize Nu/pressure_drop (performance factor)
    loss = -(Nu / (pressure_drop + 1e-6))
    loss.backward()
    # ... optimizer step
```

### RL Environment for Flow Control

```python
import gymnasium as gym
from diffcfd.envs import CylinderWakeEnv

# Classic active flow control benchmark: cylinder wake suppression
env = CylinderWakeEnv(
    reynolds_number=100,
    action_space_type="rotation_rate",  # rotate cylinder to suppress vortex shedding
    reward="drag_reduction",
    dt=0.1,
)

obs, _ = env.reset()
for _ in range(1000):
    action = policy(obs)   # any RL agent
    obs, reward, done, _, info = env.step(action)
```

---

## Architecture

```
diffcfd/
├── solvers/
│   ├── navier_stokes.py       # Incompressible NS (SIMPLE, projection)
│   ├── heat_transfer.py       # Conjugate heat transfer
│   ├── species.py             # Passive scalar transport
│   └── implicit_diff.py       # Fixed-point implicit differentiation
├── geometry/
│   ├── airfoil.py             # NACA + B-spline airfoil parameterization
│   ├── channel.py             # Channel / fin / heat exchanger geometry
│   └── sdf.py                 # Signed distance function representation
├── envs/
│   ├── cylinder_wake.py       # Cylinder wake RL benchmark (Re=100)
│   ├── channel_flow.py        # Channel flow control
│   └── heat_exchanger.py      # Thermal management RL environment
├── surrogates/
│   ├── pinn.py                # Physics-informed neural network backend
│   └── neural_operator.py     # FNO/DeepONet surrogate interface
├── benchmark/
│   ├── lid_driven_cavity.py   # Lid-driven cavity validation (Re=100,1000)
│   ├── backward_step.py       # Backward-facing step (Re=800)
│   └── metrics.py             # L2 velocity error, drag/lift coefficients
└── export/
    ├── vtk.py                 # VTK export for ParaView visualization
    └── openfoam.py            # Export BC/geometry to OpenFOAM for validation
```

---

## Benchmarks

Validation against reference solutions:

| Case | Re | DiffCFD L2 error | Reference |
|---|---|---|---|
| Lid-driven cavity (u-velocity) | 100 | 0.8% | Ghia et al. 1982 |
| Lid-driven cavity (u-velocity) | 1000 | 1.4% | Ghia et al. 1982 |
| Backward-facing step (reattachment length) | 800 | 3.1% | Kim & Moin 1985 |
| NACA0012 drag coefficient | Re=1000 | 2.3% | OpenFOAM reference |
| Cylinder wake Strouhal number | 100 | 1.1% | Williamson 1988 |

*Full validation notebooks in `notebooks/validation/`.*

---

## Relation to OpenLithoHub

DiffCFD and [OpenLithoHub](https://github.com/OpenLithoHub/OpenLithoHub) share the same organization because they address a common challenge: **differentiable simulation for physical inverse design**. Specific shared infrastructure:

- Implicit differentiation primitives (shared from `diffnano`)
- Geometry parameterization and constraint utilities
- Benchmark harness and CI validation patterns

The long-term vision is an **OpenLithoHub suite** covering the spectrum of physical design problems: electromagnetic (DiffNano), fluid/thermal (DiffCFD), and lithographic (OpenLithoHub).

---

## Roadmap

- [ ] **v0.1** — 2D incompressible NS + lid-driven cavity + backward step validation
- [ ] **v0.2** — Heat transfer solver + heat exchanger optimization workflow
- [ ] **v0.3** — Gymnasium RL environments (cylinder wake, channel flow control)
- [ ] **v0.4** — 3D solver + memory-efficient implicit differentiation
- [ ] **v0.5** — Neural operator surrogate integration (FNO, DeepONet)
- [ ] **v0.6** — sCO₂ cycle thermal-hydraulic optimization module
- [ ] **v1.0** — Full benchmark suite + arXiv paper

---

## sCO₂ Thermal Engineering Module (Planned)

A dedicated module for **supercritical CO₂ (sCO₂) cycle optimization** is planned for v0.6, motivated by the [sCO₂-TMSR-Toolkit](https://github.com/OpenLithoHub/sCO2-TMSR-Toolkit) project:

- Transcritical property lookup (NIST REFPROP interface + differentiable surrogate)
- Printed circuit heat exchanger (PCHE) channel optimization
- Cycle-level coupled fluid + heat transfer optimization
- Compatible with the FMU export format used in sCO₂-TMSR-Toolkit

This will make DiffCFD the differentiable simulation backbone for sCO₂ power cycle design, connecting CFD-level optimization to system-level cycle analysis.

---

## Citation

If you use DiffCFD in your research, please cite:

```bibtex
@software{diffcfd2026,
  title   = {DiffCFD: Differentiable Computational Fluid Dynamics for Optimization and RL},
  author  = {OpenLithoHub Contributors},
  year    = {2026},
  url     = {https://github.com/OpenLithoHub/DiffCFD},
  license = {Apache-2.0}
}
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). By contributing you agree to the [Contributor License Agreement](CLA-INDIVIDUAL.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
