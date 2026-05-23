# DiffCFD

> **Differentiable Computational Fluid Dynamics for Steady-State Inverse Design and Reinforcement Learning**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-pre--implementation-orange.svg)]()

**DiffCFD** is a PyTorch-native differentiable fluid dynamics solver focused on **steady-state incompressible flow**, **shape and parameter optimization**, and **gradient-attached `gymnasium.Env` reinforcement learning environments** — built around a single technical commitment: matrix-free implicit differentiation through SIMPLE-converged steady states, so that gradients of any quantity of interest with respect to geometry or boundary conditions are exact at convergence and use **O(N) memory regardless of the number of solver iterations**.

> **Status (2026-05):** Pre-implementation. The repository currently contains design documents only. No working code, no released package on PyPI. See [Roadmap](#roadmap) for milestones and [`plans/development-plan.md`](plans/development-plan.md) for the technical and patent strategy.

---

## Why DiffCFD?

Production CFD tools (OpenFOAM, ANSYS Fluent, SU2) are accurate but not differentiable — you cannot backpropagate through them. Existing differentiable CFD frameworks each have a structural gap:

- **PhiFlow / JAX-Fluids**: transient time-stepping; no steady-state implicit differentiation
- **HydroGym**: standard `gymnasium.Env` only on non-differentiable Firedrake backend; differentiable JAX backend uses `gymnax` (not standard gymnasium)
- **FluidGym**: PISO transient + `GymFluidEnv.step()` calls `.detach()` — gymnasium-compatible and differentiable modes are mutually exclusive

**DiffCFD targets the intersection that is empty in all prior work:**

```
PyTorch-native × incompressible FV/SIMPLE × steady-state implicit diff × standard gymnasium.Env (gradient-attached)
```

This combination makes DiffCFD useful for:

- **Steady-state shape optimization** — geometry → SIMPLE → drag/Nusselt → `loss.backward()` with O(N) memory
- **Single-step contextual-bandit RL** — design parameters as actions, steady-state physics as the environment
- **Quasi-steady-state flow control** — sequential MDP where each step is a steady-state solve under updated boundary conditions
- **Coupled optimization** — fluid + heat + species + geometry jointly through one autograd graph

See [`plans/development-plan.md`](plans/development-plan.md) §"Competitive Landscape Analysis" for a detailed source-code-verified comparison against HydroGym, FluidGym, PhiFlow, JAX-Fluids, and NVIDIA Modulus.

---

## Design Philosophy

DiffCFD is intentionally **not** a full-featured CFD code. Explicit trade-offs:

| DiffCFD | Production CFD (OpenFOAM, Fluent) |
|---|---|
| Differentiable end-to-end (autograd through SIMPLE) | Not differentiable |
| **CPU-first**, GPU-capable | CPU-first, MPI-parallel |
| 2D incompressible NS + heat transfer | Full compressible, complex turbulence models |
| Structured Cartesian + Brinkman immersed boundary | Unstructured, body-fitted meshes |
| O(N) memory backward via implicit diff | N/A |
| **Single-laptop runs at 64²–128²** | Cluster-scale meshes |
| Gradient-based optimization, RL, surrogates | High-fidelity production simulation |

Use DiffCFD for **optimization loops and ML training at small-to-moderate grid sizes**. Use OpenFOAM for **final validation and production runs**.

---

## Compute Requirements

DiffCFD's v0.1 milestone is designed to run on a **single CPU laptop** without GPU acceleration:

| Configuration | Use case | Hardware |
|---|---|---|
| **64² grid, 2D, CPU** | v0.05/v0.1 development, lid-driven cavity validation, Poiseuille gradient verification | Any modern laptop, ~8 GB RAM |
| **128² grid, 2D, CPU** | Cylinder wake, backward-facing step | 16+ GB RAM |
| **256² grid, 2D, GPU optional** | Aerodynamic shape optimization (v0.4) | GPU recommended but not required |
| **3D** | Out of scope for v0.x; planned for post-v1.0 | — |

This is a deliberate scope choice: matrix-free GMRES backward at 64² fits comfortably in a few hundred MB; the bottleneck is wall-clock time, not memory. CPU runtime targets are minutes per gradient step at 64², not hours.

---

## Planned Features (Pre-Implementation)

These are design targets, not shipped functionality. Track progress in [Roadmap](#roadmap).

- **Differentiable steady-state Navier-Stokes (v0.1)** — incompressible 2D, SIMPLE pressure-velocity coupling, structured Cartesian + SDF-based Brinkman penalization for embedded geometry
- **Implicit differentiation (v0.1)** — fixed-point gradient via matrix-free GMRES on the unrelaxed physics residual; O(N) memory backward independent of forward iteration count
- **Conjugate heat transfer (v0.2)** — coupled NS + energy with differentiable Nusselt number
- **Differentiable sCO₂ property surrogate (v0.2)** — neural surrogate against NIST REFPROP in the transcritical region
- **Standard `gymnasium.Env` with gradient-attached `step()` (v0.3)** — Mode A (single-step contextual bandit) and Mode B (sequential quasi-steady-state); compatible with Stable-Baselines3, CleanRL
- **Frozen eddy viscosity turbulence (v0.35)** — load μ_t from external RANS for engineering-relevant Re
- **Aerodynamic shape optimization (v0.4)** — NACA + B-spline parameterized airfoils with multi-objective Pareto

---

## Roadmap

DiffCFD is developed by a single contributor in evening time. Timelines below reflect realistic part-time pace, not full-time engineering. Each milestone is gated by validation criteria documented in [`plans/development-plan.md`](plans/development-plan.md).

| Milestone | Scope | Estimated time | Status |
|---|---|---|---|
| **v0.05** *(internal, never released)* | Unrolled SIMPLE on lid-driven cavity Re=100, autograd through full iteration; cross-validation reference for v0.1 | 8–12 weeks | not started |
| **v0.1** *(CN patent filing gate)* | 2D incompressible NS + matrix-free GMRES implicit diff + Poiseuille analytical gradient verification + Ghia validation Re=100/1000 | 14–20 weeks after v0.05 | not started |
| **v0.2** | Conjugate heat transfer + sCO₂ property surrogate + PCHE Nu validation | ~3 months after v0.1 | not started |
| **v0.3** | `gymnasium.Env` cylinder wake (Mode B) + heat exchanger fin (Mode A); APG vs SB3 PPO sample efficiency benchmark | ~3 months after v0.2 | not started |
| **v0.35** | Frozen eddy viscosity for Re > 5000 duct flows | ~6 weeks after v0.3 | not started |
| **v0.4** | NACA + B-spline aerodynamic shape optimization | ~3 months after v0.35 | not started |
| **v0.5** | FNO/DeepONet surrogate-in-the-loop | TBD | not started |
| **v0.6** | sCO₂ PCHE optimization, integration with sCO2-TMSR-Toolkit | TBD | not started |
| **v1.0** | Full benchmark suite + arXiv paper | TBD | not started |

**Total wall-clock to v0.1 (CN filing-ready):** estimated 22–32 weeks of evening-time work.

---

## Patent and Open-Source Strategy

DiffCFD is being developed under a **patent-first, then open-source** strategy. The repository is intentionally code-empty until China's National Intellectual Property Administration (CNIPA) confirms the priority date for the core algorithmic claims.

The strict release sequence is:

1. Implement v0.05 + v0.1 locally on `dev/*` branches; **never push solver code to `main`**
2. Submit CN invention patent application
3. Wait for **申请号 + 申请日 written confirmation** (not 受理通知书 — that arrives later and is a formality check, not the priority anchor)
4. **File PCT international application** within 12 months using CN as priority base — this preserves EP/JP/KR/US national-stage rights
5. **Only after PCT is filed** push solver code publicly

This sequence matters because China's Art. 24 novelty grace period does **not** cover the inventor's own GitHub publication, and EPO has no self-disclosure grace period at all. A premature public push permanently destroys novelty in China and Europe with no recovery path.

Full details — including the dual-function architecture rationale for C1, freedom-to-operate analysis vs HydroGym/FluidGym/PhiFlow, and the C1+C2+C3+C4 claim structure — are in [`plans/development-plan.md`](plans/development-plan.md).

---

## Quickstart (Aspirational — Not Yet Functional)

The API below illustrates the v0.1 target shape. **None of this code runs today.** It is included to show how implicit-diff steady-state shape optimization will look once v0.1 lands.

```python
# v0.1 target — not yet implemented
import torch
from diffcfd.solvers import NavierStokes2D
from diffcfd.geometry import BSplineWall

geometry = BSplineWall(n_control_points=8)
control_points = geometry.initial().requires_grad_(True)

solver = NavierStokes2D(
    reynolds_number=100,
    grid=(64, 64),
    device="cpu",                  # CPU-first: 64² runs on a laptop
    backward="implicit_diff",      # matrix-free GMRES, O(N) memory
)

optimizer = torch.optim.Adam([control_points], lr=1e-2)
for step in range(100):
    sdf = geometry.signed_distance(control_points)
    u, p = solver.solve_steady(sdf, inlet_velocity=1.0)
    pressure_drop = solver.pressure_drop(u, p)
    pressure_drop.backward()       # exact analytical gradient via implicit diff
    optimizer.step()
    optimizer.zero_grad()
```

Track progress on the [Roadmap](#roadmap).

---

## Validation Targets (Aspirational)

The v0.1 acceptance gate requires the following validation criteria to pass before CN patent filing. These are **targets**, not measured results:

| Case | Re | Target | Reference |
|---|---|---|---|
| Lid-driven cavity (u-velocity, grid-converged) | 100 | L2 < 1% | Ghia et al. 1982 |
| Lid-driven cavity (u-velocity, grid-converged) | 1000 | L2 < 2% | Ghia et al. 1982 |
| Poiseuille pressure-drop gradient ∂ΔP/∂U_inlet | — | < 0.01% vs analytical 12μL/h² | Closed form |
| Backward-facing step reattachment length | 800 | < 5% | Kim & Moin 1985 |
| `torch.autograd.gradcheck` on subcomponents | — | passes | — |
| Adjoint GMRES convergence under Brinkman ε=1e-3 | 1000 | < 200 iterations | — |

Cross-validation requirement: implicit-diff gradients must agree with v0.05 unrolled-SIMPLE gradients to better than 0.1% on a fixed test case.

---

## Citation

Pre-publication. A `CITATION.cff` will be added with the v0.1 release.

---

## Contributing

This repository is currently in a pre-implementation, patent-sensitive phase. **Pull requests touching solver core (`diffcfd/solvers/*`) are not being accepted before the CN priority date is confirmed.** Discussion issues, documentation suggestions, and benchmark target proposals are welcome.

A full `CONTRIBUTING.md` and CLA will be added with the v0.1 public release.

## License

Apache 2.0 — see [LICENSE](LICENSE) (will be added with the first public release).
