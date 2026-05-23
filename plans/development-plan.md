# DiffCFD — Development Plan

**Status:** Pre-implementation planning
**Created:** 2026-05-23
**Patent strategy:** China first-filing before any core algorithm push

---

## Patent Strategy

1. Implement core algorithms locally (do NOT push until CN filing)
2. Submit China invention patent application (locks priority date)
3. Push code to GitHub the same day or next day (open source)
4. File PCT within 12 months using CN filing as priority base

**Do NOT push the following until CN filing:**
- `diffcfd/solvers/navier_stokes.py` — implicit differentiation through SIMPLE (core claim)
- `diffcfd/solvers/implicit_diff.py` — fixed-point differentiation primitives
- Any code implementing the claims listed in Section 5 below

---

## v0.1 Milestone — 2D Incompressible NS + Validation

**Target:** 2-3 months
**Gate for CN patent filing**

### Core deliverables

- [ ] `diffcfd/solvers/navier_stokes.py` — differentiable 2D incompressible NS
  - Finite volume discretization on structured grid
  - SIMPLE pressure-velocity coupling (steady state)
  - Implicit differentiation via `torch.linalg.solve` + custom backward
  - Fixed-point implicit differentiation for converged steady-state (memory-efficient)
- [ ] `diffcfd/solvers/boundary.py`
  - Inlet (Dirichlet velocity), outlet (Neumann pressure), no-slip wall
  - All BC parameters differentiable (inlet velocity, wall temperature)
- [ ] `diffcfd/geometry/channel.py`
  - Structured mesh generation for channel / pipe geometries
  - B-spline wall parameterization (differentiable geometry)
- [ ] Validation suite:
  - Lid-driven cavity Re=100, Re=1000 (Ghia et al. 1982)
  - Backward-facing step Re=800 (Kim & Moin 1985)
  - Poiseuille flow (analytical solution)
- [ ] `diffcfd/export/vtk.py` — VTK export for ParaView visualization

---

## v0.2 Milestone — Heat Transfer + Optimization Workflow

**Target:** 2 months after v0.1

- [ ] `diffcfd/solvers/heat_transfer.py`
  - Conjugate heat transfer (coupled fluid + solid energy equation)
  - Nusselt number as differentiable output
- [ ] `diffcfd/workflows/heat_exchanger.py`
  - Fin geometry optimization: maximize Nu / pressure_drop
  - Adam optimizer loop with fabrication constraints
- [ ] Validation: PCHE (printed circuit heat exchanger) against published correlations
- [ ] sCO₂ property interface (NIST REFPROP lookup + differentiable surrogate for
  transcritical region — feeds into sCO₂-TMSR-Toolkit integration)

---

## v0.3 Milestone — Gymnasium RL Environments

**Target:** 2 months after v0.2

- [ ] `diffcfd/envs/cylinder_wake.py`
  - Classic active flow control: cylinder wake suppression (Re=100)
  - Action: cylinder rotation rate
  - Reward: drag reduction
  - Compatible with stable-baselines3, CleanRL
- [ ] `diffcfd/envs/channel_flow.py`
  - Turbulent channel flow control via wall blowing/suction
- [ ] `diffcfd/envs/heat_exchanger.py`
  - Thermal management RL: dynamic fin actuation

---

## v0.4 — Aerodynamic Shape Optimization

- [ ] `diffcfd/geometry/airfoil.py` — NACA + B-spline airfoil
- [ ] `diffcfd/workflows/aero.py` — drag/lift optimization
- [ ] Validation: NACA0012 at Re=1000 vs OpenFOAM reference

---

## v0.5 — Neural Operator Surrogates

- [ ] FNO (Fourier Neural Operator) training interface
  - Use DiffCFD as ground-truth solver to generate training data
  - Train FNO surrogate; compare inference speed vs accuracy
- [ ] DeepONet interface
- [ ] Surrogate-in-the-loop optimization (fast surrogate + periodic DiffCFD correction)

---

## v0.6 — sCO₂ Thermal-Hydraulic Module

Integration with [sCO₂-TMSR-Toolkit](https://github.com/OpenLithoHub/sCO2-TMSR-Toolkit):

- [ ] Transcritical CO₂ property surrogate (differentiable, covers pseudocritical region)
- [ ] PCHE channel shape optimization (maximize compactness factor)
- [ ] Cycle-level coupled optimization: CFD-level PCHE + system-level Brayton cycle
- [ ] FMU export of optimized PCHE for use in sCO₂-TMSR-Toolkit Modelica models

---

## Patent Claims (Draft — Pre-filing, Confidential)

### C1 — Fixed-point implicit differentiation through CFD steady-state
A method for computing exact gradients with respect to geometry and boundary condition parameters through a converged steady-state CFD solution using fixed-point implicit differentiation, such that memory consumption is independent of the number of solver iterations and gradients are exact (not approximate via finite differences or truncated unrolling).

### C2 — Differentiable CFD as a Gymnasium reinforcement learning environment
A software interface that wraps a differentiable fluid dynamics solver as a standard Gymnasium environment, such that (i) the environment step function returns physically consistent flow-field observations, (ii) the differentiable solver provides analytical policy gradients to the RL training loop without finite-difference estimation, and (iii) the same solver is used for both environment rollout and gradient computation, eliminating the sim-to-sim gap between environment and gradient model.

### C3 — Coupled shape and boundary condition optimization with fabrication constraints
A gradient-based optimization framework that simultaneously optimizes fluid dynamic geometry (airfoil shape, fin geometry, channel profile) and boundary conditions (inlet velocity, wall temperature) subject to manufacturing constraints (minimum feature size, curvature), using a shared differentiable loss that combines fluid dynamic objective (drag, Nusselt number, pressure drop) with fabrication penalty.

### C4 — sCO₂ transcritical property surrogate for differentiable cycle optimization
A differentiable neural network surrogate for supercritical CO₂ thermophysical properties in the transcritical region (0.9Tc–1.1Tc, 0.9Pc–1.5Pc) trained against NIST REFPROP data, with guaranteed physical consistency (monotonicity of density, positivity of Cp), enabling end-to-end gradient-based optimization of sCO₂ Brayton cycles from CFD-level heat exchanger design to system-level cycle efficiency.

---

## Key References

- Ghia et al. (1982) — High-Re solutions for incompressible flow using the Navier-Stokes equations. *Journal of Computational Physics*
- Kim & Moin (1985) — Application of a fractional-step method to incompressible NS equations. *Journal of Computational Physics*
- Rabault et al. (2019) — Artificial neural networks trained through deep reinforcement learning discover control strategies for active flow control. *Journal of Fluid Mechanics*
- Li et al. (2021) — Fourier Neural Operator for parametric PDEs. *ICLR 2021*
- Belbute-Peres et al. (2020) — Combining differentiable PDE solvers and graph neural networks. *ICML 2020*
- Kochkov et al. (2021) — Machine learning–accelerated computational fluid dynamics. *PNAS*

---

## Competitive Landscape

| Tool | Differentiable | GPU-native | RL env | Open-source | Status |
|---|---|---|---|---|---|
| OpenFOAM | No | No | No | Yes | Production CFD |
| SU2 | Adjoint only | No | No | Yes | Active |
| JAX-Fluids | Yes (JAX) | Yes | No | Yes | Research |
| PhiFlow | Yes | Yes | Partial | Yes | Active |
| **DiffCFD** | **Yes (PyTorch)** | **Yes** | **Yes** | **Yes** | **Building** |

**Differentiation from JAX-Fluids and PhiFlow:** DiffCFD is explicitly designed for (1) RL environment interface, (2) implicit differentiation for steady-state (memory-efficient), and (3) sCO₂ thermal engineering use case. Neither competitor targets these three together.
