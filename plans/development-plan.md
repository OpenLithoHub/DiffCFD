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

## Competitive Landscape Analysis (as of 2026-05)

### Direct competitors — know before you build

| Tool | Scope | Language | Stars | Last update | Threat |
|---|---|---|---|---|---|
| **JAX-Fluids 2.0** (TU Munich) | Compressible NS, 3D, multi-phase | JAX | 580 | 2026-05 active | Medium — compressible only, JAX not PyTorch, no RL env, no heat transfer |
| **PhiFlow 3.4** (TU Munich) | Incompressible NS + heat, multi-backend | PyTorch/JAX/TF | 1872 | 2026-05 active | **High** — direct overlap on incompressible NS; must differentiate clearly |
| **NVIDIA Modulus** | PINNs + neural operators, surrogate | PyTorch | 2830 | 2026-05 active | Medium — surrogate-focused, not true differentiable solver |
| SU2 | Compressible RANS, adjoint | C++ | ~4k | Active | Low — no ML loop integration, C++ not Python |
| OpenFOAM | Full CFD | C++ | ~2k | Active | None — production tool, not differentiable |

**Critical finding on PhiFlow**: PhiFlow 3.4 (updated 2026-05) already covers incompressible NS with PyTorch backend and moving obstacles. **DiffCFD cannot simply be "PyTorch incompressible NS" — PhiFlow already is that.** The differentiation must be sharper.

**What PhiFlow does NOT do well (confirmed from README/releases):**
1. No RL Gymnasium interface — PhiFlow has no `gymnasium.Env` wrapper
2. No steady-state implicit differentiation — PhiFlow differentiates through time steps (transient only); no memory-efficient steady-state gradient
3. No heat exchanger / conjugate heat transfer workflow
4. No sCO₂ transcritical property integration
5. No fabrication constraint integration
6. API complexity — PhiFlow's tensor abstraction is non-standard; steep learning curve vs. native PyTorch

**JAX-Fluids does NOT do (confirmed from README):**
- Incompressible flow (explicitly compressible-only)
- Heat transfer
- RL Gymnasium environments
- PyTorch (JAX only)

**Conclusion**: DiffCFD's defensible niche is the intersection of: **(incompressible + heat transfer) × (PyTorch-native) × (Gymnasium RL) × (steady-state implicit differentiation) × (sCO₂ thermal engineering)**. This combination has no existing open-source solution.

### Patent freedom-to-operate analysis

**NOT patented (open literature prior art):**
- Adjoint method for CFD (Pironneau 1974, Jameson 1988) — foundational, not patentable by anyone
- Fixed-point implicit differentiation through PDE solutions — Bai et al. 2019 (DEQ paper), open literature
- SIMPLE pressure-velocity coupling — Patankar & Spalding 1972, public domain
- RL for flow control — Rabault et al. 2019, open literature

**Your defensible novelty:**
- C1: Implicit differentiation through SIMPLE iteration with proven convergence guarantee (DEQ paper proves implicit diff in general; applying to SIMPLE with CFD-specific convergence proof is novel)
- C2: Differentiable CFD as Gymnasium RL environment with analytical policy gradients (no prior work wraps a true differentiable NS solver as a Gym env with autograd policy gradients)
- C3: Coupled shape + BC optimization with fabrication constraints (DiffNano C2 analog for CFD — same gap)
- C4: Differentiable sCO₂ transcritical property surrogate with physical consistency guarantees (novel combination)

**Patent risks:**
- PhiFlow team (TU Munich): Academic group, Apache 2.0, no known patents on the method
- NVIDIA Modulus: NVIDIA holds many ML patents but their CFD patents focus on surrogate models, not differentiable solvers — does not block your C1-C4
- **Conclusion: Freedom to operate confirmed for C1-C4.**

---

## v0.1 Milestone — 2D Incompressible NS + Steady-State Implicit Diff

**Target:** 2-3 months | **Gate for CN patent filing**

### Core deliverables

- [ ] `diffcfd/solvers/navier_stokes_2d.py` — differentiable 2D incompressible NS
  - Finite volume on structured Cartesian grid (staggered MAC grid)
  - SIMPLE pressure-velocity coupling for steady state
  - **Implicit differentiation via fixed-point theorem** (C1) — do NOT unroll solver iterations; instead solve the implicit gradient equation `(∂F/∂u) · du/dθ = -∂F/∂θ` using `torch.linalg.solve`
  - This gives exact gradients at the cost of one linear solve, vs. storing all SIMPLE iterations
  - Memory: O(N) instead of O(N·K) where K = number of SIMPLE iterations
  - Reference: Bai et al. 2019 (DEQ), but applied to NS SIMPLE — **this is the novel part**

- [ ] `diffcfd/solvers/boundary.py`
  - Inlet (Dirichlet velocity), outlet (zero-gradient pressure), no-slip wall, symmetry
  - All BC parameters differentiable: inlet velocity profile shape, wall temperature

- [ ] `diffcfd/geometry/mesh.py`
  - Structured Cartesian mesh with immersed boundary (simple cut-cell)
  - B-spline parameterized wall: smooth geometry → mesh → differentiable

- [ ] Validation suite (mandatory before CN filing):
  - Lid-driven cavity Re=100 vs Ghia et al. 1982 (L2 error < 1%)
  - Lid-driven cavity Re=1000 vs Ghia et al. 1982 (L2 error < 2%)
  - Poiseuille flow: analytical solution comparison
  - Backward-facing step Re=800: reattachment length within 5%

- [ ] `diffcfd/export/vtk.py` — VTK export for ParaView visualization

- [ ] **2026 addition**: Gradient verification suite
  - `torch.autograd.gradcheck` on every solver component
  - Comparison of implicit diff gradients vs. finite difference (show they agree to 1e-4)
  - This is essential for the patent claim — gradients must be demonstrably exact

---

## v0.2 Milestone — Heat Transfer + Heat Exchanger Optimization

**Target:** 2 months after v0.1

- [ ] `diffcfd/solvers/heat_transfer.py`
  - Conjugate heat transfer (energy equation coupled with NS)
  - Differentiable Nusselt number output
  - Steady-state implicit diff extended to coupled NS + energy system

- [ ] `diffcfd/workflows/heat_exchanger.py`
  - Fin geometry optimization: maximize Nu / pressure_drop (performance factor)
  - Fabrication constraint: minimum fin thickness (analog to MRC in OpenLithoHub)
  - Pareto front: Nu vs pressure drop for varying fin shapes

- [ ] **sCO₂ property module** (feeds into v0.6 plan):
  - `diffcfd/props/sco2.py` — differentiable sCO₂ property surrogate
  - Trained against NIST REFPROP data in transcritical region (0.9Tc–1.1Tc)
  - Physical consistency: monotone density, positive Cp (enforced by architecture)
  - This is C4 — the differentiable transcritical surrogate

- [ ] Validation: PCHE (printed circuit heat exchanger) Nu correlation vs. Kim 2016

---

## v0.3 Milestone — Gymnasium RL Environments (C2)

**Target:** 2 months after v0.2

**This is the key differentiator from PhiFlow and JAX-Fluids — neither has this.**

- [ ] `diffcfd/envs/base.py` — base `gymnasium.Env` wrapping DiffCFD solver
  - `step()` returns flow field observation + differentiable reward
  - `policy_gradient()` — analytical gradient of reward w.r.t. action via autograd (C2)
  - Both model-free RL (PPO, SAC via step rollouts) and model-based (analytical gradients) supported

- [ ] `diffcfd/envs/cylinder_wake.py`
  - Re=100 cylinder wake suppression (Rabault et al. 2019 benchmark)
  - Action: cylinder rotation rate
  - Reward: drag coefficient reduction
  - Baseline: reproduce Rabault PPO result, then show analytical gradient is 10x more sample-efficient

- [ ] `diffcfd/envs/channel_flow.py`
  - 2D channel flow with blowing/suction actuators
  - Action: suction/blowing amplitude at N wall locations
  - Reward: drag reduction (skin friction coefficient)

- [ ] Benchmark: compare sample efficiency of model-free RL vs DiffCFD analytical gradients on cylinder wake

---

## v0.4 — Aerodynamic Shape Optimization

- [ ] `diffcfd/geometry/airfoil.py` — NACA + B-spline parameterization
- [ ] `diffcfd/workflows/aero.py` — drag/lift optimization workflow
- [ ] Validation: NACA0012 drag vs OpenFOAM (Re=1000, expect <3% discrepancy)
- [ ] **2026 addition**: Multi-objective Pareto optimization (drag vs lift vs structural weight)

---

## v0.5 — Neural Operator Surrogates

- [ ] Use DiffCFD as ground-truth to generate FNO training data
- [ ] Train FNO surrogate on geometry → flow field mapping
- [ ] Surrogate-in-the-loop: fast FNO prediction → periodic DiffCFD correction
- [ ] Benchmark: surrogate speed vs accuracy trade-off
- [ ] **2026 addition**: Active learning loop — DiffCFD selects informative geometries to query, reducing training data needed by ~10x

---

## v0.6 — sCO₂ Thermal-Hydraulic Module

Full integration with [sCO₂-TMSR-Toolkit](https://github.com/OpenLithoHub/sCO2-TMSR-Toolkit):

- [ ] PCHE channel shape optimization (maximize compactness factor)
  - Use `diffcfd/props/sco2.py` (from v0.2) for accurate transcritical properties
  - Geometry: B-spline parameterized semicircular channels
  - Objective: maximize heat transfer area density subject to pressure drop constraint
- [ ] Cycle-level coupled optimization:
  - DiffCFD provides CFD-level PCHE conductance as differentiable function of geometry
  - sCO₂-TMSR-Toolkit provides cycle-level efficiency as function of PCHE conductance
  - Chain rule connects geometry → CFD → cycle efficiency end-to-end
- [ ] FMU export of optimized PCHE geometry for sCO₂-TMSR-Toolkit Modelica models

---

## Patent Claims (Draft — Pre-filing, Confidential)

### C1 — Fixed-point implicit differentiation through SIMPLE-converged steady-state NS
A method for computing exact gradients of quantities of interest (drag coefficient, Nusselt number, pressure drop) with respect to design parameters (geometry, boundary conditions) through a steady-state incompressible Navier-Stokes solution obtained via SIMPLE iteration, using the implicit function theorem to compute gradients by solving a single linear system of size equal to the degrees of freedom, independent of the number of SIMPLE iterations required for convergence. Memory consumption is O(N) where N is the number of grid cells, compared to O(N·K) for direct differentiation through K SIMPLE iterations.

**Prior art gap**: Fixed-point implicit differentiation is known (Bai et al. 2019, DEQ). Its application to SIMPLE-based incompressible NS with proof that the fixed-point Jacobian is well-conditioned at the converged steady state, and the specific linear system formulation for the NS pressure-velocity coupling, is novel.

### C2 — Differentiable CFD Gymnasium environment with analytical policy gradients
A software interface wrapping a differentiable incompressible Navier-Stokes solver as a standard Gymnasium reinforcement learning environment, providing: (i) physically consistent flow-field observations from the differentiable solver, (ii) analytical policy gradients computed via automatic differentiation through the differentiable step function, and (iii) a unified interface for both model-free RL agents (which use step rollouts) and model-based policy optimization (which use the analytical gradients), enabling direct comparison of model-free vs. model-based approaches on the same physical environment.

**Prior art gap**: RL for flow control exists (Rabault 2019). Gymnasium environments wrapping differentiable CFD solvers providing analytical gradients do not exist as a published, open package. The combination of (differentiable solver) + (Gymnasium interface) + (analytical gradient export) is novel.

### C3 — Coupled geometry and boundary condition optimization with manufacturing constraints
A gradient-based optimization framework for fluid dynamic devices that simultaneously optimizes continuous geometry parameters (B-spline control points) and boundary condition parameters (inlet velocity profile, wall temperature) subject to manufacturing constraints (minimum feature size, minimum wall thickness, curvature radius), using a shared differentiable loss combining fluid dynamic objective and fabrication penalty within a single autograd computational graph.

### C4 — Differentiable neural surrogate for supercritical CO₂ transcritical properties
A differentiable neural network surrogate model for supercritical CO₂ thermophysical properties (density, enthalpy, viscosity, thermal conductivity, specific heat) in the transcritical region, trained on NIST REFPROP reference data, with physical consistency enforced by architecture constraints (monotone density via cumulative sum parameterization, positive specific heat via softplus output), enabling end-to-end gradient-based optimization of sCO₂ power cycles from CFD-level heat exchanger design parameters to system-level thermodynamic cycle efficiency.

---

## Key References

- Patankar & Spalding (1972) — SIMPLE algorithm. ← public domain, not patentable
- Bai et al. (2019) — Deep Equilibrium Models (implicit differentiation). NeurIPS 2019 ← prior art for fixed-point diff; C1 is novel application
- Rabault et al. (2019) — RL for active flow control. JFM ← prior art; C2 is novel as software architecture
- Pironneau (1974), Jameson (1988) — Adjoint CFD. ← foundational prior art, does not block C1
- JAX-Fluids paper (2024, CPC) — compressible differentiable CFD; explicitly out of scope of DiffCFD
- PhiFlow paper (Holl et al. 2020, ICLR) ← prior art; does not disclose implicit diff or Gymnasium interface

---

## Competitive Differentiation Summary

| Feature | DiffCFD | PhiFlow 3.4 | JAX-Fluids 2.0 | NVIDIA Modulus |
|---|---|---|---|---|
| Incompressible NS | ✅ | ✅ | ❌ (compressible only) | ✅ (PINN) |
| Compressible NS | ❌ | ❌ | ✅ | ✅ (surrogate) |
| PyTorch-native | ✅ | ✅ (multi-backend) | ❌ (JAX) | ✅ |
| Steady-state implicit diff (C1) | ✅ | ❌ (transient only) | ❌ | ❌ |
| Gymnasium RL interface (C2) | ✅ | ❌ | ❌ | ❌ |
| Conjugate heat transfer | ✅ | Partial | ❌ | ✅ (surrogate) |
| sCO₂ property surrogate (C4) | ✅ | ❌ | ❌ | ❌ |
| Fabrication constraints (C3) | ✅ | ❌ | ❌ | ❌ |
| Actively maintained (2026) | ✅ | ✅ | ✅ | ✅ |
