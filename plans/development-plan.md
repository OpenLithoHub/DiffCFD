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

**Patent strategy note (revised per legal risk analysis):**
- CN首申只覆盖v0.1已验证的**C1**（稳态隐式微分+Poiseuille解析梯度验证作为实施例）
- C2在C1申请提交后单独申请（需要v0.3 cylinder wake基准作为实施例）
- C3/C4各自单独申请，等对应milestone完成后提交
- **开源时机**：收到CN受理回执后再push，不是"申请当天"——消除时序风险，成本为零

**Do NOT push the following until CN filing:**
- `diffcfd/solvers/navier_stokes.py` — implicit differentiation through SIMPLE (core claim)
- `diffcfd/solvers/implicit_diff.py` — fixed-point differentiation primitives
- Any code implementing the claims listed in Section 5 below

---

## Competitive Landscape Analysis (as of 2026-05, fact-checked)

### Direct competitors — know before you build

| Tool | Scope | Language | Stars | Last update | Threat |
|---|---|---|---|---|---|
| **JAX-Fluids 2.0** (TU Munich) | Compressible NS, 3D, multi-phase | JAX | 580 | 2026-05 active | Medium — compressible only, JAX not PyTorch, no RL env, no heat transfer |
| **PhiFlow 3.4.0** (TU Munich) | Incompressible NS + heat, multi-backend | PyTorch/JAX/TF | 1872 | 2025-08-02 | **High** — direct overlap on incompressible NS; must differentiate clearly |
| **HydroGym** (Brunton group) | RL + CFD, 61+ envs, multi-backend | JAX/Firedrake | 121 | 2026-05 active | **High for C2** — covers Gymnasium-compatible RL+CFD with differentiable backend; see analysis below |
| **NVIDIA Modulus** | PINNs + neural operators, surrogate | PyTorch | 2830 | 2026-05 active | Medium — surrogate-focused, not true differentiable solver |
| SU2 | Compressible RANS, adjoint | C++ | ~4k | Active | Low — no ML loop integration, C++ not Python |
| OpenFOAM | Full CFD | C++ | ~2k | Active | None — production tool, not differentiable |

**PhiFlow 3.4.0 fact-check** (latest release 2025-08-02, NOT "2026-05" as previously stated):
- Covers incompressible NS with PyTorch/JAX/TF backends
- No standard `gymnasium.Env` interface
- No steady-state implicit differentiation (transient time-stepping only)
- No conjugate heat transfer workflow
- No sCO₂ property integration
- Non-standard tensor abstraction — steep learning curve vs. native PyTorch

**HydroGym fact-check** (critical — was missing from previous plan, confirmed from source code):
- ✅ Standard `gymnasium` interface for **Firedrake backend** (FEM, non-differentiable) — `core.py` imports `gymnasium as gym`, SB3 examples use `DummyVecEnv`, PPO, SAC, TD3
- ✅ 61+ environments, actively maintained (v1.0.1 released 2026-04), expanding fast
- ✅ Differentiable backends exist: JAX (Kolmogorov, channel) + JAX-Fluids (compressible only)
- ❌ **JAX backend (differentiable) uses `gymnax`, NOT standard `gymnasium`** — confirmed from JAX README: "environments follow the gymnax interface (reset_env/step_env with explicit params)"; incompatible with SB3/CleanRL without wrapper
- ❌ JAX backend uses **pseudo-spectral + Runge-Kutta-Crank-Nicolson transient** — NOT finite-volume SIMPLE; no steady-state implicit differentiation
- ❌ JAX-Fluids backend: only **2 environments, both compressible** (Shock Vector Control Ma>1, turbulent channel Re_tau=180)
- ❌ No heat transfer, no sCO₂, no fabrication constraints, no PyTorch

**Critical nuance confirmed by source code**: There are TWO separate interfaces in HydroGym:
1. **Non-differentiable backends (Firedrake, MAIA, Nek)** → standard `gymnasium` ✅ + SB3 compatible ✅ → **but NOT differentiable**
2. **Differentiable backends (JAX, JAX-Fluids)** → `gymnax` interface ❌ not standard gymnasium → **differentiable but not SB3 compatible**

The combination "differentiable solver + standard gymnasium" does NOT exist in HydroGym. This confirms the C2 claim stands, but the framing must be precise: the gap is the **intersection of differentiability and standard gymnasium**, not just one or the other.

**Revised C2 threat assessment — final**: C2 survives, but the physical kernel (FV/SIMPLE/steady-state implicit diff) remains the **primary** differentiator; the gymnasium interface is a **secondary** differentiator (HydroGym's differentiable envs use gymnax). HydroGym is rapidly expanding (L4DC 2025, v1.0.1 2026-04) — risk that they add FV incompressible backend within 12-18 months is real. **C1 is the durable claim; C2 must be filed before HydroGym closes the gap.**

**JAX-Fluids does NOT do (confirmed from README):**
- Incompressible flow (explicitly compressible-only, confirmed)
- Heat transfer
- Standard gymnasium interface (uses gymnax via HydroGym)
- PyTorch (JAX only)

**Conclusion**: DiffCFD's defensible niche is confirmed as: **(PyTorch-native) × (incompressible FV + SIMPLE) × (steady-state implicit diff, C1) × (standard gymnasium.Env, C2-revised) × (conjugate heat transfer) × (sCO₂ thermal engineering)**. This combination has no existing open-source solution.

### Patent freedom-to-operate analysis

**NOT patented (open literature prior art):**
- Adjoint method for CFD (Pironneau 1974, Jameson 1988) — foundational, not patentable by anyone
- Fixed-point implicit differentiation through PDE solutions — Bai et al. 2019 (DEQ paper), open literature
- SIMPLE pressure-velocity coupling — Patankar & Spalding 1972, public domain
- RL for flow control — Rabault et al. 2019, open literature
- HydroGym (Gymnasium + CFD combination) — open source MIT license, prior art for generic combination

**Your defensible novelty (final revised after full source-code analysis):**
- C1: Implicit differentiation through SIMPLE-converged steady-state NS — **primary claim, most durable**; HydroGym, PhiFlow, JAX-Fluids all use transient solvers; none implement this
- C2 (**final framing**): The intersection of (differentiable incompressible FV solver) + (standard `gymnasium.Env` with SB3/CleanRL compatibility) — HydroGym's split: differentiable backends use `gymnax` (not gymnasium); standard gymnasium backends are non-differentiable; the intersection is empty in HydroGym. **Filing urgency: HIGH** — HydroGym may close this gap within 12-18 months; file C2 alongside C1 in CN first application, do not defer to separate filing
- C3: Coupled shape + BC optimization with fabrication constraints
- C4: Differentiable sCO₂ transcritical property surrogate

**Filing strategy revision (per risk analysis):**
- CN首申同时覆盖C1+C2（不分开）——HydroGym扩张速度快，C2窗口可能12-18个月内关闭
- C1+C2共用同一个实施例（cylinder wake：C1提供解析梯度，C2提供gymnasium接口）
- C3/C4仍分案申请
- **开源时机**：收到CN受理回执后push，不早于此

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
  - **Analytical gradient verification (required for C1 patent claim)**:
    - Poiseuille flow pressure drop ∂ΔP/∂U_inlet has closed-form solution (ΔP = 12μLU/h²)
    - Implicit diff gradient must match analytical value to < 0.01% — stronger than `gradcheck` at 1e-4
    - Document this result explicitly in the CN patent filing as proof of "exact gradient" claim

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

### C2 — PyTorch-native incompressible FV solver as standard gymnasium.Env with steady-state analytical gradients
A software interface wrapping a **PyTorch-native incompressible finite-volume Navier-Stokes solver** as a standard `gymnasium.Env` reinforcement learning environment, providing: (i) physically consistent flow-field observations, (ii) **analytical steady-state policy gradients computed via implicit differentiation** (not finite-difference estimation and not transient-unrolling), and (iii) a unified interface compatible with Stable-Baselines3, CleanRL, and other standard RL libraries. The technical effect is a reduction of 10–50× in policy gradient sample complexity versus model-free RL on the same environment, as the analytical gradient replaces stochastic gradient estimation.

**Prior art gap (final, after source-code analysis)**: HydroGym has a split architecture — standard `gymnasium` interface exists on Firedrake/MAIA/Nek backends (non-differentiable); differentiable backends (JAX pseudo-spectral, JAX-Fluids compressible) use `gymnax` (not standard gymnasium) and are incompatible with SB3/CleanRL. The intersection of (differentiable solver) ∩ (standard gymnasium interface) is **empty in HydroGym**. DiffCFD fills this intersection with incompressible FV + SIMPLE + steady-state implicit differentiation. **CNIPA eligibility**: technical effect = 10-50× reduction in policy gradient sample steps vs model-free RL (Rabault cylinder wake, Re=100), measurable and benchmarkable.

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
- HydroGym (Clagemann et al., L4DC 2025, arXiv:2512.17534) — Gymnasium-compatible CFD RL platform; prior art for generic Gym+CFD; does NOT implement PyTorch FV + steady-state implicit diff
- PhiFlow paper (Holl & Thuerey, ICML 2024) ← prior art (note: NOT "Holl et al. 2020 ICLR", that was an older version)

---

## Competitive Differentiation Summary

**Key insight from source-code analysis**: HydroGym has a split architecture — standard gymnasium interface exists only on non-differentiable backends (Firedrake/FEM); differentiable backends (JAX) use gymnax. The intersection of "differentiable + standard gymnasium" is **empty in HydroGym**, which is the core of C2.

| Feature | DiffCFD | PhiFlow 3.4.0 | JAX-Fluids 2.0 | HydroGym (Firedrake) | HydroGym (JAX diff.) | NVIDIA Modulus |
|---|---|---|---|---|---|---|
| Incompressible FV (SIMPLE) | ✅ | ✅ (projection) | ❌ compressible | ✅ FEM (not FV) | ❌ spectral | ✅ PINN |
| PyTorch-native | ✅ | ✅ multi-backend | ❌ JAX | ❌ Firedrake | ❌ JAX | ✅ |
| Steady-state implicit diff (C1) | ✅ | ❌ transient | ❌ | ❌ | ❌ | ❌ |
| Standard gymnasium.Env | ✅ | ❌ | ❌ | ✅ | ❌ gymnax | ❌ |
| Differentiable + standard gymnasium (C2 intersection) | ✅ | ❌ | ❌ | ❌ not diff. | ❌ not gymnasium | ❌ |
| Conjugate heat transfer | ✅ | Partial | ❌ | ❌ | ❌ | ✅ surrogate |
| sCO₂ property surrogate (C4) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fabrication constraints (C3) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Actively maintained (2026) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
