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

## v0.05 Internal Milestone — Unrolled SIMPLE (Risk Gate, Not Released)

**Target:** 3-4 weeks | **Internal only — validates physics before implicit diff**

This phase is not a public release. It de-risks v0.1 by separating two problems:
(a) correctness of the forward NS solver and boundary conditions
(b) correctness of the implicit differentiation

- [ ] Implement forward SIMPLE solver (unrolled, full autograd through iterations)
- [ ] Validate forward field: lid-driven cavity Re=100 vs Ghia et al. 1982
- [ ] Confirm autograd works through unrolled iterations (gradcheck passes)
- [ ] Measure memory at N=64²: confirm O(N·K) blowup as expected
- [ ] This unrolled version is the **cross-validation reference** for v0.1 implicit diff
- [ ] Do NOT commit to main — keep on `dev/unrolled` branch, never push to public

---

## v0.1 Milestone — 2D Incompressible NS + Steady-State Implicit Diff

**Target:** 2-3 months after v0.05 | **Gate for CN patent filing**

### Core deliverables

- [ ] `diffcfd/solvers/navier_stokes_2d.py` — differentiable 2D incompressible NS
  - Finite volume on structured Cartesian grid (staggered MAC grid)
  - SIMPLE pressure-velocity coupling for steady state
  - **Implicit differentiation via fixed-point theorem** (C1):
    - Do NOT use `torch.linalg.solve` (dense LAPACK — O(N²) memory, defeats the purpose)
    - Correct approach: **matrix-free GMRES via `torch.func.jvp`**
      - The implicit gradient equation `(∂F/∂u)ᵀ · λ = ∂L/∂u` is solved with GMRES
      - Matvec oracle = `lambda v: torch.func.jvp(F, u, v)[1]` — never materializes Jacobian
      - Memory: O(N) (only flow field + GMRES Krylov vectors, ~10-50 vectors)
      - This is the correct O(N) implementation — `torch.func.jvp/vjp` confirmed available in PyTorch 2.8
    - Cross-validate: implicit diff gradients must agree with unrolled SIMPLE gradients (v0.05) to < 0.1%
  - Reference: Bai et al. 2019 (DEQ) + JFNK literature for matrix-free Krylov in CFD

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
  - **Gradient verification (three-layer, required for C1 patent claim)**:
    1. `torch.autograd.gradcheck` — catches implementation bugs
    2. Complex-step derivative approximation — machine-precision reference, eliminates finite-difference cancellation error; industry gold standard for gradient verification
    3. Analytical: Poiseuille pressure drop ∂ΔP/∂U_inlet = 12μL/h² — closed-form, must match to < 0.01%
    - All three layers documented in CN patent filing as proof of "exact gradient" claim
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

**This is the key differentiator from PhiFlow and JAX-Fluids.**

### Architecture note: steady-state solver in MDP context

Standard Gymnasium is designed for transient MDP (sequential state transitions). A steady-state solver creates a conceptual mismatch that must be explicitly handled. DiffCFD supports two modes:

**Mode A — Shape/parameter optimization (single-step contextual bandit):**
- `env.step(action)` changes geometry/BC parameters → runs SIMPLE to new steady state → returns reward
- Each `step()` = one full steady-state solve
- Analytical gradient flows directly from reward through SIMPLE via implicit diff (C1)
- Use case: heat exchanger fin shape, airfoil drag minimization
- RL algorithm: policy gradient with analytical gradient (not PPO/SAC — those are for Mode B)

**Mode B — Quasi-steady-state control (sequential episode):**
- Action changes a control parameter (e.g., inlet velocity, cylinder rotation)
- Steady state A → control change → steady state B = one MDP step
- Natural episode length: N control steps before terminal condition
- Use case: cylinder wake suppression, active flow control
- RL algorithm: PPO/SAC compatible (standard Gymnasium episode structure)
- Analytical gradient also available per step (C2 claim)

Both modes share the same `gymnasium.Env` interface. Mode A/B selected via `env_config`.

- [ ] `diffcfd/envs/base.py` — base `gymnasium.Env`
  - `step()` supports both Mode A (single-step) and Mode B (sequential)
  - `policy_gradient()` — analytical gradient via implicit diff (C2)
  - Compatible with Stable-Baselines3, CleanRL (standard gymnasium)

- [ ] `diffcfd/envs/cylinder_wake.py` — Mode B benchmark
  - Re=100, rotating cylinder action (Rabault et al. 2019 baseline)
  - Benchmark: DiffCFD analytical gradient vs PPO vs HydroGym-Firedrake PPO
  - **This is the C2 patent embodiment** — document sample efficiency improvement

- [ ] `diffcfd/envs/heat_exchanger.py` — Mode A benchmark
  - Fin geometry optimization as single-step contextual bandit
  - Shows C1+C2 combination: implicit diff (C1) enables Mode A analytical gradient

- [ ] Add to C2 dependent claims (per IP strategy):
  - Dependent claim: C2 + C4 (sCO₂ property surrogate in the Gymnasium env loop)
  - Fallback if pure gymnasium interface is deemed obvious: combined C2+C4 remains novel

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

**sCO₂ open-source strategy (C4 commercial value protection)**:
- v0.2 open-sources ideal gas / incompressible thermal solver (no commercial risk)
- `diffcfd/props/sco2.py` (transcritical surrogate, C4) is **NOT open-sourced at v0.2**
- C4 remains private until: (a) CN patent for C4 receives receipt (受理通知书), then open-source; OR (b) v0.6 milestone, whichever comes first
- Rationale: C4 has standalone commercial value for energy system optimization (sCO₂ power cycle, heat pump, supercritical extraction) — independent of the CFD framework

---

## Patent Claims (Draft — Pre-filing, Confidential)

### C1 — Fixed-point implicit differentiation through SIMPLE-converged steady-state NS
A method for computing exact gradients of quantities of interest (drag coefficient, Nusselt number, pressure drop) with respect to design parameters (geometry, boundary conditions) through a steady-state incompressible Navier-Stokes solution obtained via SIMPLE iteration, using the implicit function theorem to compute gradients by solving a single linear system of size equal to the degrees of freedom, independent of the number of SIMPLE iterations required for convergence. Memory consumption is O(N) where N is the number of grid cells, compared to O(N·K) for direct differentiation through K SIMPLE iterations.

**Prior art gap**: Fixed-point implicit differentiation is known (Bai et al. 2019, DEQ). Its application to SIMPLE-based incompressible NS with proof that the fixed-point Jacobian is well-conditioned at the converged steady state, and the specific linear system formulation for the NS pressure-velocity coupling, is novel.

### C2 — PyTorch-native incompressible FV solver as standard gymnasium.Env with steady-state analytical gradients
A software interface wrapping a PyTorch-native incompressible finite-volume Navier-Stokes solver as a standard `gymnasium.Env`, supporting two RL usage modes: (Mode A) single-step contextual bandit for geometry/parameter optimization, where each `step()` runs SIMPLE to a new steady state and returns an analytical gradient via implicit differentiation; (Mode B) sequential quasi-steady-state episode where each `step()` transitions between consecutive steady states under discrete control actions, compatible with standard RL algorithms (PPO, SAC). Both modes provide analytical policy gradients via implicit differentiation (C1), reducing policy gradient sample complexity by 10-50× versus model-free RL. Compatible with Stable-Baselines3 and CleanRL without modification.

**Prior art gap (final, after source-code analysis)**: HydroGym has a split architecture — standard `gymnasium` interface exists on Firedrake/MAIA/Nek backends (non-differentiable); differentiable backends (JAX pseudo-spectral, JAX-Fluids compressible) use `gymnax` and are incompatible with SB3/CleanRL. The intersection of (differentiable FV steady-state solver) ∩ (standard gymnasium) ∩ (analytical gradient export) is empty in all prior work. **CNIPA eligibility**: technical effect = 10-50× sample complexity reduction on Rabault cylinder wake (Re=100), measurable.

**Dependent claim (fallback if pure gymnasium interface deemed obvious)**: C2 combined with C4 — sCO₂ transcritical property surrogate integrated in the gymnasium env reward loop; this combination remains novel independently.

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
- HydroGym (Clagemann et al., L4DC 2025, arXiv:2512.17534) — Gymnasium-compatible CFD RL; differentiable backends use gymnax (not gymnasium) + transient spectral; does NOT implement PyTorch FV + steady-state implicit diff
- jax-cfd (Kochkov et al., PNAS 2021, google/jax-cfd) — **unmaintained** (confirmed from README: "no longer maintained"); incompressible FVM + spectral, JAX only, no gymnasium, no steady-state implicit diff; cited for completeness
- PhiFlow paper (Holl & Thuerey, ICML 2024) ← prior art (note: NOT "Holl et al. 2020 ICLR")

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
