# DiffCFD — Development Plan

**Status:** Pre-implementation planning
**Created:** 2026-05-23
**Patent strategy:** China first-filing before any core algorithm push

---

## Patent Strategy

1. Implement core algorithms locally (do NOT push until CN filing)
2. Submit China invention patent application — this establishes the **申请日 (filing date)**
3. **Open-source gate: push ONLY after receiving 申请号 + 申请日 confirmation** (NOT after 受理通知书, which arrives days/weeks after filing and is a formality check, not the priority date anchor)
4. File PCT within 12 months using CN filing date as priority base

**Legal basis for the open-source gate (must not be misunderstood):**
China Patent Law Art. 24 grace period covers ONLY four narrow categories: (1) state-of-emergency first public disclosure for public interest, (2) first display at a government-recognized international exhibition, (3) first publication at a designated academic/technical conference, (4) unauthorized disclosure by a third party without the inventor's consent. **GitHub open-source push falls into NONE of these categories.** A pre-filing GitHub push permanently destroys novelty in China with zero grace period available. There is no rescue path.

**Patent strategy note (revised per legal risk analysis):**
- CN首申只覆盖v0.1已验证的**C1**（稳态隐式微分+Poiseuille解析梯度验证作为实施例）
- C2在C1申请提交后单独申请（需要v0.3 cylinder wake基准作为实施例）
- C3/C4各自单独申请，等对应milestone完成后提交
- **开源时机**：收到CN受理回执后再push，不是"申请当天"——消除时序风险，成本为零

**Do NOT push the following until CN filing:**
- `diffcfd/solvers/navier_stokes.py` — implicit differentiation through SIMPLE (core claim)
- `diffcfd/solvers/implicit_diff.py` — fixed-point differentiation primitives
- Any code implementing the claims listed in Section 5 below

**CNIPA secrecy review (Art. 4, Patent Law):**
- CNIPA must conduct a secrecy review before granting the application; applications involving state security or significant national interest may be designated secret. In practice: pure CFD algorithms (C1, C2) and sCO₂ energy algorithms (C4) are civilian applications with zero secrecy risk. NACA airfoil (v0.4) is civilian aerodynamics with negligible risk. **TMSR / nuclear reactor context** carries slightly more scrutiny — not because C1/C2 are sensitive, but because the stated application domain (TMSR) might attract attention.
- **Mitigation (low cost, zero implementation effort)**: Do not mention TMSR, nuclear, or reactor in the DiffCFD patent claims. The patent claims should be domain-agnostic (incompressible NS, sCO₂ thermophysical properties). The sCO₂ application to nuclear systems is in the *application description*, not the claims — this is normal engineering patent practice.
- **Open-source gate (unchanged)**: wait for 申请号 + 申请日 confirmation. For pure CFD algorithms with no secrecy review expected, this is typically same-day or next-day after online filing.
- Secrecy review result: if CNIPA issues a secrecy notice (unlikely for C1-C4), do NOT push code until the notice is resolved. In all other cases, proceed with the open-source gate as described.

**Operational security — DiffCFD public repo naming:**
- The public DiffCFD repo and all code therein should be domain-agnostic: incompressible fluid dynamics, heat transfer, sCO₂ thermophysical properties as general scientific computing tools.
- **Do NOT use "TMSR", "reactor", "nuclear", or "thorium" anywhere in the public DiffCFD repository** (README, code comments, issue tracker, package metadata). These terms belong in a private or separate domain-application repo.
- The v0.6 integration with sCO₂-TMSR-Toolkit can reference "sCO₂ power cycle" or "sCO₂ thermal-hydraulic systems" without mentioning the nuclear application explicitly.
- Rationale: reduces CNIPA secrecy review trigger surface; consistent with standard practice for dual-use engineering tools.

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
- C2无宽限期退路：即便自己先开源C2实现，中国也无法靠宽限期补救（理由同上，GitHub公开不属于Art.24四类情形）。这进一步强化"C1+C2合并首申、不可拖延"的结论
- C1+C2共用同一个实施例（cylinder wake：C1提供解析梯度，C2提供gymnasium接口）
- C3/C4仍分案申请
- **开源时机**：收到**申请号 + 申请日确认**后push（不是受理通知书——受理通知书是形式审查回执，晚于申请日数天到数周，不是优先权锚点）

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
      - `torch.func.jvp/vjp` confirmed available in PyTorch 2.8
    - **Preconditioner: pyamg / scipy ILU (non-differentiable, that's fine)** — the preconditioner only needs to make GMRES converge; it does NOT need to be differentiated through. Use `pyamg` (algebraic multigrid, 646 stars, updated 2026-05-20, actively maintained) or `scipy.sparse.linalg.spilu` (ILU) as a black-box preconditioner. Block-Jacobi (previous recommendation) is adequate only at low Re; ILU/AMG is required for Re≥1000. **Acceptance gate: adjoint GMRES converges within 200 iterations at Re=1000 with pyamg/ILU preconditioner.**
    - Cross-validate: implicit diff gradients must agree with unrolled SIMPLE gradients (v0.05) to < 0.1%
  - Reference: Bai et al. 2019 (DEQ) + JFNK literature for matrix-free Krylov in CFD

- [ ] `diffcfd/solvers/boundary.py`
  - Inlet (Dirichlet velocity), outlet (zero-gradient pressure), no-slip wall, symmetry
  - All BC parameters differentiable: inlet velocity profile shape, wall temperature

- [ ] `diffcfd/geometry/mesh.py`
  - Structured Cartesian mesh with **SDF-based Brinkman penalization** instead of naive cut-cell
    - Naive cut-cell: step function at the fluid/solid boundary → gradient discontinuity → autograd fails or gives wrong gradients through geometry changes
    - Brinkman penalization: add a porosity term `(1-χ(φ)) · u / ε` to the momentum equation where `φ` is a signed distance field (SDF) and `χ` is a smooth Heaviside of the SDF — gradient is well-defined everywhere via the smooth SDF
    - SDF computed from B-spline wall geometry (differentiable via implicit function); Heaviside uses `β`-continuation (start soft, progressively sharpen)
    - This is the standard approach in differentiable topology optimization (Lazarov & Sigmund 2016); applies directly to DiffCFD's immersed boundary
  - B-spline parameterized wall: smooth geometry → SDF → Heaviside mask → differentiable penalization

- [ ] Validation suite (mandatory before CN filing):
  - **Grid convergence study** (Richardson extrapolation on lid-driven cavity): run at 32², 64², 128² to confirm mesh-independent result before reporting error vs Ghia — prevents "numbers were tuned" critique in patent examination
  - Lid-driven cavity Re=100 vs Ghia et al. 1982 (L2 error < 1%, grid-converged)
  - Lid-driven cavity Re=1000 vs Ghia et al. 1982 (L2 error < 2%, grid-converged)
  - Poiseuille flow: analytical solution comparison
  - Backward-facing step Re=800: reattachment length within 5%
  - **Gradient verification (three-layer, required for C1 patent claim)**:
    1. `torch.autograd.gradcheck` — catches implementation bugs
    2. **Complex-step derivative approximation** — machine-precision reference, eliminates FD cancellation error. **Caveat (confirmed by spike test)**: `torch.clamp` and boundary condition operations fail on `complex64` dtype. Complex-step is therefore **not applicable to the full SIMPLE forward pass**. Apply it only to isolated differentiable sub-components (pressure Poisson solve, convection term) where no clamp/relu ops exist. For full-solver gradient verification, use analytical comparison only.
    3. Analytical: Poiseuille pressure drop ∂ΔP/∂U_inlet = 12μL/h² — closed-form, must match to < 0.01%
    - All verified layers documented in CN patent filing as proof of "exact gradient" claim
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
- **Why gymnasium interface for Mode A?** Mode A is deterministic optimization, not sequential MDP. Rationale for keeping gymnasium interface: (1) enables SB3/CleanRL ecosystem reuse without modification; (2) unified API lets users switch Mode A↔B without code changes; (3) contextual bandit IS a degenerate MDP (episode length = 1) and is valid gymnasium usage. This must be explicitly stated in patent claims to address "software interface is not a technical feature" objection — the technical effect is the analytical gradient, not the interface itself.

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
  - Benchmark:
    - **Baseline**: SB3 PPO (model-free, does NOT use analytical gradients — standard gymnasium rollouts only)
    - **DiffCFD method**: Analytic Policy Gradient (APG) — uses `env.policy_gradient()` from implicit diff to directly update policy parameters without rollouts
    - Report: policy converges in 10-50× fewer `env.step()` calls with APG vs SB3 PPO
    - **Important**: SB3 PPO compatibility proves the environment is a standard gymnasium.Env; APG proves the analytical gradient is useful. These are separate claims that happen to use the same environment.
  - **This is the C2 patent embodiment** — document sample efficiency improvement with specific numbers

- [ ] `diffcfd/envs/heat_exchanger.py` — Mode A benchmark
  - Fin geometry optimization as single-step contextual bandit
  - Shows C1+C2 combination: implicit diff (C1) enables Mode A analytical gradient

- [ ] Add to C2 dependent claims (per IP strategy):
  - Dependent claim: C2 + C4 (sCO₂ property surrogate in the Gymnasium env loop)
  - Fallback if pure gymnasium interface is deemed obvious: combined C2+C4 remains novel

---

## v0.35 Milestone — Turbulence Model (Frozen Eddy Viscosity)

**Target:** 1 month after v0.3 | **Unlocks Re > ~5000 and engineering-relevant flows**

The laminar NS solver (v0.1) is limited to low-to-moderate Re (< ~2000 in 2D). Engineering flows (heat exchangers, ducts, external aero) are often turbulent. A turbulence model is needed before v0.4 aerodynamic optimization is useful.

**Approach: frozen eddy viscosity (simplest differentiable extension):**
- Run standard non-differentiable RANS solve (e.g., using k-ω SST in OpenFOAM or as a warm-start) to get the eddy viscosity field `μ_t(x)`
- Freeze `μ_t` as a non-differentiable constant (it does not participate in autograd)
- Run DiffCFD SIMPLE with effective viscosity `μ_eff = μ + μ_t` — this IS differentiable through geometry/BC
- This gives differentiable gradients w.r.t. geometry with a fixed turbulence correction
- Limitation: gradients do not account for how turbulence changes with geometry; acceptable for small design perturbations

- [ ] `diffcfd/solvers/turbulence.py` — frozen eddy viscosity loader
  - Input: eddy viscosity field `μ_t` (from file or precomputed OpenFOAM/scipy solver)
  - Output: effective viscosity tensor for use in SIMPLE momentum equation
  - Differentiable: yes (μ_t is a constant tensor; SIMPLE autograd flows through μ_eff as usual)

- [ ] (Optional / stretch) Spalart-Allmaras one-equation model — if SA equation is implemented inside the DiffCFD SIMPLE loop, μ_t becomes coupled and can be differentiated. Higher implementation cost but gives consistent gradients at moderate turbulence intensity.
  - Validation target: flat plate turbulent boundary layer (Spalart & Allmaras 1992 original test case)

- [ ] Validation: duct flow at Re=10,000 — compare Nusselt number prediction vs Dittus-Boelter correlation (within 15%); this is the acceptance gate for frozen eddy viscosity being useful for heat exchanger design

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
A software interface wrapping a PyTorch-native incompressible finite-volume Navier-Stokes solver as a standard `gymnasium.Env`, supporting two RL usage modes: (Mode A) single-step contextual bandit for geometry/parameter optimization, where each `step()` runs SIMPLE to a new steady state and returns an analytical gradient via implicit differentiation; (Mode B) sequential quasi-steady-state episode where each `step()` transitions between consecutive steady states under discrete control actions, compatible with standard RL algorithms (PPO, SAC). Both modes provide analytical policy gradients via implicit differentiation (C1). **Sample efficiency claim**: on the Rabault cylinder wake benchmark (Re=100), Analytic Policy Gradient (APG) — which uses the C1 implicit gradient instead of Monte Carlo rollouts — achieves convergence in 10-50× fewer environment interactions than model-free PPO (SB3 baseline). The speedup comes from APG **replacing** PPO with gradient-based policy updates; SB3 PPO is used as the **baseline comparator** and does not itself use or see the analytical gradients. Compatible with Stable-Baselines3 and CleanRL without modification for standard model-free RL; the analytical gradient is exported via `env.policy_gradient()` for APG use.

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
- Rabault et al. (2019) — RL for active flow control. JFM ← prior art; C2 is novel as specific combination
- Pironneau (1974), Jameson (1988) — Adjoint CFD. ← foundational prior art, does not block C1
- JAX-Fluids paper (2024, CPC) — compressible differentiable CFD; explicitly out of scope of DiffCFD
- HydroGym (Clagemann et al., L4DC 2025, arXiv:2512.17534) — Gymnasium-compatible CFD RL; differentiable backends use gymnax + transient spectral; does NOT implement PyTorch FV + steady-state implicit diff
- jax-cfd (Kochkov et al., PNAS 2021, google/jax-cfd) — **unmaintained** (README confirmed); incompressible FVM + spectral, JAX only, no gymnasium, no steady-state implicit diff
- **PhiFlow** primary citation: Holl & Thuerey, ICML 2024 ← official required citation
- **PhiFlow** secondary citation: Holl et al., ICLR 2020, "Learning to Control PDEs with Differentiable Physics" ← also required per official README; both must appear in prior art list

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

---

## Assumptions & To-Verify (Review Periodically)

This plan's competitive moat depends on the following external facts. If any is overturned, the corresponding claim strategy must be revised. Each entry shows what changes if the assumption is wrong.

| Assumption | Last verified | If overturned → |
|---|---|---|
| HydroGym differentiable backends use `gymnax`, NOT standard `gymnasium` | 2026-05-23 (source code confirmed) | C2 loses gymnasium-interface differentiator; C1 (FV/SIMPLE/steady-state) becomes sole claim |
| HydroGym has no incompressible FV/SIMPLE backend | 2026-05-23 (README + backend dirs) | C1+C2 intersection shrinks; file immediately |
| PhiFlow 3.4.0 has no steady-state implicit diff | 2026-05-23 (releases + README) | C1 threatened; verify immediately |
| PhiFlow 3.4.0 has no standard gymnasium.Env | 2026-05-23 (README) | C2 threatened |
| JAX-Fluids covers compressible only (no incompressible) | 2026-05-23 (README confirmed) | C1 scope narrows |
| jax-cfd is unmaintained | 2026-05-23 (README: "no longer maintained") | If revived, reassess threat level |
| No existing PyTorch-native incompressible FV + gymnasium + implicit diff package | 2026-05-23 | Core premise of project collapses; do a PyPI/GitHub search before filing |
| `torch.func.jvp` supports matrix-free GMRES (PyTorch 2.8) | 2026-05-23 (confirmed) | If removed in future PyTorch version, find alternative |
| `torch.clamp` fails on complex dtype | 2026-05-23 (confirmed by spike) | complex-step applicability unchanged; this blocks full-solver complex-step |
| HydroGym arXiv:2512.17534 is the correct paper ID | 2026-05-23 (README citation confirmed) | Update reference if wrong |

**Review cadence**: check HydroGym and PhiFlow release notes every 2 months. If either adds incompressible FV + gymnasium combination, accelerate CN filing immediately.
