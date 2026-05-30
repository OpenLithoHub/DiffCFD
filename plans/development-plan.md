# DiffCFD — Development Plan

**Status:** v0.7 complete, pre-filing phase
**Created:** 2026-05-23
**Updated:** 2026-05-29
**Patent strategy:** China first-filing, then PCT, then public push

---

## Strategic Context (read this first)

DiffCFD sits inside a two-project portfolio decision the author made on 2026-05-23, after a comparative analysis of three candidate projects (DiffCFD, DiffNano, OpenLithoHub). The conclusion that frames everything below:

- **OpenLithoHub** (computational lithography, GitHub-public 2026-05) is in **damage-control mode**: CN/EP novelty is already lost because GitHub self-publication is not covered by China Patent Law Art. 24 or by EPO grace period; surviving routes are US 35 USC §102(b)(1) (12-month inventor self-disclosure grace), JP/KR (12-month), and TW (12-month). The action there is a US Provisional filing within ~51 weeks of the first relevant commit, run by patent counsel — no further code work required.
- **DiffNano** (differentiable EM with cross-domain DFM coupling) was **dropped**: requires multi-GPU 3D FDTD experiments to produce CN-filing embodiments; the author has only a single CPU laptop. The defensible claim space (C4 + C5) also shrank to two pillars after TORCWA / TorchRDIT / FDTDX / tidy3d.plugins.autograd / meent prior-art review.
- **DiffCFD** is the **prime focus**: 2D incompressible NS at 64²–128² runs on a CPU laptop without GPU, the patent corpus is still untouched (no public code), and four claim pillars (C1 + C2 + C3 + C4) survive freedom-to-operate analysis.

**One-line summary**: OpenLithoHub is a hold-the-line operation handled by counsel; DiffCFD is where new engineering effort goes.

### Resource constraints driving the plan

- **Single contributor**, evening-time pace (~10–20 hours/week realistic)
- **CPU only** — no local GPU, no cloud GPU budget assumed; HuggingFace free tier is not adequate for sustained training runs
- **No team** for parallel benchmarking, validation, or documentation
- All milestone timelines below have been re-scaled from the original full-time estimates

The original DiffCFD plan estimated v0.05 in "3-4 weeks" and v0.1 in "2-3 months" — those numbers assumed full-time engineering. Re-scaled to part-time: **v0.05 ≈ 8–12 weeks, v0.1 ≈ 14–20 weeks after v0.05**, total wall-clock to CN-filing-ready ≈ **22–32 weeks**.

### EP-preserving release sequence (revised — replaces "push after 申请号" rule)

The original plan said "push to public GitHub after CN 申请号 + 申请日 confirmation." That rule is **insufficient** — it preserves CN priority but **destroys EP novelty**, because EPO has no self-disclosure grace period and a public push between CN filing and PCT filing makes the inventor's own code prior art against the EP national-stage entry.

**Revised release sequence:**

1. Implement v0.05 + v0.1 locally on `dev/*` branches; **`main` stays code-empty**
2. Submit CN invention patent application; receive **申请号 + 申请日 written confirmation** (not 受理通知书)
3. **File PCT international application within 12 months** of CN priority date, using CN as the Paris Convention priority base
4. **Only after PCT submission is confirmed** push solver code publicly to `main`
5. Continue PCT national-stage entries (US, JP, KR, TW, EP) at 30–31 months from CN priority

The CN→PCT→push order preserves EP novelty for the duration of the PCT-stage 30-month window. If the public push happens between step 2 and step 3, EP rights are lost permanently.

This rule applies to **all** code that embodies any of C1–C4. Domain-agnostic supporting code (validation harness, VTK export, geometry parameterization without claim-bearing solver internals) can release earlier under a separate review.

---

## Patent Strategy

1. Implement core algorithms locally (do NOT push until **PCT** filing — see "EP-preserving release sequence" above)
2. Submit China invention patent application — this establishes the **申请日 (filing date)**
3. Receive **申请号 + 申请日 written confirmation** (NOT 受理通知书, which arrives days/weeks after filing and is a formality check, not the priority date anchor)
4. File PCT within 12 months using CN filing date as priority base — **only after PCT is on file may the code be pushed publicly**, otherwise EP novelty is destroyed

**Legal basis for the open-source gate (must not be misunderstood):**
China Patent Law Art. 24 grace period covers ONLY four narrow categories: (1) state-of-emergency first public disclosure for public interest, (2) first display at a government-recognized international exhibition, (3) first publication at a designated academic/technical conference, (4) unauthorized disclosure by a third party without the inventor's consent. **GitHub open-source push falls into NONE of these categories.** A pre-filing GitHub push permanently destroys novelty in China with zero grace period available. There is no rescue path.

**⚠️ Disclaimer**: The legal analysis in this section (Art. 24, Art. 4 secrecy review, Art. 25 hardware binding) represents the author's understanding for planning purposes and is NOT legal advice. All patent strategy decisions must be confirmed with a qualified Chinese patent attorney before filing. In particular: (a) the Art. 25 "hardware binding" framing for algorithm claims should be reviewed by a CNIPA-registered agent; (b) the secrecy review risk assessment is based on general knowledge and must be confirmed for the specific application domain.

**Patent strategy note (revised per legal risk analysis):**
- CN首申只覆盖v0.1已验证的**C1**（稳态隐式微分+Poiseuille解析梯度验证作为实施例）
- C2在C1申请提交后单独申请（需要v0.3 cylinder wake基准作为实施例）
- C3/C4各自单独申请，等对应milestone完成后提交
- **开源时机**：**等到 PCT 提交确认后**再 push（不是 CN 申请号确认那一刻）。CN 申请号锁定的是 CN 优先权和 PCT 优先权基础；EP 没有自主公开宽限期，CN 提交后到 PCT 提交前的窗口期 push 会让自己的 GitHub 代码成为 EP 端的现有技术。受理通知书是形式审查回执，不是 PCT 触发条件。

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
| **FluidGym** (Becktepe et al.) | AFC, 2D+3D, PISO transient, PyTorch | PyTorch | 41 | 2026-05 active | **Medium for C2** — PyTorch-native + differentiable + "gymnasium-like API"; does NOT implement steady-state SIMPLE or implicit diff; see analysis below |
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

**FluidGym fact-check** (source code confirmed 2026-05-23, commit 5ec3a8784c3a, arXiv:2601.15015):
- ✅ PyTorch-native, GPU-accelerated, fully open-source (MIT), actively maintained
- ✅ Incompressible flow environments: cylinder wake (2D), airfoil, Rayleigh-Bénard convection, turbulent channel flow
- ✅ `differentiable=True` mode: unrolls PISO time steps through autograd
- ✅ PPO and SAC baselines provided, pre-trained models on HuggingFace
- ✅ **Has a true `gymnasium.Env` subclass**: `GymFluidEnv(gymnasium.Env)` in `src/fluidgym/integration/gymnasium.py` (line 13: `class GymFluidEnv(Env)` where `Env` is `from gymnasium import Env`). Also `VecFluidEnv(SB3VecEnv)` for SB3. The README claim of "Gymnasium / SB3 integration" is accurate.
- ❌ **Solver: PISOtorch / PICT — PISO algorithm, explicitly transient**. Docstring line 1: *"FluidGym simulation class based on PISOtorch implemented in PICT."* `PISOtorch` is the real Python class (imported from `fluidgym.simulation.extensions`); PICT (arXiv:2505.16992) is the underlying TU Munich solver project. NOT SIMPLE. No fixed-point solve. C1 is not threatened.
- ❌ **`GymFluidEnv.step()` calls `.detach().cpu().numpy()` on every output** — severs the autograd graph. `VecFluidEnv` (SB3) does the same. FluidGym has the **identical structural split as HydroGym**: differentiable mode (native `FluidEnv`) vs gymnasium-compatible mode (`GymFluidEnv`) are mutually exclusive — you cannot have both simultaneously.
- ❌ No steady-state implicit differentiation — differentiable mode unrolls all PISO time steps (O(N·T) memory)
- ❌ No heat transfer, no sCO₂, no fabrication constraints

**FluidGym threat assessment (corrected)**: The earlier claim "FluidGym is not a gymnasium.Env subclass" was wrong — it has `GymFluidEnv(Env)`. The correct framing: **FluidGym's gymnasium wrapper detaches gradients in step(); its differentiable mode does not expose a gymnasium.Env interface. SB3-compatible and differentiable are mutually exclusive in FluidGym.** C2 gap is structural (gradient detachment), not nominal. The C2 claim must be reframed accordingly.

**Key insight now consistent across all competitors — the universal split**:
- HydroGym: differentiable = gymnax interface (not `gymnasium.Env`); gymnasium-compatible = Firedrake (not differentiable)
- FluidGym: differentiable = native `FluidEnv` (gradient preserved); gymnasium-compatible = `GymFluidEnv` (`.detach()` in `step()`, gradient severed)
- DiffCFD (proposed): differentiable implicit diff + `gymnasium.Env` subclass where `step()` returns **gradient-attached tensors** — this intersection is empty in all prior work

**Revised C2 patent framing**: The claim is NOT "we subclass gymnasium.Env" (FluidGym does that too, with detach). The claim is: "a `gymnasium.Env` subclass where the differentiable solver's computational graph is preserved through `step()` — no `.detach()` — enabling `policy_gradient()` to return exact analytical gradients." This is the precise, verifiable, source-code-defensible gap.

**HydroGym fact-check** (source code re-confirmed 2026-05-23):
- ✅ Standard `gymnasium` interface for **Firedrake backend** (FEM, non-differentiable) — `core.py` imports `gymnasium as gym`, SB3 examples use `DummyVecEnv`, PPO, SAC, TD3
- ✅ 61+ environments, actively maintained (v1.0.1 released 2026-04), expanding fast
- ✅ Differentiable backends exist: JAX (Kolmogorov, channel) + JAX-Fluids (compressible only)
- ❌ **JAX backend (differentiable) uses `gymnax`, NOT standard `gymnasium`** — re-confirmed from current source code: `hydrogym/jax/env_core.py` imports `from gymnax.environments import environment, spaces`; `JAXFlowEnv` inherits `gymnax.environments.environment.Environment`, not `gymnasium.Env`. This is still gymnax as of 2026-05-23.
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
- **开源时机**：**等到 PCT 提交确认后**再 push（不是 CN 申请号确认那一刻）。CN 申请号锁定的是 CN 优先权和 PCT 优先权基础；EP 没有自主公开宽限期，CN 提交后到 PCT 提交前的窗口期 push 会让自己的 GitHub 代码成为 EP 端的现有技术。受理通知书是形式审查回执，不是 PCT 触发条件。

---

## v0.05 Internal Milestone — Unrolled SIMPLE (Risk Gate, Not Released)

**Target:** 8-12 weeks part-time (single contributor, evenings) | **Internal only — validates physics before implicit diff**

This phase is not a public release. It de-risks v0.1 by separating two problems:
(a) correctness of the forward NS solver and boundary conditions
(b) correctness of the implicit differentiation

- [x] Implement forward SIMPLE solver (unrolled, full autograd through iterations)
- [x] Validate forward field: lid-driven cavity Re=100 vs Ghia et al. 1982
- [x] Confirm autograd works through unrolled iterations (gradcheck passes)
- [x] Measure memory at N=64²: confirm O(N·K) blowup as expected
- [x] **Anderson Acceleration (optional, add after baseline converges)**:
  - Wrap the SIMPLE velocity/pressure update loop with Anderson mixing (history depth m=5)
  - Implementation: maintain the last m residual vectors; solve a small m×m least-squares at each iteration to extrapolate the next iterate
  - Expected gain: 50–80% reduction in iteration count to convergence (Anderson 1965; well-documented for fixed-point solvers)
  - This is a pure engineering optimization — not a patent claim, no novelty requirement
  - **Compatibility with C1**: Anderson acceleration changes how you reach the fixed point, not the fixed point itself. The implicit gradient in v0.1 is computed at the converged state using the unrelaxed physics residual R — Anderson does not affect this. The two are fully orthogonal.
  - Acceptance gate: measure wall-clock convergence time on lid-driven cavity Re=100 and Re=1000 with and without Anderson; report speedup factor
- [x] This unrolled version is the **cross-validation reference** for v0.1 implicit diff
- [x] Do NOT commit to main — keep on `dev/unrolled` branch, never push to public

---

## v0.1 Milestone — 2D Incompressible NS + Steady-State Implicit Diff

**Target:** 14-20 weeks part-time after v0.05 | **Gate for CN patent filing**

### Core deliverables

- [x] `diffcfd/solvers/navier_stokes_2d.py` — differentiable 2D incompressible NS
  - Finite volume on structured Cartesian grid (staggered MAC grid)
  - SIMPLE pressure-velocity coupling for steady state
  - **Implicit differentiation via fixed-point theorem** (C1) — **dual-function architecture (critical)**:
    - **Forward pass**: under-relaxed SIMPLE iteration to convergence, optionally with Anderson Acceleration (history depth m=5) for faster convergence. Under-relaxation (α ≈ 0.3–0.7 on velocity, 0.1–0.3 on pressure) is required for stability; without it, SIMPLE diverges at Re > ~100. **Anderson acceleration is a forward-only component — it must not be applied to or confused with the backward residual function.**
    - **Backward pass (implicit gradient)**: the fixed-point is defined by the **pure physics residual** `R(u*, θ) = 0`, NOT by the relaxed iteration map. This distinction is critical: under-relaxation modifies the iteration `u_{k+1} = u_k + α·(F(u_k) - u_k)` but does NOT change the fixed point `u*` (where `F(u*) = u*`). The Jacobian `∂R/∂u` evaluated at `u*` must be computed from the unrelaxed physics residual `R`. **If you naively use the relaxed iteration as the operator in `jvp`, the Jacobian is wrong by a factor related to (1-α); gradients will be systematically off.** Solution: maintain two separate functions — `simulate(theta)` (forward, uses relaxation) and `residual(u, theta)` (pure NS residual, no relaxation) — and pass only `residual` to the GMRES `jvp` oracle.
    - This **"dual-function architecture"** must be stated explicitly in the C1 patent claim as a distinguishing technical detail: it is the specific implementation that ensures the implicit gradient is exact at the converged steady state. A corollary of this architecture: the backward pass depends only on the value of the physics residual `R(u*, θ)` at the converged fixed point, not on the iteration trajectory taken to reach it. This means the forward solver is free to use any convergence acceleration technique (Anderson Acceleration, momentum restart, etc.) without affecting gradient correctness — a property that should be noted in the C1 claim description as evidence of the method's engineering generality.
    - Do NOT use `torch.linalg.solve` (dense LAPACK — O(N²) memory, defeats the purpose)
    - Correct backward: **matrix-free GMRES via `torch.func.jvp(residual, u, v)[1]`**
      - The adjoint equation `(∂R/∂u)ᵀ · λ = ∂L/∂u` is solved with GMRES using the matvec oracle
      - Memory: O(N) (only flow field + GMRES Krylov vectors, ~10-50 vectors)
      - `torch.func.jvp/vjp` confirmed available in PyTorch 2.8
    - **Preconditioner: pyamg / scipy ILU** — non-differentiable, that's fine; preconditioner only needs to make GMRES converge, not be differentiated through. Use `pyamg` (algebraic multigrid) or `scipy.sparse.linalg.spilu` (ILU). **Acceptance gate: adjoint GMRES converges within 200 iterations at Re=1000.**
    - **Brinkman stiffness warning**: when Brinkman penalization is active (ε ~ 1e-3 to 1e-5), the zero-order `u/ε` term creates condition number O(1/ε) in the momentum Jacobian. At ε=1e-3, this is manageable with AMG. At ε=1e-5 (needed for sharp interfaces), standard pyamg may fail to precondition adequately. Mitigation: use ε=1e-3 as default and increase only if interface sharpness is insufficient. **Acceptance gate must be tested with Brinkman active (ε=1e-3), not just on pure fluid domains** — this is also a prerequisite for C3 (topology optimization relies on Brinkman): if implicit diff fails under Brinkman, C3 is unsupported.
    - Cross-validate: implicit diff gradients must agree with unrolled SIMPLE gradients (v0.05) to < 0.1%
  - Reference: Bai et al. 2019 (DEQ) + JFNK literature for matrix-free Krylov in CFD

- [x] `diffcfd/solvers/boundary.py`
  - Inlet (Dirichlet velocity), outlet (zero-gradient pressure), no-slip wall, symmetry
  - All BC parameters differentiable: inlet velocity profile shape, wall temperature

- [x] `diffcfd/geometry/mesh.py`
  - Structured Cartesian mesh with **SDF-based Brinkman penalization** instead of naive cut-cell
    - Naive cut-cell: step function at the fluid/solid boundary → gradient discontinuity → autograd fails or gives wrong gradients through geometry changes
    - Brinkman penalization: add a porosity term `(1-χ(φ)) · u / ε` to the momentum equation where `φ` is a signed distance field (SDF) and `χ` is a smooth Heaviside of the SDF — gradient is well-defined everywhere via the smooth SDF
    - SDF computed from B-spline wall geometry (differentiable via implicit function); Heaviside uses `β`-continuation (start soft, progressively sharpen)
    - This is the standard approach in differentiable topology optimization (Lazarov & Sigmund 2016); applies directly to DiffCFD's immersed boundary
  - B-spline parameterized wall: smooth geometry → SDF → Heaviside mask → differentiable penalization

- [x] Validation suite (mandatory before CN filing):
  - **Grid convergence study** (Richardson extrapolation on lid-driven cavity): run at 32², 64², 128² to confirm mesh-independent result before reporting error vs Ghia — prevents "numbers were tuned" critique in patent examination
  - Lid-driven cavity Re=100 vs Ghia et al. 1982 (L2 error < 1%, grid-converged)
  - Lid-driven cavity Re=1000 vs Ghia et al. 1982 (L2 error < 2%, grid-converged)
  - Poiseuille flow: analytical solution comparison
  - Backward-facing step Re=800: reattachment length within 5%
  - **Gradient verification (for C1 patent claim)**:
    1. `torch.autograd.gradcheck` — catches implementation bugs; finite-difference precision (~1e-5)
    2. **Complex-step** — applicable only to isolated sub-components without clamp/relu (confirmed: `torch.clamp` fails on complex64). **NOT applicable to the full SIMPLE forward pass.** Scope: pressure Poisson solve, convection term in isolation.
    3. Analytical: Poiseuille pressure drop ∂ΔP/∂U_inlet = 12μL/h² — **primary proof of gradient exactness for the full solver** (closed-form, must match to < 0.01%)
    - In the CN patent filing, describe the gradient proof honestly: "analytical comparison (Poiseuille, <0.01%) is the primary full-solver verification; complex-step provides sub-component verification." Do NOT claim three independent full-solver verification layers when layer 2 only covers sub-components — a false claim of "three-layer full-solver proof" is attackable in examination.

- [x] `diffcfd/export/vtk.py` — VTK export for ParaView visualization

- [ ] **Pre-filing novelty search (blocking gate — must complete before CN filing)** (last checked 2026-05-23):
  - Search PyPI for `pip search differentiable navier-stokes gymnasium` equivalents (PyPI search, GitHub search)
  - Check: HydroGym latest release notes — has `gymnasium.Env` subclass been added to any differentiable backend?
  - Check: FluidGym latest release — has SIMPLE steady-state been added? Has `gymnasium.Env` subclassing been added?
  - Check: PhiFlow latest release — has steady-state implicit diff been added?
  - Document results with dates; if any competitor closes the gap, consult patent attorney before filing
  - **This is a blocking gate, not a note** — assign a date by v0.1 completion (not "do a search before filing" as an afterthought)

---

## v0.2 Milestone — Heat Transfer + Heat Exchanger Optimization

**Target:** 2 months after v0.1

- [x] `diffcfd/solvers/heat_transfer.py`
  - Conjugate heat transfer (energy equation coupled with NS)
  - Differentiable Nusselt number output
  - Steady-state implicit diff extended to coupled NS + energy system

- [x] `diffcfd/workflows/heat_exchanger.py`
  - Fin geometry optimization: maximize Nu / pressure_drop (performance factor)
  - Fabrication constraint: minimum fin thickness (analog to MRC in OpenLithoHub)
  - Pareto front: Nu vs pressure drop for varying fin shapes

- [x] **sCO₂ property module** (feeds into v0.6 plan):
  - `diffcfd/props/sco2.py` — differentiable sCO₂ property surrogate
  - Trained against NIST REFPROP data in transcritical region (0.9Tc–1.1Tc)
  - Physical consistency: monotone density, positive Cp (enforced by architecture)
  - This is C4 — the differentiable transcritical surrogate

- [x] Validation: PCHE (printed circuit heat exchanger) Nu correlation vs. Kim 2016

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

- [x] `diffcfd/envs/base.py` — base `gymnasium.Env`
  - `step()` supports both Mode A (single-step) and Mode B (sequential)
  - `policy_gradient()` — analytical gradient via implicit diff (C2)
  - Compatible with Stable-Baselines3, CleanRL (standard gymnasium)
  - **Mode A SB3 config (required to avoid NaN)**: SB3 PPO with default settings (gamma=0.99, gae_lambda=0.95, n_steps=2048) will produce NaN or fail to converge for Mode A (episode length=1), because the value network bootstraps from a future that doesn't exist. Provide a `MODE_A_SB3_CONFIG` preset: `gamma=0`, `gae_lambda=0`, `n_steps=1` — this degenerates PPO to pure policy gradient on immediate reward (contextual bandit). Document this in the env's `__init__` docstring with a warning if the user instantiates Mode A without this config.
  - **APG reparameterization (required for stochastic policy with APG)**: `policy_gradient()` returns `dL/da` (gradient of physical loss w.r.t. action). If the policy is **deterministic** (`a = π(s; θ)` directly), the chain rule `dL/dθ = dL/da · da/dθ` applies directly — no reparameterization needed. If the policy is **stochastic** (`a ~ N(μ(s;θ), σ²)`), the sampling step cuts the autograd graph. Solution: use the **reparameterization trick** (`a = μ(s;θ) + σ·ε`, ε ~ N(0,1)) so `da/dθ` flows through the mean network. Both deterministic and reparameterized stochastic policies must be supported. The Rabault cylinder benchmark uses a deterministic policy; document which mode is used in the C2 embodiment to avoid patent ambiguity.

- [x] `diffcfd/envs/cylinder_wake.py` — Mode B benchmark
  - Re=100, rotating cylinder action (Rabault et al. 2019 baseline)
  - Benchmark:
    - **Baseline**: SB3 PPO (model-free, does NOT use analytical gradients — standard gymnasium rollouts only)
    - **DiffCFD method**: Analytic Policy Gradient (APG) — uses `env.policy_gradient()` from implicit diff to directly update policy parameters without rollouts
    - Report: policy converges in 10-50× fewer `env.step()` calls with APG vs SB3 PPO
    - **Important**: SB3 PPO compatibility proves the environment is a standard gymnasium.Env; APG proves the analytical gradient is useful. These are separate claims that happen to use the same environment.
  - **This is the C2 patent embodiment** — document sample efficiency improvement with specific numbers

- [x] `diffcfd/envs/heat_exchanger.py` — Mode A benchmark
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

- [x] `diffcfd/solvers/turbulence.py` — frozen eddy viscosity loader
  - Input: eddy viscosity field `μ_t` (from file or precomputed OpenFOAM/scipy solver)
  - Output: effective viscosity tensor for use in SIMPLE momentum equation
  - Differentiable: yes (μ_t is a constant tensor; SIMPLE autograd flows through μ_eff as usual)

- [ ] (Optional / stretch) Spalart-Allmaras one-equation model — if SA equation is implemented inside the DiffCFD SIMPLE loop, μ_t becomes coupled and can be differentiated. Higher implementation cost but gives consistent gradients at moderate turbulence intensity.
  - Validation target: flat plate turbulent boundary layer (Spalart & Allmaras 1992 original test case)

- [x] Validation: duct flow at Re=10,000 — compare Nusselt number prediction vs Dittus-Boelter correlation (within 15%); this is the acceptance gate for frozen eddy viscosity being useful for heat exchanger design

- [x] **Frozen μ_t perturbation validity bound (required for downstream v0.4/v0.6 use)**:
  - Frozen eddy viscosity produces correct gradient direction ONLY for small geometry perturbations — for large changes, μ_t should be recomputed
  - Determine the perturbation bound: run paired tests where (a) gradient is computed with frozen μ_t, (b) gradient is computed by re-solving RANS after each geometry step; compare gradient direction (cosine similarity). Report the geometry perturbation magnitude at which cosine similarity drops below 0.9.
  - Document this bound explicitly in v0.35 API and in v0.4/v0.6 optimization workflows: "frozen μ_t valid for perturbations < X% of characteristic length"
  - In v0.4 airfoil and v0.6 PCHE: add a spot-check step every N optimization iterations where RANS is re-solved to verify the frozen μ_t gradient direction is still reliable

---

## v0.4 — Aerodynamic Shape Optimization

- [x] `diffcfd/geometry/airfoil.py` — NACA + B-spline parameterization
- [x] `diffcfd/workflows/aero.py` — drag/lift optimization workflow
- [x] Validation: NACA0012 drag vs OpenFOAM (Re=1000, expect <3% discrepancy)
- [ ] **2026 addition**: Multi-objective Pareto optimization (drag vs lift vs structural weight)

---

## v0.5 — Neural Operator Surrogates

- [x] Use DiffCFD as ground-truth to generate FNO training data
- [x] Train FNO surrogate on geometry → flow field mapping
- [x] Surrogate-in-the-loop: fast FNO prediction → periodic DiffCFD correction
- [x] Benchmark: surrogate speed vs accuracy trade-off
- [ ] **2026 addition**: Active learning loop — DiffCFD selects informative geometries to query, reducing training data needed by ~10x

---

## v0.6 — sCO₂ Thermal-Hydraulic Module

Full integration with [sCO₂-TMSR-Toolkit](https://github.com/OpenLithoHub/sCO2-TMSR-Toolkit):

- [x] PCHE channel shape optimization (maximize compactness factor)
  - Use `diffcfd/props/sco2.py` (from v0.2) for accurate transcritical properties
  - Geometry: B-spline parameterized semicircular channels
  - Objective: maximize heat transfer area density subject to pressure drop constraint
- [x] Cycle-level coupled optimization:
  - DiffCFD provides CFD-level PCHE conductance as differentiable function of geometry
  - sCO₂-TMSR-Toolkit provides cycle-level efficiency as function of PCHE conductance
  - Chain rule connects geometry → CFD → cycle efficiency end-to-end
- [ ] FMU export of optimized PCHE geometry for sCO₂-TMSR-Toolkit Modelica models

**sCO₂ open-source strategy (C4 commercial value protection)**:
- v0.2 open-sources ideal gas / incompressible thermal solver (no commercial risk)
- `diffcfd/props/sco2.py` (transcritical surrogate, C4) is **NOT open-sourced at v0.2**
- C4 remains private until: (a) CN patent for C4 receives **申请号 + 申请日 confirmation** (NOT 受理通知书), then open-source; OR (b) v0.6 milestone, whichever comes first
- Rationale: C4 has standalone commercial value for energy system optimization (sCO₂ power cycle, heat pump, supercritical extraction) — independent of the CFD framework

**Plugin architecture for C4 commercial isolation (implement from v0.2)**:
- From v0.2 onwards, all fluid property calls go through an abstract interface: `class ThermophysicalProps(ABC): density, viscosity, conductivity, cp` — these are abstract methods
- The open-source core (`diffcfd`) depends only on this abstract interface, never on any specific implementation
- C4 lives in a separate private repo `diffcfd-sco2-pro` that implements `ThermophysicalProps` for sCO₂
- v0.6 integration imports `diffcfd-sco2-pro` as an optional dependency; the core lib does not depend on it
- This architecture: (a) cleanly separates open-source core from commercial module at the Python import level; (b) allows dual-licensing where core is Apache-2.0 and `diffcfd-sco2-pro` is commercial; (c) prevents "open-source community forking the commercial module" since it's in a separate repo with a different license
- Design this abstract interface at v0.2 (when the first property call appears), NOT at v0.6 — retrofitting an interface into an existing codebase is expensive

---

## v0.7 — Rust-Accelerated Forward Kernels (Completed 2026-05-29)

Forward SIMPLE kernels (momentum assembly, pressure system, SDF computation) migrated to Rust via PyO3/maturin. Backward pass (implicit diff) remains pure PyTorch for vjp compatibility.

- [x] `src/momentum.rs` — Sparse CSR momentum system assembly
- [x] `src/pressure.rs` — SIMPLE pressure correction system assembly
- [x] `src/sdf.rs` — B-spline SDF with rayon parallelism
- [x] `src/simple.rs` — Full SIMPLE forward loop with faer sparse solve
- [x] `src/lib.rs` — PyO3 module bridge
- [x] CI integration: maturin develop + Rust toolchain in GitHub Actions
- [x] Validation: 70/70 unit tests pass, all validation tests pass

## Patent Claims (Draft — Pre-filing, Confidential)

### C1 — Fixed-point implicit differentiation through SIMPLE-converged steady-state NS
A method for computing exact gradients of quantities of interest (drag coefficient, Nusselt number, pressure drop) with respect to design parameters (geometry, boundary conditions) through a steady-state incompressible Navier-Stokes solution obtained via SIMPLE iteration, using the implicit function theorem to compute gradients by solving a single linear system of size equal to the degrees of freedom, independent of the number of SIMPLE iterations required for convergence. Memory consumption is O(N) where N is the number of grid cells, compared to O(N·K) for direct differentiation through K SIMPLE iterations.

**Dual-function architecture (key claim detail)**: The method maintains two distinct computational functions: (1) a forward solver using under-relaxed SIMPLE iteration (under-relaxation factors on velocity and pressure required for numerical stability); (2) a pure physics residual function R(u, θ) = 0 without relaxation, used exclusively for the implicit gradient computation. The implicit gradient is computed by solving (∂R/∂u)ᵀ λ = ∂L/∂u via matrix-free GMRES using Jacobian-vector products of R (not of the relaxed iteration). This separation ensures that the gradient is exact at the converged steady state despite the use of relaxation in the forward pass — a distinction not present in naive differentiation through relaxed iteration.

**Prior art gap**: Fixed-point implicit differentiation is known (Bai et al. 2019, DEQ). Its application to SIMPLE-based incompressible NS requires (a) the dual-function separation of relaxed forward solver and unrelaxed physics residual, (b) proof that the fixed-point Jacobian ∂R/∂u is well-conditioned at the converged steady state, and (c) the specific matrix-free Krylov formulation for the NS pressure-velocity coupling. None of these three elements appear in prior work.

**Hardware binding for CNIPA eligibility**: Claims are framed as "a GPU tensor computation method for computing gradients of fluid dynamic quantities of interest via matrix-free Krylov-based implicit differentiation through SIMPLE-converged incompressible Navier-Stokes, wherein the Jacobian-vector products are evaluated using automatic differentiation primitives (JVP) on GPU tensor arrays, with O(N) memory complexity." The hardware and memory framing ties the algorithm to specific technical effects and prevents Art. 25 "mathematical method" rejection.

### C2 — PyTorch-native incompressible FV solver as standard gymnasium.Env with steady-state analytical gradients
A `gymnasium.Env` subclass (gymnasium ≥ 0.26) wrapping a PyTorch-native incompressible finite-volume Navier-Stokes solver, where `step()` returns gradient-attached PyTorch tensors — no `.detach()` call — preserving the autograd computational graph through the differentiable solver. This enables `policy_gradient()` to return exact analytical gradients of the reward with respect to policy parameters via implicit differentiation (C1). The environment supports two RL modes: (Mode A) single-step contextual bandit for geometry/parameter optimization; (Mode B) sequential quasi-steady-state episode compatible with standard RL algorithms (PPO, SAC). Compatible with Stable-Baselines3 and CleanRL for model-free RL; the analytical gradient is exported via `env.policy_gradient()` for APG use.

**Prior art gap — structural (source-code verified, not README-level)**:
- HydroGym (commit bf2c2dd2, 2026-05-12): differentiable backends (JAX) use `gymnax.environments.environment.Environment`, NOT `gymnasium.Env`; no `.step()` in gymnasium format
- FluidGym (commit 5ec3a8784c, 2026-05-06): `GymFluidEnv(gymnasium.Env)` subclass EXISTS but `step()` calls `.detach().cpu().numpy()` on all outputs — autograd graph severed; differentiable mode exposed only through native `FluidEnv` (non-gymnasium)
- In both tools: (differentiable mode) and (gymnasium.Env-compatible mode) are mutually exclusive
- DiffCFD is the first tool where a `gymnasium.Env` subclass preserves the autograd graph through `step()`

**CNIPA eligibility**: technical effect = (1) 10-50× sample complexity reduction vs SB3 PPO baseline on Rabault cylinder wake (Re=100), measurable; (2) exact analytical gradient (O(N) memory) vs O(N·T) unrolled PISO in FluidGym, measurable.

**Dependent claim (fallback if pure gymnasium interface deemed obvious)**: C2 combined with C4 — sCO₂ transcritical property surrogate integrated in the gymnasium env reward loop; this combination remains novel independently.

### C3 — Coupled geometry and boundary condition optimization with manufacturing constraints
A gradient-based optimization framework for fluid dynamic devices that simultaneously optimizes continuous geometry parameters (B-spline control points) and boundary condition parameters (inlet velocity profile, wall temperature) subject to manufacturing constraints (minimum feature size, minimum wall thickness, curvature radius), using a shared differentiable loss combining fluid dynamic objective and fabrication penalty within a single autograd computational graph.

**Differentiable geometric filter (required for CNIPA eligibility)**: Manufacturing constraints are enforced via a **PDE-based Helmholtz filter** (∂²ρ_filtered/∂x² - r²·∇²ρ_filtered = ρ, where r is the minimum length scale radius) applied to the density field before the Heaviside projection and Brinkman penalization. This filter is implemented as a differentiable sparse linear solve; its gradient flows through the entire shape→filter→Heaviside→Brinkman→NS→objective chain. Alternatively, morphological erosion/dilation via differentiable convolution achieves the same effect. The specific mathematical operator used (Helmholtz filter or morphological convolution) must be stated in the claim — a general "penalty term" is not sufficient for CNIPA eligibility and will be rejected as lacking a "technical solution."

**Hardware binding (CNIPA anti-algorithm-rejection)**: The claim is framed as a GPU tensor computation method: "a method for computing, on a graphics processing unit with parallel tensor memory layout, the gradient of a fluid dynamic objective with respect to geometry control parameters through a chain comprising: (1) PDE Helmholtz filter as differentiable sparse linear solve; (2) smooth Heaviside projection; (3) Brinkman momentum penalty; (4) SIMPLE pressure-velocity iteration; (5) implicit differentiation via matrix-free Krylov solver; all executed within a single PyTorch autograd computational graph." The hardware framing prevents the "mathematical method" rejection under CN Patent Law Art. 25.

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
- **FluidGym** (Becktepe, Franz, Thuerey, Peitz, arXiv:2601.15015, 2026) — PyTorch-native PISO transient differentiable CFD + "gymnasium-like" (duck-typed) RL env; does NOT implement SIMPLE steady-state or true `gymnasium.Env` subclass; prior art for PyTorch differentiable CFD benchmarking, does not anticipate C1 or C2
- jax-cfd (Kochkov et al., PNAS 2021, google/jax-cfd) — **unmaintained** (README confirmed); incompressible FVM + spectral, JAX only, no gymnasium, no steady-state implicit diff
- **PhiFlow** primary citation: Holl & Thuerey, ICML 2024 ← official required citation
- **PhiFlow** secondary citation: Holl et al., ICLR 2020, "Learning to Control PDEs with Differentiable Physics" ← also required per official README; both must appear in prior art list

---

## Competitive Differentiation Summary

**Key insight from source-code analysis**: HydroGym has a split architecture — standard gymnasium interface exists only on non-differentiable backends (Firedrake/FEM); differentiable backends (JAX) use gymnax, confirmed from current source code. FluidGym uses PISO transient (not SIMPLE steady-state) and a duck-typed `FluidEnvLike` protocol (not a `gymnasium.Env` subclass). The intersection of "differentiable + true gymnasium.Env subclass + SIMPLE steady-state" is **empty in all prior work**, which is the core of C1+C2.

| Feature | DiffCFD | PhiFlow 3.4.0 | JAX-Fluids 2.0 | HydroGym (Firedrake) | HydroGym (JAX diff.) | FluidGym | NVIDIA Modulus |
|---|---|---|---|---|---|---|---|
| Incompressible FV (SIMPLE) | ✅ | ✅ (projection) | ❌ compressible | ✅ FEM (not FV) | ❌ spectral | ❌ PISO transient | ✅ PINN |
| PyTorch-native | ✅ | ✅ multi-backend | ❌ JAX | ❌ Firedrake | ❌ JAX | ✅ | ✅ |
| Steady-state implicit diff (C1) | ✅ | ❌ transient | ❌ | ❌ | ❌ | ❌ unrolled PISO | ❌ |
| True `gymnasium.Env` subclass | ✅ | ❌ | ❌ | ✅ | ❌ gymnax | ❌ FluidEnvLike | ❌ |
| Differentiable + true gymnasium subclass (C2 intersection) | ✅ | ❌ | ❌ | ❌ not diff. | ❌ not gymnasium | ❌ not gymnasium | ❌ |
| Conjugate heat transfer | ✅ | Partial | ❌ | ❌ | ❌ | ❌ | ✅ surrogate |
| sCO₂ property surrogate (C4) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fabrication constraints (C3) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Actively maintained (2026) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

*Table verified: 2026-05-23 from source code for HydroGym (jcallaham/hydrogym) and FluidGym (safe-autonomous-systems/fluidgym). Review cadence: every 2 months.*

---

## Assumptions & To-Verify (Review Periodically)

This plan's competitive moat depends on the following external facts. If any is overturned, the corresponding claim strategy must be revised. Each entry shows what changes if the assumption is wrong.

| Assumption | Last verified | Source anchor | If overturned → |
|---|---|---|---|
| HydroGym differentiable backends use `gymnax`, NOT standard `gymnasium` | 2026-05-23 | jcallaham/hydrogym commit bf2c2dd2, `hydrogym/jax/env_core.py` line 22 | C2 loses gymnasium-interface differentiator; C1 (FV/SIMPLE/steady-state) becomes sole claim |
| HydroGym has no incompressible FV/SIMPLE backend | 2026-05-23 | README + backend dirs | C1+C2 intersection shrinks; file immediately |
| FluidGym uses PISO transient (not SIMPLE steady-state) | 2026-05-23 | safe-autonomous-systems/fluidgym commit 5ec3a8784c, `simulation.py` docstring | C1 directly threatened; file immediately |
| FluidGym's `GymFluidEnv.step()` detaches gradients (`.detach().cpu().numpy()`) | 2026-05-23 | `src/fluidgym/integration/gymnasium.py` `__to_np()` method | C2 core distinction weakens; need alternative framing |
| PhiFlow 3.4.0 has no steady-state implicit diff | 2026-05-23 | releases + README | C1 threatened; verify immediately |
| PhiFlow 3.4.0 has no standard gymnasium.Env | 2026-05-23 | README | C2 threatened |
| JAX-Fluids covers compressible only (no incompressible) | 2026-05-23 | README confirmed | C1 scope narrows |
| jax-cfd is unmaintained | 2026-05-23 | README: "no longer maintained" | If revived, reassess threat level |
| No existing tool where differentiable solver + gymnasium.Env + no .detach() in step() | 2026-05-23 | HydroGym + FluidGym source code | Core premise of C2 collapses; do PyPI/GitHub search before filing |
| `torch.func.jvp` supports matrix-free GMRES (PyTorch 2.8) | 2026-05-23 | confirmed | If removed in future PyTorch version, find alternative |
| `torch.clamp` fails on complex dtype | 2026-05-23 | confirmed by spike | complex-step applicability unchanged; this blocks full-solver complex-step |
| HydroGym arXiv:2512.17534 is the correct paper ID | 2026-05-23 | README citation confirmed | Update reference if wrong |

**Review cadence**: check HydroGym and FluidGym release notes every 2 months (assigned: project owner). If either adds incompressible FV + gymnasium combination with gradient-attached step(), accelerate CN filing immediately. Record new commit hashes and dates on each review.
