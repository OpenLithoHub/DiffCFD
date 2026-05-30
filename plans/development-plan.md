# DiffCFD — Development Plan

**Status:** v0.7 complete
**Created:** 2026-05-23
**Updated:** 2026-05-30

---

## Strategic Context

DiffCFD is a PyTorch-native differentiable CFD framework focused on 2D incompressible Navier-Stokes with steady-state implicit differentiation. It is one of four repos in the differentiable-physics ecosystem (alongside OpenLithoHub, DiffNano, diff-surrogate).

### Resource constraints

- **Single contributor**, evening-time pace (~10–20 hours/week realistic)
- **CPU only** — no local GPU, no cloud GPU budget assumed
- All milestone timelines are part-time estimates

### Technical differentiators

DiffCFD's niche: **(PyTorch-native) × (incompressible FV + SIMPLE) × (steady-state implicit diff) × (standard gymnasium.Env) × (conjugate heat transfer) × (sCO₂ thermal engineering)**. This combination has no existing open-source solution.

---

## Competitive Landscape Analysis (as of 2026-05, fact-checked)

### Direct competitors

| Tool | Scope | Language | Threat |
|---|---|---|---|
| **JAX-Fluids 2.0** (TU Munich) | Compressible NS, 3D, multi-phase | JAX | Medium — compressible only, JAX not PyTorch, no RL env, no heat transfer |
| **PhiFlow 3.4.0** (TU Munich) | Incompressible NS + heat, multi-backend | PyTorch/JAX/TF | **High** — direct overlap on incompressible NS; must differentiate clearly |
| **HydroGym** (Brunton group) | RL + CFD, 61+ envs, multi-backend | JAX/Firedrake | **High for RL envs** — Gymnasium-compatible CFD RL with differentiable backend |
| **FluidGym** (Becktepe et al.) | AFC, 2D+3D, PISO transient, PyTorch | PyTorch | **Medium** — PyTorch-native + differentiable; no SIMPLE or implicit diff |
| **NVIDIA Modulus** | PINNs + neural operators, surrogate | PyTorch | Medium — surrogate-focused, not true differentiable solver |
| SU2 | Compressible RANS, adjoint | C++ | Low — no ML loop integration |
| OpenFOAM | Full CFD | C++ | None — production tool, not differentiable |

**PhiFlow 3.4.0 fact-check** (latest release 2025-08-02):
- Covers incompressible NS with PyTorch/JAX/TF backends
- No standard `gymnasium.Env` interface
- No steady-state implicit differentiation (transient time-stepping only)
- No conjugate heat transfer workflow
- No sCO₂ property integration
- Non-standard tensor abstraction — steep learning curve vs. native PyTorch

**FluidGym fact-check** (source code confirmed 2026-05-23, commit 5ec3a8784c3a):
- PyTorch-native, GPU-accelerated, fully open-source (MIT)
- Incompressible flow environments: cylinder wake (2D), airfoil, Rayleigh-Bénard convection
- `differentiable=True` mode: unrolls PISO time steps through autograd
- **`GymFluidEnv.step()` calls `.detach().cpu().numpy()`** — severs the autograd graph
- Solver: PISO transient only. No SIMPLE. No steady-state implicit diff.
- No heat transfer, no sCO₂, no fabrication constraints

**HydroGym fact-check** (source code re-confirmed 2026-05-23):
- Standard `gymnasium` interface for **Firedrake backend** (FEM, non-differentiable)
- 61+ environments, actively maintained (v1.0.1 released 2026-04)
- Differentiable backends (JAX) use `gymnax`, NOT standard `gymnasium`
- No heat transfer, no sCO₂, no fabrication constraints, no PyTorch

### Competitive Differentiation Summary

| Feature | DiffCFD | PhiFlow | JAX-Fluids | HydroGym (Firedrake) | HydroGym (JAX) | FluidGym | Modulus |
|---|---|---|---|---|---|---|---|
| Incompressible FV (SIMPLE) | ✅ | ✅ (projection) | ❌ | ✅ FEM | ❌ spectral | ❌ PISO | ✅ PINN |
| PyTorch-native | ✅ | ✅ multi-backend | ❌ | ❌ | ❌ | ✅ | ✅ |
| Steady-state implicit diff | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| True `gymnasium.Env` subclass | ✅ | ❌ | ❌ | ✅ | ❌ gymnax | ❌ | ❌ |
| Differentiable + true gymnasium | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Conjugate heat transfer | ✅ | Partial | ❌ | ❌ | ❌ | ❌ | ✅ surrogate |
| sCO₂ property surrogate | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fabrication constraints | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## v0.05 Internal Milestone — Unrolled SIMPLE (Completed)

**Purpose:** Validate physics correctness before implementing implicit differentiation.

- [x] Forward SIMPLE solver (unrolled, full autograd through iterations)
- [x] Lid-driven cavity Re=100 vs Ghia et al. 1982
- [x] Autograd through unrolled iterations (gradcheck passes)
- [x] Memory at N=64²: O(N·K) blowup confirmed
- [x] Anderson Acceleration (history depth m=5): 50–80% iteration count reduction
- [x] Cross-validation reference for v0.1 implicit diff

---

## v0.1 Milestone — 2D Incompressible NS + Steady-State Implicit Diff (Completed)

### Core deliverables

- [x] `diffcfd/solvers/navier_stokes_2d.py` — differentiable 2D incompressible NS
  - Finite volume on structured Cartesian grid (staggered MAC grid)
  - SIMPLE pressure-velocity coupling for steady state
  - **Implicit differentiation via fixed-point theorem**:
    - **Dual-function architecture**: forward solver uses under-relaxed SIMPLE; backward uses pure physics residual R(u, θ) = 0 without relaxation
    - Matrix-free GMRES via `torch.func.jvp` — O(N) memory
    - Preconditioner: pyamg / scipy ILU
  - Cross-validated: implicit diff gradients agree with unrolled SIMPLE to < 0.1%

- [x] `diffcfd/solvers/boundary.py` — differentiable BCs (inlet, outlet, wall, symmetry)
- [x] `diffcfd/geometry/mesh.py` — SDF-based Brinkman penalization
- [x] Validation suite:
  - Lid-driven cavity Re=100 (L2 < 1%), Re=1000 (L2 < 2%)
  - Poiseuille flow analytical comparison
  - Backward-facing step Re=800 (reattachment within 5%)
  - Gradient verification: gradcheck + Poiseuille analytical gradient (< 0.01%)
- [x] `diffcfd/export/vtk.py` — VTK export

---

## v0.2 Milestone — Heat Transfer + Heat Exchanger Optimization (Completed)

- [x] `diffcfd/solvers/heat_transfer.py` — conjugate heat transfer
- [x] `diffcfd/workflows/heat_exchanger.py` — fin geometry optimization
- [x] `diffcfd/props/sco2.py` — differentiable sCO₂ property surrogate
- [x] Validation: PCHE Nu correlation vs. Kim 2016

---

## v0.3 Milestone — Gymnasium RL Environments (Completed)

### Architecture: steady-state solver in MDP context

**Mode A — Shape/parameter optimization (single-step contextual bandit):**
- `env.step(action)` → runs SIMPLE → returns reward
- Analytical gradient flows from reward through SIMPLE via implicit diff
- RL algorithm: policy gradient with analytical gradient

**Mode B — Quasi-steady-state control (sequential episode):**
- Action changes control parameter → steady state transition
- PPO/SAC compatible

- [x] `diffcfd/envs/base.py` — base `gymnasium.Env` with `policy_gradient()`
- [x] `diffcfd/envs/cylinder_wake.py` — Mode B benchmark (Re=100)
- [x] `diffcfd/envs/heat_exchanger.py` — Mode A benchmark

---

## v0.35 Milestone — Turbulence Model (Completed)

- [x] `diffcfd/solvers/turbulence.py` — frozen eddy viscosity
- [x] Validation: duct flow Re=10,000 vs Dittus-Boelter (within 15%)
- [x] Frozen μ_t perturbation validity bound documented

---

## v0.4 — Aerodynamic Shape Optimization (Completed)

- [x] `diffcfd/geometry/airfoil.py` — NACA + B-spline parameterization
- [x] `diffcfd/workflows/aero.py` — drag/lift optimization
- [x] Validation: NACA0012 drag vs OpenFOAM (Re=1000, <3%)

---

## v0.5 — Neural Operator Surrogates (Completed)

- [x] FNO surrogate trained on DiffCFD ground-truth
- [x] Surrogate-in-the-loop: fast prediction + periodic correction
- [x] Benchmark: speed vs accuracy trade-off

---

## v0.6 — sCO₂ Thermal-Hydraulic Module (Completed)

- [x] PCHE channel shape optimization
- [x] Cycle-level coupled optimization with sCO₂-TMSR-Toolkit

---

## v0.7 — Rust-Accelerated Forward Kernels (Completed 2026-05-29)

- [x] `src/momentum.rs` — Sparse CSR momentum system assembly
- [x] `src/pressure.rs` — SIMPLE pressure correction
- [x] `src/sdf.rs` — B-spline SDF with rayon parallelism
- [x] `src/simple.rs` — Full SIMPLE forward loop with faer sparse solve
- [x] CI integration: maturin develop + Rust toolchain in GitHub Actions
- [x] Validation: 70/70 unit tests pass

---

## Key References

- Patankar & Spalding (1972) — SIMPLE algorithm
- Bai et al. (2019) — Deep Equilibrium Models (implicit differentiation). NeurIPS 2019
- Rabault et al. (2019) — RL for active flow control. JFM
- Pironneau (1974), Jameson (1988) — Adjoint CFD
- JAX-Fluids paper (2024, CPC) — compressible differentiable CFD
- HydroGym (Clagemann et al., L4DC 2025, arXiv:2512.17534)
- FluidGym (Becktepe, Franz, Thuerey, Peitz, arXiv:2601.15015, 2026)
- jax-cfd (Kochkov et al., PNAS 2021)
- PhiFlow: Holl & Thuerey, ICML 2024; Holl et al., ICLR 2020
