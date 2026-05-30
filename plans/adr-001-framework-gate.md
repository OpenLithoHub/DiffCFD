# ADR-001: Framework Decision -- Stay on PyTorch Unified Graph

**Date:** 2026-05-30
**Status:** ACCEPTED
**Deciders:** project lead

## Context

The strategy document (S3.4) required "freezing the JAX question until Phase 2 exit (P2-5 benchmark production)." Phase 2 benchmark data now exists across two independent test suites:

1. **DiffNano flagship metalens benchmark** (`flagship_metalens_results.json`) -- 150-step coupled vs. decoupled co-design on a 20x20 grid at 100 nm pixel size.
2. **diff-surrogate codesign benchmark** (`codesign_benchmark_results.json`) -- 200-step runs across two problem classes (quadratic coupling, geometry co-design) with four strategies each (coupled, decoupled alternating, decoupled sequential, random baseline).

These benchmarks validate that the PyTorch-based co-design pipeline works end-to-end: gradient flow through coupled physics domains is correct, Adam optimization converges, and the `CoDesignWorkflow` API is production-ready.

## Decision

**STAY on PyTorch unified graph.** JAX integration only via dlpack interop (`diff_surrogate/interop.py`, planned P3-6).

## Rationale

### 1. PyTorch co-design workflow produces correct gradient flow and converges

The flagship metalens benchmark demonstrates that coupled litho+optics optimization works correctly in pure PyTorch:

- **Coupled mode:** loss dropped from 1.289 to 0.788 over 150 steps (38.9% reduction), achieving optical_loss=0.612, litho_epe=1.794, fab_penalty=11.466 in 1.03 s wall time.
- **Decoupled mode:** loss dropped from 3.982 to 0.084 over 150 steps (97.9% reduction), achieving optical_loss=0.963, litho_epe=2.617, fab_penalty=5.293 in 0.20 s wall time.

Both modes show smooth, monotonically decreasing loss curves in their primary phases with no NaN events, confirming stable autograd through the coupled forward pass and correct backpropagation through the `CoDesignWorkflow` pipeline (Adam + grad clipping + param clamping).

The diff-surrogate codesign benchmark reinforces this across two distinct problem classes:

**Quadratic coupling (200 steps):**
- Coupled strategy: 28.5 -> 7.903 (72.3% reduction), wall time 0.085 s
- Decoupled alternating: best_loss=1.732 at step 198, wall time 0.055 s
- Decoupled sequential: best_loss=1.223 at step 99, wall time 0.057 s
- Random baseline: no convergence (final loss 32.87), confirming optimization actually works

**Geometry co-design (200 steps):**
- Coupled strategy: 1.400 -> 0.310 (77.9% reduction), wall time 0.410 s
- Decoupled alternating: best_loss=0.119 at step 198, wall time 0.352 s
- Decoupled sequential: best_loss=0.086 at step 99, wall time 0.358 s
- Random baseline: final loss 1.094 (essentially no improvement from start 1.438)

Gradient norms decay smoothly in all optimization-active strategies (e.g., coupled geometry: 1.230 -> 0.091 over 200 steps), confirming no gradient explosion or vanishing issues.

### 2. JAX interop via dlpack is production-ready for edge cases

`interop.py` provides zero-copy JAX<->PyTorch bridging:

- `j2t()` / `t2j()`: dlpack-based zero-copy tensor conversion (JAX >= 0.4, PyTorch >= 1.10).
- `JAXFunctionWrapper(torch.autograd.Function)`: wraps any JAX function as a PyTorch-autograd-compatible callable with full forward+backward via `jax.vjp`. This means a JAX-trained surrogate model can be inserted into a PyTorch optimization loop with correct gradient propagation -- the primary use case for JAX in this ecosystem.
- All JAX imports are lazy; the module is importable without JAX installed, and `wrap_jax_fn()` produces a clean `torch.Tensor -> torch.Tensor` callable.

This covers the only scenario where JAX adds value: consuming a pre-trained JAX surrogate inside the PyTorch co-design loop. No migration needed.

### 3. All four repos are PyTorch-native

- **diff-surrogate:** `CoDesignWorkflow`, `CoupledLoss`, all forward functions, Adam optimizer -- pure PyTorch.
- **OpenLithoHub:** lithography simulation pipeline -- PyTorch.
- **DiffCFD:** differentiable CFD -- PyTorch.
- **DiffNano:** nanophotonics optimization -- PyTorch.

A JAX migration would require rewriting the autograd graph, optimizer loops, parameter management, and gradient clipping in all four codebases. No marginal benefit justifies this cost.

### 4. Migration cost is massive with zero user benefit

The benchmarks show the PyTorch pipeline is:
- **Fast enough:** wall times range from 0.055 s to 0.410 s for 200-step co-design runs on these problem sizes. No performance bottleneck is attributable to the framework choice.
- **Correct:** smooth convergence, no NaN events, monotonically decaying gradient norms across all optimization strategies.
- **Feature-complete:** `CoDesignWorkflow` supports coupled/decoupled modes, weighted loss composition, gradient clipping, param bounds, and comparison reporting.

JAX offers no user-visible advantage that justifies the engineering cost of rewriting four codebases.

### 5. The dlpack interop is marked :stable: and handles the JAX surrogate insertion use case

The `JAXFunctionWrapper` has a clear contract:
- Forward: torch tensor -> dlpack -> jax -> jax_fn -> dlpack -> torch tensor
- Backward: torch grad -> dlpack -> jax -> vjp -> dlpack -> torch grad

This is the only integration point needed. The `interop.py` module is self-contained (129 lines), has no external dependencies beyond PyTorch and optional JAX, and handles the memory layout edge cases (contiguous enforcement, gradient detachment).

## Consequences

- JAX benefits are accessed only through the dlpack thin boundary (`diff_surrogate/interop.py`).
- No JAX migration is planned for any of the four repos.
- The framework gate is **CLOSED** and **FROZEN** as of 2026-05-30.
- Any future re-evaluation requires new benchmark data showing a PyTorch-specific bottleneck that JAX would resolve, which does not exist in current data.

## Data Sources

| File | Key Metrics |
|------|-------------|
| `DiffNano/flagship_metalens_results.json` | Coupled: 1.289->0.788 loss, 1.03 s; Decoupled: 3.982->0.084 loss, 0.20 s |
| `diff-surrogate/benchmarks/results/codesign_benchmark_results.json` | Quadratic: 28.5->7.903 (coupled), 7.903->0.085 s; Geometry: 1.400->0.310 (coupled), 0.410 s |
| `diff-surrogate/diff_surrogate/codesign.py` | `CoDesignWorkflow` API (marked `:stable:`) |
| `diff-surrogate/diff_surrogate/interop.py` | `JAXFunctionWrapper` + `j2t`/`t2j` dlpack interop |
