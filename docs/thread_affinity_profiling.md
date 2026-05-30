# Thread Affinity Profiling

## Background

DiffCFD uses a hybrid architecture: Rust (via rayon) for geometry/SDF
computations and PyTorch for the differentiable solver. When both run
concurrently, they can oversubscribe CPU cores, causing thread contention.

The `single_torch_thread` context manager in
`diffcfd.utils.threading` reduces PyTorch intra-op threads to 1
during Rust-heavy sections, freeing cores for rayon.

## Profiling Method

The script `scripts/profile_thread_contention.py` runs a representative
workload (channel flow forward + backward on a 32x16 grid) in two modes:

1. **Baseline**: default PyTorch thread count
2. **Pinned**: `single_torch_thread` active during forward + backward

It measures wall time, CPU utilization, and computes the speedup ratio.

## Results

Running on a machine with 8+ CPU cores, typical results show:

| Metric | Baseline | With single_torch_thread | Change |
|--------|----------|--------------------------|--------|
| Wall time | ~1.5 s | ~1.5 s | < 5% |
| CPU utilization | ~60% | ~45% | -15% |

### Verdict: Thread affinity is NOT needed

Contention is under 5% because of the execution model:

- **Rust SDF runs during geometry preprocessing** in the forward pass,
  not concurrently with `torch.backward()`.
- **The backward pass uses implicit differentiation (GMRES)**, which is
  purely PyTorch/scipy -- no rayon involvement.
- **Sequential execution** means there is no actual overlap between Rust
  and PyTorch threads in the current codebase.

## When to Re-evaluate

Thread affinity may become relevant if future changes introduce:

1. **Concurrent Rust + PyTorch execution** (e.g., async geometry updates
   during optimization)
2. **Pipeline-parallel architectures** where one thread runs Rust SDF
   while another runs torch.autograd
3. **Multi-objective optimizations** with mixed Rust/PyTorch workloads

If any of these are introduced, re-run the profiling script:

```bash
python scripts/profile_thread_contention.py
```

Results are saved to `scripts/profile_results.json`.
