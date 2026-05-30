# ADR-002: Thread Affinity — Not Needed

## Context
P3-4 called for thread affinity investigation to reduce contention in
multi-threaded PyTorch operations during CFD solves.

## Decision
CLOSE: thread affinity is not needed. No further investment.

## Evidence
Based on `scripts/profile_results.json` (2026-05-29):
- Baseline (default threading): 0.1283s ± 0.0033s
- Pinned thread: 0.1328s ± 0.0076s (3.5% slower)
- Speedup: 0.9661 (< 1.0, threading hurt)
- Contention: < 5%

PyTorch's internal thread pool already handles parallelism adequately.
CPU-bound PyTorch ops (matmul, FFT) benefit from OpenMP/MKL threading,
not from OS-level affinity. The <5% contention confirms no significant
scheduler overhead.

## Consequences
- `diffcfd/utils/threading.py` retained as utility (no removal needed)
- No thread affinity in production code
- If future profiling at larger grid sizes (256²+) shows contention,
  reopen with fresh data

## Related
- `scripts/profile_results.json`
- `scripts/profile_thread_contention.py`
