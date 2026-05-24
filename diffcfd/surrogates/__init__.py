"""Neural operator surrogates for fast flow field prediction (v0.5).

FNO (Fourier Neural Operator) learns the mapping from geometry/BC parameters
to steady-state flow fields (u, v, p), trained on DiffCFD ground-truth data.

Use cases:
  - Surrogate-in-the-loop optimization: fast FNO prediction with periodic
    DiffCFD correction for accuracy
  - Active learning: DiffCFD selects informative geometries to query,
    reducing training data needed
"""
