"""Thread affinity utilities for Rust rayon / PyTorch interop.

Provides a context manager to temporarily reduce PyTorch intra-op threads
before calling Rust SDF (rayon-parallel) to avoid CPU oversubscription.

NOTE: In the current codebase, Rust SDF runs as geometry preprocessing in
the forward pass, not concurrently with torch.backward().  This means
CPU contention is unlikely in practice.  This module is provided as a
low-cost safety net for future codepaths where Rust and torch _do_ overlap.

Usage::

    from diffcfd.utils.threading import single_torch_thread

    with single_torch_thread():
        sdf = rust_bspline_sdf(...)  # rayon uses all cores, torch uses 1

After the ``with`` block, torch thread count is restored automatically.
"""

from __future__ import annotations

from typing import Iterator

import torch

__all__ = ["single_torch_thread"]


class single_torch_thread:
    """Context manager that sets ``torch.set_num_threads(1)`` on enter and
    restores the original count on exit.

    Use around Rust rayon-parallel calls when profiling shows CPU
    oversubscription between rayon worker threads and torch intra-op
    threads.
    """

    def __init__(self) -> None:
        self._saved: int | None = None

    def __enter__(self) -> "single_torch_thread":
        self._saved = torch.get_num_threads()
        torch.set_num_threads(1)
        return self

    def __exit__(self, *args: object) -> None:
        if self._saved is not None:
            torch.set_num_threads(self._saved)
