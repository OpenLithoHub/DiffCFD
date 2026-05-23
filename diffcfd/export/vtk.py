"""VTK export for ParaView visualization of flow fields."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor


def save_vtk(u: Tensor, p: Tensor, mesh, path: str | Path) -> None:
    """Write velocity and pressure fields to a VTK legacy file (.vtk).

    Outputs a structured grid VTK file readable by ParaView.
    Tensors are detached and converted to float32 numpy before writing.

    Args:
        u: Velocity field tensor (2, ny, nx) — [u_x, u_y].
        p: Pressure field tensor (ny, nx).
        mesh: CartesianMesh instance providing grid coordinates.
        path: Output file path (will add .vtk extension if missing).
    """
    raise NotImplementedError("Implement VTK export in v0.1.")
