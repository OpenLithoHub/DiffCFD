"""VTK export for ParaView visualization of flow fields."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor


def save_vtk(
    ux: Tensor,
    uy: Tensor,
    p: Tensor,
    mesh,
    path: str | Path,
    T: Tensor | None = None,
) -> None:
    """Write velocity and pressure fields to a VTK legacy file (.vtk).

    Outputs a STRUCTURED_GRID VTK file readable by ParaView.
    Tensors are detached and converted to float64 numpy before writing.

    Args:
        ux: x-velocity on x-faces, shape (ny, nx+1).
        uy: y-velocity on y-faces, shape (ny+1, nx).
        p: Pressure field tensor (ny, nx).
        mesh: CartesianMesh instance providing grid coordinates.
        path: Output file path (will add .vtk extension if missing).
        T: Optional temperature field (ny, nx).
    """
    path = Path(path)
    if path.suffix != ".vtk":
        path = path.with_suffix(".vtk")

    ny, nx = p.shape
    dx, dy = mesh.dx, mesh.dy

    ux_np = ux.detach().cpu().numpy().astype("float64")
    uy_np = uy.detach().cpu().numpy().astype("float64")
    p_np = p.detach().cpu().numpy().astype("float64")

    # Interpolate staggered velocities to cell centers
    import numpy as np

    # u_x at cell centers: average of left and right face values
    ux_cc = 0.5 * (ux_np[:, :-1] + ux_np[:, 1:])  # (ny, nx)
    # u_y at cell centers: average of bottom and top face values
    uy_cc = 0.5 * (uy_np[:-1, :] + uy_np[1:, :])  # (ny, nx)

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("DiffCFD output\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")

        # X coordinates (cell centers)
        f.write(f"X_COORDINATES {nx} double\n")
        for i in range(nx):
            f.write(f"{(i + 0.5) * dx:.10e} ")
        f.write("\n")

        # Y coordinates (cell centers)
        f.write(f"Y_COORDINATES {ny} double\n")
        for j in range(ny):
            f.write(f"{(j + 0.5) * dy:.10e} ")
        f.write("\n")

        f.write("Z_COORDINATES 1 double\n")
        f.write("0.0\n")

        f.write(f"CELL_DATA {nx * ny}\n")

        # Velocity vector
        f.write("VECTORS velocity float64\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{ux_cc[j, i]:.10e} {uy_cc[j, i]:.10e} 0.0\n")

        # Pressure scalar
        f.write("SCALARS pressure float64 1\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{p_np[j, i]:.10e}\n")

        # Temperature (optional)
        if T is not None:
            T_np = T.detach().cpu().numpy().astype("float64")
            f.write("SCALARS temperature float64 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{T_np[j, i]:.10e}\n")
