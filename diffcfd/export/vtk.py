"""VTK export for ParaView visualization of flow fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import Tensor


def _cell_center_velocity(ux: Tensor, uy: Tensor) -> tuple:
    """Interpolate staggered MAC velocities to cell centers."""
    ux_np = ux.detach().cpu().numpy().astype("float64")
    uy_np = uy.detach().cpu().numpy().astype("float64")
    ux_cc = 0.5 * (ux_np[:, :-1] + ux_np[:, 1:])
    uy_cc = 0.5 * (uy_np[:-1, :] + uy_np[1:, :])
    return ux_cc, uy_cc


def _cell_center_vorticity(ux_cc, uy_cc, dx: float, dy: float) -> np.ndarray:
    """Compute z-component of vorticity at cell centers: ω = ∂v/∂x - ∂u/∂y."""
    dvdx = np.zeros_like(uy_cc)
    dudy = np.zeros_like(ux_cc)
    dvdx[:, 1:-1] = (uy_cc[:, 2:] - uy_cc[:, :-2]) / (2 * dx)
    dvdx[:, 0] = (uy_cc[:, 1] - uy_cc[:, 0]) / dx
    dvdx[:, -1] = (uy_cc[:, -1] - uy_cc[:, -2]) / dx
    dudy[1:-1, :] = (ux_cc[2:, :] - ux_cc[:-2, :]) / (2 * dy)
    dudy[0, :] = (ux_cc[1, :] - ux_cc[0, :]) / dy
    dudy[-1, :] = (ux_cc[-1, :] - ux_cc[-2, :]) / dy
    return dvdx - dudy


def save_vtk(
    ux: Tensor,
    uy: Tensor,
    p: Tensor,
    mesh,
    path: str | Path,
    T: Tensor | None = None,
    extra_scalars: dict[str, Tensor] | None = None,
) -> None:
    """Write velocity and pressure fields to a VTK legacy file (.vtk).

    Outputs a RECTILINEAR_GRID VTK file readable by ParaView with:
    - Cell-centered velocity vector (interpolated from staggered MAC grid)
    - Cell-centered pressure
    - Cell-centered velocity magnitude
    - Cell-centered z-vorticity
    - Optional temperature field
    - Optional additional scalar fields

    Args:
        ux: x-velocity on x-faces, shape (ny, nx+1).
        uy: y-velocity on y-faces, shape (ny+1, nx).
        p: Pressure field tensor (ny, nx).
        mesh: CartesianMesh instance providing grid coordinates.
        path: Output file path (will add .vtk extension if missing).
        T: Optional temperature field (ny, nx).
        extra_scalars: Optional dict of name → scalar field (ny, nx).
    """
    path = Path(path)
    if path.suffix != ".vtk":
        path = path.with_suffix(".vtk")
    path.parent.mkdir(parents=True, exist_ok=True)

    ny, nx = p.shape
    dx, dy = mesh.dx, mesh.dy

    ux_cc, uy_cc = _cell_center_velocity(ux, uy)
    p_np = p.detach().cpu().numpy().astype("float64")
    vel_mag = np.sqrt(ux_cc ** 2 + uy_cc ** 2)
    vorticity = _cell_center_vorticity(ux_cc, uy_cc, dx, dy)

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("DiffCFD output\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")

        f.write(f"X_COORDINATES {nx} double\n")
        for i in range(nx):
            f.write(f"{(i + 0.5) * dx:.10e} ")
        f.write("\n")

        f.write(f"Y_COORDINATES {ny} double\n")
        for j in range(ny):
            f.write(f"{(j + 0.5) * dy:.10e} ")
        f.write("\n")

        f.write("Z_COORDINATES 1 double\n")
        f.write("0.0\n")

        f.write(f"CELL_DATA {nx * ny}\n")

        f.write("VECTORS velocity float64\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{ux_cc[j, i]:.10e} {uy_cc[j, i]:.10e} 0.0\n")

        for name, data in [("pressure", p_np), ("velocity_magnitude", vel_mag),
                           ("vorticity", vorticity)]:
            f.write(f"SCALARS {name} float64 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{data[j, i]:.10e}\n")

        if T is not None:
            T_np = T.detach().cpu().numpy().astype("float64")
            f.write("SCALARS temperature float64 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{T_np[j, i]:.10e}\n")

        if extra_scalars:
            for name, field in extra_scalars.items():
                data = field.detach().cpu().numpy().astype("float64")
                f.write(f"SCALARS {name} float64 1\n")
                f.write("LOOKUP_TABLE default\n")
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{data[j, i]:.10e}\n")
