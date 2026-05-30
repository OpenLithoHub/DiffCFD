"""Geometric primitives: SDFs for common shapes used in DiffCFD environments.

All SDF functions return a signed distance field on the mesh cell centers.
Convention: positive inside fluid, negative inside solid.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffcfd.geometry.mesh import CartesianMesh


def cylinder_sdf(
    mesh: CartesianMesh,
    center_x: float,
    center_y: float,
    radius: float,
) -> Tensor:
    """SDF for a circular cylinder (2D cross-section).

    Positive inside fluid (outside cylinder), negative inside solid.
    """
    x, y = mesh.cell_centers()
    dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return dist - radius


def channel_sdf(
    mesh: CartesianMesh,
    wall_y_bottom: float = 0.0,
    wall_y_top: float | None = None,
) -> Tensor:
    """SDF for channel walls (flat plates at y=bottom and y=top).

    For pure channel flow (no immersed body), returns all-positive SDF
    (entire domain is fluid). The wall positions are handled by the
    boundary condition system rather than Brinkman penalization.
    """
    return torch.ones(mesh.ny, mesh.nx, device=mesh.device)


def rectangle_sdf(
    mesh: CartesianMesh,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> Tensor:
    """SDF for an axis-aligned rectangle (solid region).

    Negative inside solid, positive outside.
    """
    x, y = mesh.cell_centers()
    dx = torch.maximum(x_min - x, x - x_max)
    dy = torch.maximum(y_min - y, y - y_max)
    outside = torch.sqrt(torch.clamp(dx, min=0) ** 2 + torch.clamp(dy, min=0) ** 2)
    inside = torch.clamp(torch.maximum(dx, dy), max=0)
    return outside + inside


def naca0012_sdf(
    mesh: CartesianMesh,
    chord: float = 1.0,
    leading_edge_x: float = 0.25,
    center_y: float = 0.5,
    angle_deg: float = 0.0,
    thickness: float = 0.12,
) -> Tensor:
    """SDF for a NACA symmetric airfoil.

    Args:
        mesh: CartesianMesh instance.
        chord: Chord length.
        leading_edge_x: x-position of leading edge.
        center_y: y-position of chord line.
        angle_deg: Angle of attack in degrees.
        thickness: Maximum thickness as fraction of chord (default 0.12 for NACA 0012).
    """
    x, y = mesh.cell_centers()
    theta = torch.tensor(angle_deg * 3.14159265 / 180.0, device=mesh.device)

    # Rotate coordinates into airfoil frame
    dx = x - (leading_edge_x + chord / 2)
    dy = y - center_y
    x_rot = dx * torch.cos(theta) + dy * torch.sin(theta) + chord / 2
    y_rot = -dx * torch.sin(theta) + dy * torch.cos(theta)

    # NACA thickness distribution: t/c = 5*t*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)
    xn = x_rot / chord
    xn = torch.clamp(xn, 0.0, 1.0)
    half_thickness = (
        chord
        * thickness
        / 0.2
        * (
            0.2969 * torch.sqrt(xn + 1e-10)
            - 0.1260 * xn
            - 0.3516 * xn**2
            + 0.2843 * xn**3
            - 0.1015 * xn**4
        )
    )

    # SDF: distance to nearest surface point (approximate)
    inside_x = (x_rot >= 0) & (x_rot <= chord)
    y_dist = torch.abs(y_rot) - half_thickness
    signed_dist = torch.where(
        inside_x,
        y_dist,
        torch.sqrt(
            torch.minimum(
                (x_rot) ** 2 + y_rot**2,
                (x_rot - chord) ** 2 + y_rot**2,
            )
        ),
    )
    return signed_dist
