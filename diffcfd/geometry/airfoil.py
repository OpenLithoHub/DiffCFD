"""Aerodynamic shape parameterization: NACA 4-digit and B-spline airfoils.

Provides differentiable geometry → SDF → Brinkman pipeline for airfoil
shape optimization within the DiffCFD SIMPLE framework.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.geometry.shapes import naca0012_sdf


class NACA4Digit:
    """NACA 4-digit airfoil parameterization.

    Generates airfoil SDF from NACA parameters (m, p, t) where:
    - m: maximum camber (fraction of chord)
    - p: location of maximum camber (fraction of chord)
    - t: maximum thickness (fraction of chord)

    Args:
        chord: Chord length.
        leading_edge_x: x-position of leading edge.
        center_y: y-position of chord line.
        angle_deg: Angle of attack in degrees.
    """

    def __init__(
        self,
        chord: float = 1.0,
        leading_edge_x: float = 0.25,
        center_y: float = 0.5,
        angle_deg: float = 0.0,
    ) -> None:
        self.chord = chord
        self.le_x = leading_edge_x
        self.cy = center_y
        self.angle = angle_deg

    def sdf(
        self,
        mesh: CartesianMesh,
        thickness: float = 0.12,
        camber: float = 0.0,
        camber_pos: float = 0.4,
    ) -> Tensor:
        """Generate SDF for the airfoil.

        Args:
            mesh: CartesianMesh instance.
            thickness: Maximum thickness as fraction of chord.
            camber: Maximum camber as fraction of chord.
            camber_pos: Location of max camber as fraction of chord.

        Returns:
            SDF tensor (ny, nx). Positive in fluid, negative in solid.
        """
        if camber < 1e-8:
            # Symmetric airfoil (NACA 00xx)
            return naca0012_sdf(
                mesh, self.chord, self.le_x, self.cy, self.angle
            )

        # Cambered airfoil: compute camber line and add thickness
        x, y = mesh.cell_centers()
        theta = torch.tensor(
            self.angle * 3.14159265 / 180.0, device=mesh.device
        )

        # Rotate coordinates
        dx = x - (self.le_x + self.chord / 2)
        dy = y - self.cy
        x_rot = dx * torch.cos(theta) + dy * torch.sin(theta) + self.chord / 2
        y_rot = -dx * torch.sin(theta) + dy * torch.cos(theta)

        # Normalized x/c
        xn = x_rot / self.chord
        xn = torch.clamp(xn, 0.0, 1.0)

        # Camber line
        m = camber
        p = camber_pos
        yc = torch.where(
            xn < p,
            m / p ** 2 * (2 * p * xn - xn ** 2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xn - xn ** 2),
        )

        # Thickness distribution (same as NACA 0012 scaled)
        t = thickness
        yt = t / 0.2 * self.chord * (
            0.2969 * torch.sqrt(xn + 1e-10)
            - 0.1260 * xn
            - 0.3516 * xn ** 2
            + 0.2843 * xn ** 3
            - 0.1015 * xn ** 4
        )

        # SDF: distance from (x_rot, y_rot) to nearest surface point
        inside_x = (x_rot >= 0) & (x_rot <= self.chord)
        y_dist = torch.abs(y_rot - yc) - yt
        signed_dist = torch.where(
            inside_x,
            y_dist,
            torch.sqrt(
                torch.minimum(
                    x_rot ** 2 + (y_rot - yc) ** 2,
                    (x_rot - self.chord) ** 2 + (y_rot - yc) ** 2,
                )
            ),
        )
        return signed_dist


class BSplineAirfoil:
    """B-spline parameterized airfoil.

    Uses control points to define a closed airfoil curve, then computes
    SDF via distance to the curve. Control points are differentiable
    parameters for shape optimization.

    Args:
        n_control_points: Number of control points per surface (upper/lower).
        chord: Chord length.
        leading_edge_x: x-position of leading edge.
        center_y: y-position of chord line.
    """

    def __init__(
        self,
        n_control_points: int = 8,
        chord: float = 1.0,
        leading_edge_x: float = 0.25,
        center_y: float = 0.5,
    ) -> None:
        self.n_cp = n_control_points
        self.chord = chord
        self.le_x = leading_edge_x
        self.cy = center_y

    def initial_control_points(self) -> Tensor:
        """Generate initial control points for a NACA 0012-like shape.

        Returns:
            Control points tensor (2*n_cp, 2) — [x, y] coordinates.
        """
        n = self.n_cp
        t = torch.linspace(0, 1, n + 1)[:-1]

        # NACA 0012 thickness distribution for upper surface
        yt = 0.12 / 0.2 * (
            0.2969 * torch.sqrt(t + 1e-10)
            - 0.1260 * t
            - 0.3516 * t ** 2
            + 0.2843 * t ** 3
            - 0.1015 * t ** 4
        )

        # Upper surface: trailing edge to leading edge
        x_upper = self.le_x + self.chord * (1 - t)
        y_upper = self.cy + yt * self.chord

        # Lower surface: leading edge to trailing edge
        x_lower = self.le_x + self.chord * t
        y_lower = self.cy - yt * self.chord

        cp = torch.stack([
            torch.cat([x_upper, x_lower]),
            torch.cat([y_upper, y_lower]),
        ], dim=1)

        return cp

    def sdf(self, mesh: CartesianMesh, control_points: Tensor) -> Tensor:
        """Compute SDF from B-spline control points.

        Simplified: compute minimum distance from each cell center to the
        polygon defined by control points.

        Args:
            mesh: CartesianMesh instance.
            control_points: (2*n_cp, 2) tensor of [x, y] coordinates.

        Returns:
            SDF tensor (ny, nx).
        """
        x, y = mesh.cell_centers()
        nx, ny = mesh.nx, mesh.ny
        n_pts = control_points.shape[0]

        # Flatten grid coordinates
        pts = torch.stack([x.flatten(), y.flatten()], dim=1)  # (N, 2)

        # Compute distance from each grid point to each edge of the polygon
        min_dist_sq = torch.full((pts.shape[0],), float("inf"), device=mesh.device)

        for i in range(n_pts):
            j = (i + 1) % n_pts
            a = control_points[i]  # (2,)
            b = control_points[j]  # (2,)

            ab = b - a
            ap = pts - a
            t = torch.clamp(
                torch.sum(ap * ab, dim=1) / (torch.sum(ab * ab) + 1e-10),
                0.0, 1.0,
            )
            closest = a + t.unsqueeze(1) * ab
            dist_sq = torch.sum((pts - closest) ** 2, dim=1)
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

        dist = torch.sqrt(min_dist_sq)

        # Determine sign: positive outside, negative inside
        # Use winding number (simplified: ray casting)
        inside = torch.zeros(pts.shape[0], dtype=torch.bool, device=mesh.device)
        for i in range(n_pts):
            j = (i + 1) % n_pts
            yi = control_points[i, 1]
            yj = control_points[j, 1]
            xi = control_points[i, 0]
            xj = control_points[j, 0]

            cond = (yi > pts[:, 1]) != (yj > pts[:, 1])
            x_intersect = (pts[:, 1] - yi) / (yj - yi + 1e-10) * (xj - xi) + xi
            cross = cond & (pts[:, 0] < x_intersect)
            inside = inside ^ cross

        sdf = torch.where(inside, -dist, dist)
        return sdf.reshape(ny, nx)


def compute_forces(
    p: Tensor,
    ux: Tensor,
    uy: Tensor,
    mesh: CartesianMesh,
    sdf: Tensor,
    mu: float = 1.0,
    brinkman_eps: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Compute pressure drag and lift forces on an immersed body.

    Uses the Brinkman penalization mask to identify the body surface.

    Args:
        p: Pressure field (ny, nx).
        ux: x-velocity (ny, nx+1).
        uy: y-velocity (ny+1, nx).
        mesh: CartesianMesh instance.
        sdf: Signed distance field (ny, nx).
        mu: Dynamic viscosity.
        brinkman_eps: Brinkman coefficient used in the solve.

    Returns:
        (drag, lift): Pressure force components (scalar tensors, differentiable).
    """
    chi = mesh.sdf_to_mask(sdf, epsilon=brinkman_eps)
    body_mask = 1.0 - chi  # ~1 inside body, ~0 in fluid

    dx, dy = mesh.dx, mesh.dy

    # Pressure gradient across body surface
    dp_dx = (p[:, 1:] - p[:, :-1]) / dx  # (ny, nx-1)
    dp_dy = (p[1:, :] - p[:-1, :]) / dy  # (ny-1, nx)

    # Integrate pressure force over body surface
    # Use gradient of body_mask as surface normal indicator
    dmask_dx = (body_mask[:, 1:] - body_mask[:, :-1]) / dx  # (ny, nx-1)
    dmask_dy = (body_mask[1:, :] - body_mask[:-1, :]) / dy  # (ny-1, nx)

    # Pressure contribution (average to cell faces)
    p_face_x = 0.5 * (p[:, :-1] + p[:, 1:])  # (ny, nx-1)
    p_face_y = 0.5 * (p[:-1, :] + p[1:, :])  # (ny-1, nx)

    drag = (p_face_x * dmask_dx * dy).sum()
    lift = (p_face_y * dmask_dy * dx).sum()

    return drag, lift
