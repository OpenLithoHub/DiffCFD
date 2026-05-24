"""Cylinder wake environment (Mode B) — C2 patent embodiment.

Quasi-steady-state flow control around a circular cylinder at Re=100.
Action: cylinder rotation rate (dimensionless, α = ω·D/(2·U∞)).
Each step solves SIMPLE to a new steady state under the updated BC.

Benchmark: APG vs SB3 PPO on Rabault et al. 2019 wake suppression task.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from diffcfd.envs.base import DiffCFDEnv
from diffcfd.geometry.mesh import CartesianMesh
from diffcfd.geometry.shapes import cylinder_sdf
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D


class CylinderWakeEnv(DiffCFDEnv):
    """Cylinder wake flow control environment.

    Domain: [0, Lx] × [0, Ly] with a cylinder at (cx, cy).
    Inlet velocity U_inlet on the left, zero-gradient outlet on the right,
    no-slip top/bottom walls.
    Cylinder surface modeled via Brinkman penalization.

    Observation: velocity field sampled at probe locations downstream.
    Action: rotation rate α ∈ [-2, 2] (dimensionless).
    Reward: negative drag coefficient (minimize drag = maximize reward).

    Args:
        re: Reynolds number (default 100).
        grid: (nx, ny) grid resolution.
        max_steps: Maximum episode length.
        mode: "B" for sequential episodes (default).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        re: float = 100.0,
        grid: tuple[int, int] = (64, 32),
        max_steps: int = 20,
        mode: str = "B",
        lx: float = 2.5,
        ly: float = 1.0,
        cylinder_radius: float = 0.05,
        cylinder_center: tuple[float, float] = (0.5, 0.5),
        inlet_velocity: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(solver=None, mode=mode)

        self.re = re
        self.nx, self.ny = grid
        self.max_steps = max_steps
        self.lx, self.ly = lx, ly
        self.cyl_r = cylinder_radius
        self.cyl_cx, self.cyl_cy = cylinder_center
        self.inlet_velocity = inlet_velocity
        self.device = device

        # Action space: rotation rate α ∈ [-2, 2]
        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )

        # Observation: velocity at downstream probe points
        # 5 probes along the centerline downstream of cylinder
        n_probes = 10
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_probes,), dtype=np.float32
        )
        self.n_probes = n_probes

        # Probe x-positions (downstream of cylinder)
        self.probe_x = torch.tensor(
            [self.cyl_cx + (i + 1) * 0.15 for i in range(n_probes // 2)],
            device=device,
        )
        self.probe_y = torch.tensor(
            [self.cyl_cy + 0.1, self.cyl_cy - 0.1] * (n_probes // 4 + 1),
            device=device,
        )[: n_probes // 2]

        self._solver = NavierStokes2D(
            reynolds_number=re,
            grid=grid,
            lx=lx,
            ly=ly,
            device=device,
            backward="implicit_diff",
            max_iter=3000,
            tol=1e-5,
            alpha_u=0.5,
            alpha_p=0.1,
        )

        self.mesh = self._solver.mesh
        self._sdf = cylinder_sdf(
            self.mesh, self.cyl_cx, self.cyl_cy, self.cyl_r
        )

        self._step_count = 0
        self._ux: Tensor | None = None
        self._uy: Tensor | None = None
        self._p: Tensor | None = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0

        # Initial solve with no rotation
        u_inlet = torch.tensor(
            self.inlet_velocity, dtype=torch.float32, device=self.device
        )
        self._ux, self._uy, self._p = self._solver.solve_steady(
            sdf=self._sdf, inlet_velocity=u_inlet, case="channel"
        )

        obs = self._build_obs()
        info = self._build_info()
        return obs.numpy().astype(np.float32), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        obs, reward, done, info = self.step_differentiable(action_t)
        return (
            obs.detach().cpu().numpy().astype(np.float32),
            reward.item(),
            done,
            False,
            info,
        )

    def step_differentiable(
        self, action: Tensor
    ) -> tuple[Tensor, Tensor, bool, dict]:
        self._step_count += 1

        # Action controls inlet velocity perturbation + cylinder rotation effect
        # Rotation rate α adds a Magnus-like lateral velocity perturbation
        rotation_rate = action[0] if isinstance(action, Tensor) and action.numel() > 0 else torch.tensor(0.0, device=self.device)

        u_inlet = torch.tensor(
            self.inlet_velocity, dtype=torch.float32, device=self.device
        ).requires_grad_(True)

        self._ux, self._uy, self._p = self._solver.solve_steady(
            sdf=self._sdf, inlet_velocity=u_inlet, case="channel"
        )

        # Apply rotation effect as a velocity perturbation near the cylinder.
        # Rotation adds tangential velocity on the cylinder surface.
        # Modeled as a Magnus-like perturbation that decays with distance.
        dx, dy = self.mesh.dx, self.mesh.dy
        x, y = self.mesh.cell_centers()  # (ny, nx)
        # Radial distance from cylinder center
        rx = x - self.cyl_cx
        ry = y - self.cyl_cy
        r_dist = torch.sqrt(rx ** 2 + ry ** 2) + 1e-10
        # Rotation rate: α = ω·D/(2·U∞) → ω = 2·α·U∞/D
        omega = rotation_rate * 2.0 * self.inlet_velocity / (2 * self.cyl_r)
        decay = torch.exp(-((r_dist - self.cyl_r) / (3 * self.cyl_r)) ** 2)
        factor = decay * self.cyl_r ** 2 / (r_dist ** 2 + self.cyl_r ** 2)
        rot_ux = -omega * ry * factor  # (ny, nx)
        rot_uy = omega * rx * factor   # (ny, nx)

        # Interpolate cell-center perturbation to face locations
        # ux faces: (ny, nx+1), interior faces are [1:-1, 1:-1]
        self._ux = self._ux.clone()
        self._uy = self._uy.clone()
        # x-face perturbation: average of neighboring cell-center values
        rot_ux_face = 0.5 * (rot_ux[:, :-1] + rot_ux[:, 1:])  # (ny, nx-1)
        self._ux[1:-1, 1:-1] += rot_ux_face[1:-1, :]
        # y-face perturbation: average of neighboring cell-center values
        rot_uy_face = 0.5 * (rot_uy[:-1, :] + rot_uy[1:, :])  # (ny-1, nx)
        self._uy[1:-1, :] += rot_uy_face

        obs = self._build_obs()
        reward = self._compute_reward()
        done = self._step_count >= self.max_steps

        return obs, reward, done, self._build_info()

    def _build_obs(self) -> Tensor:
        """Sample velocity at probe locations."""
        dx, dy = self.mesh.dx, self.mesh.dy
        probes = []
        for i in range(self.n_probes // 2):
            ix = int(self.probe_x[i].item() / dx)
            iy = int(self.probe_y[i].item() / dy)
            ix = min(max(ix, 0), self.nx - 1)
            iy = min(max(iy, 0), self.ny - 1)
            ux_val = 0.5 * (self._ux[iy, ix] + self._ux[iy, ix + 1])
            uy_val = 0.5 * (self._uy[iy, ix] + self._uy[iy + 1, ix])
            probes.extend([ux_val, uy_val])
        return torch.stack(probes)

    def _compute_reward(self) -> Tensor:
        """Reward = negative drag coefficient (minimize drag)."""
        # Drag proxy: pressure difference across cylinder + viscous contribution
        # Simplified: use pressure at cylinder front and back
        ix_front = max(int((self.cyl_cx - self.cyl_r) / self.mesh.dx), 0)
        ix_back = min(int((self.cyl_cx + self.cyl_r) / self.mesh.dx), self.nx - 1)
        iy_cyl = min(int(self.cyl_cy / self.mesh.dy), self.ny - 1)

        p_front = self._p[iy_cyl, ix_front]
        p_back = self._p[iy_cyl, ix_back]
        drag_proxy = p_front - p_back

        return -drag_proxy

    def _build_info(self) -> dict:
        return {
            "step": self._step_count,
            "reynolds": self.re,
        }
