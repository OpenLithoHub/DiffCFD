"""Heat exchanger fin environment (Mode A) — single-step contextual bandit.

Action: fin shape parameters (height, spacing, angle).
Reward: Nusselt number / pressure drop (performance factor).
Single step: solve SIMPLE + energy, compute reward, done.

Demonstrates C1+C2 combination: implicit diff (C1) enables analytical
gradient for Mode A geometry optimization through the gymnasium interface.
"""

from __future__ import annotations


import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from diffcfd.envs.base import DiffCFDEnv
from diffcfd.geometry.shapes import rectangle_sdf
from diffcfd.solvers.heat_transfer import HeatTransfer2D
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D


class HeatExchangerEnv(DiffCFDEnv):
    """Heat exchanger fin geometry optimization (Mode A).

    Single-step contextual bandit: action defines fin shape parameters,
    reward = Nu / (1 + |ΔP|) where ΔP is pressure drop.

    Args:
        re: Reynolds number.
        grid: (nx, ny) grid resolution.
        lx, ly: Domain dimensions.
        device: PyTorch device.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        re: float = 100.0,
        grid: tuple[int, int] = (48, 24),
        lx: float = 2.0,
        ly: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(solver=None, mode="A")

        self.re = re
        self.nx, self.ny = grid
        self.lx, self.ly = lx, ly
        self.device = device

        # Reuse solver and mesh across steps
        self._solver = NavierStokes2D(
            reynolds_number=re,
            grid=grid,
            lx=lx,
            ly=ly,
            device=device,
            backward="implicit_diff",
            max_iter=2000,
            tol=1e-5,
        )
        self._mesh = self._solver.mesh
        self._ht = HeatTransfer2D(self._mesh, alpha=1.0 / (re * 0.71))

        # Action: [fin_height_1, fin_height_2, fin_spacing] ∈ (0, 1)
        self.action_space = gym.spaces.Box(
            low=0.05, high=0.95, shape=(3,), dtype=np.float32
        )

        # Observation: scalar performance metric
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        self._step_count = 0

    def _build_fins_sdf(self, action: Tensor) -> Tensor:
        """Build SDF for two rectangular fins from action parameters.

        All operations preserve autograd graph so gradients flow
        through action → geometry → SDF → Brinkman → SIMPLE → reward.
        """
        h1 = action[0] * self.ly * 0.4  # fin 1 height
        h2 = action[1] * self.ly * 0.4  # fin 2 height
        spacing = action[2] * 0.3 + 0.2  # spacing between fins

        fin_width = 0.03 * self.lx
        x1 = self.lx * 0.35
        x2 = x1 + spacing

        sdf1 = rectangle_sdf(self._mesh, x1, 0.0, x1 + fin_width, h1)
        sdf2 = rectangle_sdf(self._mesh, x2, 0.0, x2 + fin_width, h2)

        return torch.maximum(sdf1, sdf2)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        return np.array([0.0], dtype=np.float32), {}

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

        # Build geometry from action (differentiable)
        sdf = self._build_fins_sdf(action)

        # Solve NS flow around fins with implicit diff
        u_inlet = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        ux, uy, p = self._solver.solve_steady(
            sdf=sdf, inlet_velocity=u_inlet, case="channel"
        )

        # Solve energy equation (differentiable path)
        T_bc = {
            "bottom": ("dirichlet", 1.0),
            "top": ("dirichlet", 0.0),
            "left": ("dirichlet", 0.5),
            "right": ("neumann", 0.0),
        }
        T = self._ht.solve_differentiable(ux, uy, T_bc=T_bc, max_iter=300)

        # Reward: Nu / (1 + |ΔP|)
        Nu = self._ht.nusselt_number(T, T_hot=1.0, T_cold=0.0, L=self.ly, wall="bottom")
        dp = self._solver.pressure_drop(ux, uy, p)
        reward = Nu / (1.0 + dp.abs())

        obs = torch.tensor([reward.item()], device=self.device)
        done = True  # Mode A: single step
        info = {"Nu": Nu.item(), "dp": dp.item(), "reward": reward.item()}

        return obs, reward, done, info
