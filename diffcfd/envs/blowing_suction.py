"""Blowing/suction boundary control environment (Mode B).

Active flow control via blowing/suction slots on a cylinder surface.
Each slot applies a normal velocity BC on the cylinder surface modeled
through Brinkman penalization with a body velocity field that varies
spatially over the cylinder.

Action: blowing/suction velocities at N_slots boundary slots.
Observation: velocity field at probe locations downstream of cylinder.
Reward: negative drag coefficient (minimize drag = maximize reward).

Benchmark: AD-gradient optimization vs PPO (RL) on drag reduction task.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from diffcfd.envs.base import DiffCFDEnv
from diffcfd.geometry.shapes import cylinder_sdf
from diffcfd.solvers.navier_stokes_2d import NavierStokes2D


class BlowingSuctionEnv(DiffCFDEnv):
    """Active flow control via blowing/suction boundary conditions.

    Mode B (sequential): control blowing/suction at slots on cylinder surface
    to reduce drag. Compatible with both AD-gradient optimization and PPO (RL).

    The blowing/suction is modeled by constructing a body velocity field on
    the cylinder surface. Each slot occupies an angular sector of the cylinder.
    The body velocity at each cell within a slot is the slot's action value
    projected onto the outward normal direction.

    Action space: blowing/suction velocities at n_slots boundary slots.
    Observation: velocity field at n_probes probe locations downstream.
    Reward: negative drag coefficient (minimize drag = maximize reward).

    Args:
        grid_size: Number of cells in x and y (square domain).
        Re: Reynolds number.
        n_slots: Number of blowing/suction slots on cylinder surface.
        max_blowing: Maximum blowing/suction velocity magnitude.
        n_probes: Number of downstream velocity probes.
        max_steps: Maximum episode length.
        lx: Domain length in x.
        ly: Domain length in y.
        cylinder_radius: Cylinder radius.
        cylinder_center: (cx, cy) cylinder center.
        inlet_velocity: Free-stream inlet velocity.
        device: PyTorch device.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 64,
        Re: float = 100.0,
        n_slots: int = 4,
        max_blowing: float = 0.5,
        n_probes: int = 8,
        max_steps: int = 20,
        lx: float = 2.5,
        ly: float = 1.0,
        cylinder_radius: float = 0.05,
        cylinder_center: tuple[float, float] = (0.5, 0.5),
        inlet_velocity: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(solver=None, mode="B")

        self.re = Re
        self.nx = grid_size
        self.ny = grid_size
        self.n_slots = n_slots
        self.max_blowing = max_blowing
        self.n_probes = n_probes
        self.max_steps = max_steps
        self.lx, self.ly = lx, ly
        self.cyl_r = cylinder_radius
        self.cyl_cx, self.cyl_cy = cylinder_center
        self.inlet_velocity = inlet_velocity
        self.device = device

        # Action: blowing/suction velocity at each slot in [-max_blowing, max_blowing]
        self.action_space = gym.spaces.Box(
            low=-max_blowing, high=max_blowing, shape=(n_slots,), dtype=np.float32
        )

        # Observation: velocity magnitude at downstream probes
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_probes,), dtype=np.float32
        )

        # Probe positions downstream of cylinder
        dx_grid = lx / self.nx
        n_half = n_probes // 2
        self.probe_x = torch.tensor(
            [self.cyl_cx + (i + 2) * dx_grid * 4 for i in range(n_half)],
            device=device,
        )
        self.probe_y = torch.tensor(
            [self.cyl_cy + (0.03 + 0.03 * i) * ((-1) ** i) for i in range(n_half)],
            device=device,
        )

        # Slot angular positions: evenly spaced around cylinder
        # Slots centered on the rear half (wake side) for drag reduction
        slot_start = -np.pi / 2
        slot_end = np.pi / 2
        self.slot_angles = torch.linspace(
            slot_start, slot_end, n_slots + 1, device=device
        )
        self.slot_centers = (
            self.slot_angles[:-1] + self.slot_angles[1:]
        ) / 2

        # Build solver
        self._solver = NavierStokes2D(
            reynolds_number=Re,
            grid=(self.nx, self.ny),
            lx=lx,
            ly=ly,
            device=device,
            backward="implicit_diff",
            max_iter=2000,
            tol=1e-5,
            alpha_u=0.5,
            alpha_p=0.1,
        )

        self.mesh = self._solver.mesh
        self._sdf = cylinder_sdf(self.mesh, self.cyl_cx, self.cyl_cy, self.cyl_r)
        self._chi = self.mesh.sdf_to_mask(self._sdf, epsilon=1e-3)

        self._step_count = 0
        self._ux: Tensor | None = None
        self._uy: Tensor | None = None
        self._p: Tensor | None = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0

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

    def step_differentiable(self, action: Tensor) -> tuple[Tensor, Tensor, bool, dict]:
        """Differentiable step: action (blowing/suction) -> reward, preserving grad_fn.

        Maps slot velocities to a body velocity field on the cylinder surface,
        then solves NS to steady state and computes drag reward.
        """
        self._step_count += 1

        u_inlet = torch.tensor(
            self.inlet_velocity, dtype=torch.float32, device=self.device
        ).requires_grad_(True)

        # Build body velocity field from slot actions
        u_body_x, u_body_y = self._slot_actions_to_body_velocity(action)

        self._ux, self._uy, self._p = self._solver.solve_steady(
            sdf=self._sdf,
            inlet_velocity=u_inlet,
            case="channel",
            u_body_x=u_body_x,
            u_body_y=u_body_y,
        )

        obs = self._build_obs()
        reward = self._compute_reward()
        done = self._step_count >= self.max_steps

        return obs, reward, done, self._build_info()

    def _slot_actions_to_body_velocity(
        self, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Convert slot blowing/suction velocities to body velocity fields.

        For each cell inside the cylinder, find which angular sector it belongs
        to, then apply the corresponding slot velocity as outward normal
        velocity on the surface. Interior cells away from the surface get zero
        body velocity (standard Brinkman no-slip).
        """
        x, y = self.mesh.cell_centers()
        dx = self.mesh.dx
        nx, ny = self.nx, self.ny

        # Relative position from cylinder center
        rx = x - self.cyl_cx
        ry = y - self.cyl_cy
        r = torch.sqrt(rx**2 + ry**2)

        # Angle of each cell relative to cylinder center
        angles = torch.atan2(ry, rx)

        # Surface shell: cells within one cell width of the surface
        surface_band = (r < self.cyl_r + dx) & (r > self.cyl_r - dx * 0.5)

        # Normal direction at each cell (outward from cylinder center)
        nx_dir = rx / (r + 1e-10)
        ny_dir = ry / (r + 1e-10)

        # Assign slot velocities based on angular position
        # Each slot occupies an angular sector
        u_body_x = torch.zeros(ny, nx, device=self.device)
        u_body_y = torch.zeros(ny, nx, device=self.device)

        for i in range(self.n_slots):
            a_lo = self.slot_angles[i]
            a_hi = self.slot_angles[i + 1]

            # Cells in this angular sector and on the surface band
            in_slot = (angles >= a_lo) & (angles < a_hi) & surface_band

            slot_vel = action[i]

            u_body_x = torch.where(
                in_slot,
                slot_vel * nx_dir,
                u_body_x,
            )
            u_body_y = torch.where(
                in_slot,
                slot_vel * ny_dir,
                u_body_y,
            )

        return u_body_x, u_body_y

    def _build_obs(self) -> Tensor:
        """Sample velocity at downstream probe locations."""
        dx, dy = self.mesh.dx, self.mesh.dy
        probes = []
        n_half = self.n_probes // 2

        for i in range(n_half):
            ix = min(max(round(self.probe_x[i].item() / dx), 0), self.nx - 1)
            iy = min(max(round(self.probe_y[i].item() / dy), 0), self.ny - 1)
            ux_val = 0.5 * (self._ux[iy, ix] + self._ux[iy, ix + 1])
            uy_val = 0.5 * (self._uy[iy, ix] + self._uy[iy + 1, ix])
            probes.extend([ux_val, uy_val])

        # Pad with zeros if n_probes is odd
        if self.n_probes % 2 == 1:
            probes.append(torch.tensor(0.0, device=self.device))

        return torch.stack(probes)

    def _compute_reward(self) -> Tensor:
        """Reward = negative drag coefficient (minimize drag).

        Integrates pressure and viscous forces over the cylinder surface
        using the Brinkman mask gradient as a surface normal proxy.
        """
        dx, dy = self.mesh.dx, self.mesh.dy
        chi = self._chi

        # Pressure drag: integrate p * d(chi)/dx over domain
        dchi_dx = (chi[:, 1:] - chi[:, :-1]) / dx
        p_face_x = 0.5 * (self._p[:, :-1] + self._p[:, 1:])
        pressure_drag = (p_face_x * dchi_dx * dy).sum()

        # Viscous drag: du_x/dy at y-faces times mask gradient
        dux_dy = (self._ux[1:, :] - self._ux[:-1, :]) / dy
        dchi_dy = (chi[1:, :] - chi[:-1, :]) / dy
        dux_dy_cc = 0.5 * (dux_dy[:, :-1] + dux_dy[:, 1:])
        viscous_drag = (1.0 / self.re) * (dux_dy_cc * dchi_dy * dx).sum()

        drag = pressure_drag + viscous_drag
        return -drag

    def _build_info(self) -> dict:
        return {
            "step": self._step_count,
            "reynolds": self.re,
            "n_slots": self.n_slots,
        }


class ADGradientOptimizer:
    """Direct AD-gradient optimization of blowing/suction control.

    Uses implicit differentiation through the NS solver to compute
    d(reward)/d(action), then applies gradient ascent (maximize reward =
    minimize drag).
    """

    def __init__(self, env: BlowingSuctionEnv, lr: float = 0.01) -> None:
        self.env = env
        self.lr = lr

    def optimize(self, n_steps: int = 100) -> dict:
        """Run AD gradient optimization.

        Each step:
        1. Reset env.
        2. Create differentiable action tensor.
        3. Step through env, collect reward.
        4. Backprop reward to get d(reward)/d(action).
        5. Gradient ascent update on action.

        Returns dict with:
            rewards: list of cumulative rewards per episode.
            actions: list of action arrays per step.
            gradient_norms: list of gradient norms per step.
        """
        rewards = []
        actions = []
        gradient_norms = []

        # Initialize action (blowing/suction velocity at each slot)
        action = torch.zeros(
            self.env.n_slots, device=self.env.device, requires_grad=True
        )

        for step in range(n_steps):
            self.env.reset()

            # Re-create action with grad tracking for this step
            action_param = action.detach().clone().requires_grad_(True)

            # Single differentiable step
            _, reward, done, info = self.env.step_differentiable(action_param)

            # Backward: compute d(reward)/d(action)
            reward.backward()

            grad = action_param.grad
            grad_norm = grad.norm().item()

            # Gradient ascent (reward = negative drag, maximize it)
            with torch.no_grad():
                action = action_param + self.lr * grad
                # Clamp to action bounds
                action = action.clamp(
                    -self.env.max_blowing, self.env.max_blowing
                )

            rewards.append(reward.item())
            actions.append(action.detach().cpu().numpy().copy())
            gradient_norms.append(grad_norm)

        return {
            "rewards": rewards,
            "actions": actions,
            "gradient_norms": gradient_norms,
        }


def create_comparison_report(
    n_steps: int = 50,
    n_seeds: int = 3,
    grid_size: int = 32,
) -> dict:
    """Compare AD-gradient vs random policy on blowing/suction control.

    The AD-gradient optimizer uses implicit differentiation through the NS
    solver for exact analytical gradients. The "RL baseline" is a random
    policy; with stable-baselines3 installed, this would be replaced by PPO.

    Returns dict with:
        ad_rewards: list of reward trajectories (one per seed).
        random_rewards: list of random policy trajectories (one per seed).
        sample_efficiency: dict with AD and random steps to reach target reward.
        final_reward_comparison: dict with mean final reward for AD and random.
    """
    ad_all_rewards = []
    random_all_rewards = []

    for seed in range(n_seeds):
        # --- AD-gradient optimizer ---
        env_ad = BlowingSuctionEnv(
            grid_size=grid_size,
            Re=100.0,
            n_slots=4,
            max_blowing=0.5,
            n_probes=8,
            max_steps=1,
        )
        optimizer = ADGradientOptimizer(env_ad, lr=0.01)
        result = optimizer.optimize(n_steps=n_steps)
        ad_all_rewards.append(result["rewards"])
        env_ad.close()

        # --- Random policy baseline ---
        # TODO: Replace with PPO when stable-baselines3 is available.
        # With sb3: model = PPO("MlpPolicy", env, verbose=0); model.learn(n_steps)
        env_rand = BlowingSuctionEnv(
            grid_size=grid_size,
            Re=100.0,
            n_slots=4,
            max_blowing=0.5,
            n_probes=8,
            max_steps=1,
        )
        rng = np.random.default_rng(seed=seed + 100)
        rand_rewards = []
        for _ in range(n_steps):
            env_rand.reset()
            action = rng.uniform(
                -env_rand.max_blowing, env_rand.max_blowing, size=(env_rand.n_slots,)
            )
            _, reward, _, _, _ = env_rand.step(action)
            rand_rewards.append(reward)
        random_all_rewards.append(rand_rewards)
        env_rand.close()

    ad_arr = np.array(ad_all_rewards)
    rand_arr = np.array(random_all_rewards)

    # Sample efficiency: steps to reach 50th percentile of AD final reward
    ad_mean_final = ad_arr[:, -1].mean()
    target_reward = ad_mean_final * 0.5

    def steps_to_target(trajectories: np.ndarray, target: float) -> float:
        steps_list = []
        for traj in trajectories:
            reached = np.where(np.array(traj) >= target)[0]
            steps_list.append(reached[0] + 1 if len(reached) > 0 else len(traj))
        return float(np.mean(steps_list))

    ad_steps = steps_to_target(ad_arr, target_reward)
    rand_steps = steps_to_target(rand_arr, target_reward)

    return {
        "ad_rewards": ad_all_rewards,
        "random_rewards": random_all_rewards,
        "sample_efficiency": {
            "ad_steps_to_target": ad_steps,
            "random_steps_to_target": rand_steps,
            "target_reward": target_reward,
        },
        "final_reward_comparison": {
            "ad_mean_final": float(ad_arr[:, -1].mean()),
            "ad_std_final": float(ad_arr[:, -1].std()),
            "random_mean_final": float(rand_arr[:, -1].mean()),
            "random_std_final": float(rand_arr[:, -1].std()),
        },
    }
