"""Base gymnasium.Env for DiffCFD reinforcement learning environments (v0.3+).

C2 patent claim: step() preserves the autograd graph (no .detach() on outputs),
enabling policy_gradient() to return exact analytical gradients via implicit diff.

Mode A: single-step contextual bandit (geometry/BC optimization).
Mode B: sequential quasi-steady-state episode (flow control, compatible with SB3/PPO).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from torch import Tensor


class DiffCFDEnv(gym.Env):
    """Abstract base class for DiffCFD gymnasium environments.

    Subclasses must implement: _build_obs, _apply_action, _compute_reward,
    _is_terminal.

    Key invariant (C2): all tensors returned by step() retain grad_fn — no
    .detach() or .numpy() on quantities of interest.  The gymnasium contract
    requires numpy arrays for SB3/CleanRL compatibility; the implementation
    returns numpy arrays from step() for SB3 mode and raw tensors from
    step_differentiable() for APG mode.
    """

    def __init__(self, solver, mode: str = "B") -> None:
        super().__init__()
        self.solver = solver
        self.mode = mode  # "A" (contextual bandit) or "B" (sequential episode)
        self._state: Tensor | None = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        raise NotImplementedError

    def step(self, action) -> tuple[Any, float, bool, bool, dict]:
        """Standard gymnasium step — returns numpy arrays for SB3 compatibility."""
        raise NotImplementedError

    def step_differentiable(self, action: Tensor) -> tuple[Tensor, Tensor, bool, dict]:
        """Gradient-attached step — returns PyTorch tensors with grad_fn intact.

        This is the C2 claim: same step logic as step() but no .detach() on outputs.
        action must be a tensor with requires_grad=True for policy gradients to flow.

        Returns:
            obs: Observation tensor with grad_fn.
            reward: Scalar reward tensor with grad_fn.
            done: bool.
            info: dict.
        """
        raise NotImplementedError

    def policy_gradient(self, action: Tensor) -> Tensor:
        """Return dL/dθ (gradient of reward w.r.t. action) via implicit diff.

        Uses C1 (fixed-point implicit differentiation) internally.
        Equivalent to calling step_differentiable(action).reward.backward()
        but memory-efficient (O(N) via GMRES, not O(N·K) unrolled).

        Args:
            action: Policy output tensor (requires_grad=True).

        Returns:
            Gradient tensor, same shape as action.
        """
        raise NotImplementedError
