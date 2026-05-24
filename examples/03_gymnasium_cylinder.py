"""Example: CylinderWakeEnv gymnasium environment.

Demonstrates the C2 patent claim:
    standard gymnasium.Env with gradient-attached step()
    → policy_gradient() returns exact analytical gradients via implicit diff

The environment supports two modes:
    Mode A: single-step contextual bandit (geometry optimization)
    Mode B: sequential quasi-steady-state episode (flow control)
"""

import torch
from diffcfd import CylinderWakeEnv

env = CylinderWakeEnv(re=100, grid=(48, 24), max_steps=5, mode="B")

# Standard gymnasium reset
obs, info = env.reset()
print(f"Initial obs shape: {obs.shape}")
print(f"Initial obs: {obs[:5].tolist()}")

# Standard gymnasium step (returns numpy for SB3 compatibility)
obs, reward, done, truncated, info = env.step([0.5])
print(f"\nAfter step [0.5]:")
print(f"  reward={reward:.4f}, done={done}, info={info}")

# Differentiable step (preserves autograd graph — C2 claim)
action = torch.tensor([1.0], requires_grad=True)
obs_diff, reward_diff, done_diff, info_diff = env.step_differentiable(action)

# Compute analytical gradient of reward w.r.t. action
reward_diff.backward()
print(f"\nDifferentiable step:")
print(f"  reward={reward_diff.item():.4f}")
print(f"  dR/d(action)={action.grad.item():.6f}")

# Or use policy_gradient() directly
action2 = torch.tensor([-0.5], requires_grad=True)
grad = env.policy_gradient(action2)
print(f"\npolicy_gradient([-0.5]) = {grad.item():.6f}")
