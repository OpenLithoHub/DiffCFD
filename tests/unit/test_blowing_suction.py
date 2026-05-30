"""Tests for blowing/suction boundary control environment and AD optimizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_env_import():
    from diffcfd.envs.blowing_suction import BlowingSuctionEnv

    assert BlowingSuctionEnv is not None


def test_env_spaces():
    from diffcfd.envs.blowing_suction import BlowingSuctionEnv

    env = BlowingSuctionEnv(
        grid_size=32,
        n_slots=4,
        n_probes=8,
        max_steps=2,
    )
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (8,)
    env.close()


@pytest.mark.slow
def test_env_step_shape():
    """Action and reward shapes are correct after step."""
    from diffcfd.envs.blowing_suction import BlowingSuctionEnv

    env = BlowingSuctionEnv(
        grid_size=32,
        n_slots=4,
        n_probes=8,
        max_steps=3,
    )
    obs, info = env.reset()
    assert obs.shape == (8,)
    assert not np.any(np.isnan(obs))

    action = np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
    obs2, reward, done, truncated, info2 = env.step(action)

    assert obs2.shape == (8,)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not done  # step 1 of max_steps=3
    assert "step" in info2
    assert info2["step"] == 1
    env.close()


@pytest.mark.slow
def test_env_differentiable():
    """Gradients flow through step_differentiable."""
    from diffcfd.envs.blowing_suction import BlowingSuctionEnv

    env = BlowingSuctionEnv(
        grid_size=32,
        n_slots=4,
        n_probes=8,
        max_steps=2,
    )
    env.reset()

    action = torch.tensor(
        [0.1, -0.1, 0.05, -0.05], device=env.device, requires_grad=True
    )
    obs, reward, done, info = env.step_differentiable(action)

    assert obs.shape == (8,)
    assert reward.shape == ()
    assert reward.requires_grad

    # Backward should succeed
    reward.backward()
    assert action.grad is not None
    assert action.grad.shape == (4,)
    assert torch.isfinite(action.grad).all()
    env.close()


@pytest.mark.slow
def test_ad_optimizer_runs():
    """AD optimizer completes without error and returns valid metrics."""
    from diffcfd.envs.blowing_suction import ADGradientOptimizer, BlowingSuctionEnv

    env = BlowingSuctionEnv(
        grid_size=32,
        n_slots=4,
        n_probes=8,
        max_steps=1,
    )
    optimizer = ADGradientOptimizer(env, lr=0.01)
    result = optimizer.optimize(n_steps=3)

    assert "rewards" in result
    assert "actions" in result
    assert "gradient_norms" in result
    assert len(result["rewards"]) == 3
    assert len(result["actions"]) == 3
    assert len(result["gradient_norms"]) == 3

    for r in result["rewards"]:
        assert np.isfinite(r)
    for g in result["gradient_norms"]:
        assert np.isfinite(g)
    env.close()


@pytest.mark.slow
def test_comparison_report():
    """Comparison report produces valid metrics."""
    from diffcfd.envs.blowing_suction import create_comparison_report

    report = create_comparison_report(
        n_steps=3,
        n_seeds=2,
        grid_size=32,
    )

    assert "ad_rewards" in report
    assert "random_rewards" in report
    assert "sample_efficiency" in report
    assert "final_reward_comparison" in report

    assert len(report["ad_rewards"]) == 2
    assert len(report["random_rewards"]) == 2
    assert len(report["ad_rewards"][0]) == 3
    assert len(report["random_rewards"][0]) == 3

    se = report["sample_efficiency"]
    assert "ad_steps_to_target" in se
    assert "random_steps_to_target" in se
    assert "target_reward" in se

    frc = report["final_reward_comparison"]
    assert "ad_mean_final" in frc
    assert "random_mean_final" in frc
    assert np.isfinite(frc["ad_mean_final"])
    assert np.isfinite(frc["random_mean_final"])


@pytest.mark.slow
def test_ad_better_than_random():
    """AD-gradient achieves higher cumulative reward than random policy."""
    from diffcfd.envs.blowing_suction import create_comparison_report

    report = create_comparison_report(
        n_steps=10,
        n_seeds=2,
        grid_size=32,
    )

    # AD-gradient should converge to higher (less negative) reward than random
    # Using cumulative mean over all steps
    ad_cumulative = np.mean([np.mean(traj) for traj in report["ad_rewards"]])
    rand_cumulative = np.mean([np.mean(traj) for traj in report["random_rewards"]])

    assert ad_cumulative >= rand_cumulative, (
        f"AD mean cumulative reward ({ad_cumulative:.4f}) should be >= "
        f"random ({rand_cumulative:.4f})"
    )
