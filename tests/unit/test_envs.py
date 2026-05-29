"""Tests for gymnasium environments."""

import numpy as np
import pytest


def test_cylinder_env_import():
    from diffcfd.envs.cylinder_wake import CylinderWakeEnv

    assert CylinderWakeEnv is not None


def test_cylinder_env_spaces():
    from diffcfd.envs.cylinder_wake import CylinderWakeEnv

    env = CylinderWakeEnv(
        re=100.0,
        grid=(32, 16),
        max_steps=2,
        lx=2.5,
        ly=1.0,
        cylinder_radius=0.05,
        cylinder_center=(0.5, 0.5),
    )
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (10,)
    env.close()


def test_heat_exchanger_env_spaces():
    from diffcfd.envs.heat_exchanger import HeatExchangerEnv

    env = HeatExchangerEnv(re=50.0, grid=(24, 12))
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (1,)
    env.close()


def test_shapes_import():
    from diffcfd.geometry.shapes import cylinder_sdf, rectangle_sdf
    from diffcfd.geometry.mesh import CartesianMesh

    mesh = CartesianMesh(nx=32, ny=32)
    sdf = cylinder_sdf(mesh, 0.5, 0.5, 0.1)
    assert sdf.shape == (32, 32)
    assert sdf.min() < 0  # Inside cylinder
    assert sdf.max() > 0  # Outside cylinder

    sdf_rect = rectangle_sdf(mesh, 0.3, 0.3, 0.7, 0.7)
    assert sdf_rect.shape == (32, 32)


@pytest.mark.slow
def test_cylinder_env_step():
    from diffcfd.envs.cylinder_wake import CylinderWakeEnv

    env = CylinderWakeEnv(
        re=100.0,
        grid=(32, 16),
        max_steps=2,
        lx=2.5,
        ly=1.0,
        cylinder_radius=0.05,
        cylinder_center=(0.5, 0.5),
    )
    obs, info = env.reset()
    assert obs.shape == (10,)
    assert not np.any(np.isnan(obs))

    obs2, reward, done, truncated, info2 = env.step([0.5])
    assert isinstance(reward, float)
    assert not done  # step 1 of max_steps=2

    obs3, reward3, done3, truncated3, info3 = env.step([0.0])
    assert done3  # step 2 of max_steps=2
    env.close()


@pytest.mark.slow
def test_heat_exchanger_env_step():
    from diffcfd.envs.heat_exchanger import HeatExchangerEnv

    env = HeatExchangerEnv(re=50.0, grid=(24, 12))
    obs, info = env.reset()

    obs2, reward, done, truncated, info2 = env.step([0.5, 0.5, 0.5])
    assert done  # Mode A: single step
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert "Nu" in info2
    env.close()
