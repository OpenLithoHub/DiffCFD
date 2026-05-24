"""Tests for PCHE optimization workflow (v0.6)."""

import pytest
import torch


def test_pche_import():
    from diffcfd.workflows.pche import optimize_pche
    assert optimize_pche is not None


def test_pche_channel_sdf():
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.workflows.pche import _pche_channel_sdf

    mesh = CartesianMesh(20, 16, lx=1.0, ly=0.5)
    centers = torch.tensor([0.125, 0.25, 0.375])
    sdf = _pche_channel_sdf(mesh, centers, 0.03)
    assert sdf.shape == (16, 20)
    # Should have some negative values (inside channels)
    assert (sdf < 0).any()
    assert (sdf > 0).any()


def test_pche_optimize_runs():
    """PCHE optimization should run without errors on tiny grid."""
    from diffcfd.workflows.pche import optimize_pche

    result = optimize_pche(
        n_channels=2,
        grid=(16, 12),
        lx=0.5, ly=0.3,
        channel_radius=0.02,
        re=100.0,
        n_steps=2,
        lr=0.01,
        verbose=False,
    )
    assert "history" in result
    assert len(result["history"]["nu"]) == 2
    assert result["y_positions"].shape == (2,)
