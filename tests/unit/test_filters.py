"""Unit tests for Helmholtz filter and topology optimization."""

import torch
import pytest


def test_helmholtz_filter_import():
    from diffcfd.geometry.filters import HelmholtzFilter

    assert HelmholtzFilter is not None


def test_helmholtz_filter_uniform():
    """Uniform input should produce uniform output."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.filters import HelmholtzFilter

    mesh = CartesianMesh(16, 16)
    filt = HelmholtzFilter(mesh, radius=0.05)

    rho = torch.ones(16, 16) * 0.5
    rho_f = filt.apply(rho)
    assert rho_f.max().item() == pytest.approx(0.5, abs=1e-6)
    assert rho_f.min().item() == pytest.approx(0.5, abs=1e-6)


def test_helmholtz_filter_smooths():
    """Point source should be spread over multiple cells."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.filters import HelmholtzFilter

    mesh = CartesianMesh(20, 20, lx=1.0, ly=1.0)
    filt = HelmholtzFilter(mesh, radius=0.1)

    rho = torch.zeros(20, 20)
    rho[10, 10] = 1.0
    rho_f = filt.apply(rho)

    # Should be spread over more than 1 cell
    assert (rho_f > 0.01).sum().item() > 1
    # Peak should be reduced
    assert rho_f.max().item() < 1.0
    # Mass should be approximately conserved
    assert rho_f.sum().item() == pytest.approx(1.0, abs=0.1)


def test_helmholtz_filter_differentiable():
    """Gradient flows through the differentiable filter path."""
    from diffcfd.geometry.mesh import CartesianMesh
    from diffcfd.geometry.filters import HelmholtzFilter

    mesh = CartesianMesh(10, 10)
    filt = HelmholtzFilter(mesh, radius=0.05)

    rho = torch.rand(10, 10, requires_grad=True)
    rho_f = filt.apply_differentiable(rho, n_iter=50)
    loss = rho_f.sum()
    loss.backward()

    assert rho.grad is not None
    assert rho.grad.norm().item() > 0


def test_smooth_heaviside():
    """Smooth Heaviside projection maps [0,1] to [0,1]."""
    from diffcfd.workflows.topology import smooth_heaviside

    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    y = smooth_heaviside(x, beta=16.0)
    assert y.min() >= 0.0
    assert y.max() <= 1.0
    # At x=0.5, should be ~0.5
    assert y[2].item() == pytest.approx(0.5, abs=0.01)
    # Below 0.5 should be close to 0
    assert y[1].item() < 0.1
    # Above 0.5 should be close to 1
    assert y[3].item() > 0.9
