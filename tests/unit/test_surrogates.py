"""Tests for FNO surrogate (v0.5)."""

import torch


def test_fno_import():
    from diffcfd.surrogates.fno import FNO2D

    assert FNO2D is not None


def test_fno_forward_shape():
    from diffcfd.surrogates.fno import FNO2D

    model = FNO2D(modes=4, width=16, depth=2)
    x = torch.randn(2, 3, 16, 32)  # batch=2, 3 channels, 16x32 grid
    y = model(x)
    assert y.shape == (2, 3, 16, 32)


def test_fno_differentiable():
    from diffcfd.surrogates.fno import FNO2D

    model = FNO2D(modes=4, width=16, depth=2)
    x = torch.randn(1, 3, 8, 16, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fno_overfit_small():
    """FNO should overfit a tiny dataset quickly."""
    from diffcfd.surrogates.fno import FNO2D

    model = FNO2D(modes=4, width=32, depth=2)
    x = torch.randn(2, 3, 8, 16)
    y_target = torch.randn(2, 3, 8, 16)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - y_target) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_loss = ((model(x) - y_target) ** 2).mean().item()
    assert final_loss < 0.5, f"FNO should overfit 2 samples, got loss={final_loss:.4f}"


def test_spectral_conv():
    from diffcfd.surrogates.fno import _SpectralConv2d

    layer = _SpectralConv2d(8, 8, modes=4)
    x = torch.randn(2, 8, 16, 16)
    y = layer(x)
    assert y.shape == x.shape
