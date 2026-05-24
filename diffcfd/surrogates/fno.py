"""Fourier Neural Operator (FNO) surrogate for steady-state flow prediction.

Learns the mapping: (geometry_mask, BC_params) → (ux, uy, p) flow fields.

Architecture:
  1. Lift input channels (mask + BC) to hidden dimension
  2. N Fourier layers with spectral convolution + skip connection
  3. Project to output channels (ux, uy, p)

Each Fourier layer:
  - Spectral conv: multiply lowest k Fourier modes by learnable weights
  - Local conv: 1×1 conv for high-frequency features
  - Add + GeLU activation

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", ICLR 2021.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _SpectralConv2d(nn.Module):
    """2D Fourier layer: learns k×k spectral weights on lowest frequency modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        m = self.modes

        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply learnable spectral weights on low-frequency modes
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m_h = min(m, H)
        m_w = min(m, W // 2 + 1)
        out_ft[:, :, :m_h, :m_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m_h, :m_w],
            self.weights[:, :, :m_h, :m_w],
        )

        return torch.fft.irfft2(out_ft, s=(H, W))


class _FNOBlock(nn.Module):
    """Single FNO layer: spectral conv + local conv + activation."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = _SpectralConv2d(width, width, modes)
        self.local = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.spectral(x) + self.local(x))


class FNO2D(nn.Module):
    """Fourier Neural Operator for 2D steady-state flow prediction.

    Input channels:
      - Geometry mask (1 channel): Brinkman penalization mask χ ∈ [0, 1]
      - BC parameters (2 channels): inlet_velocity, encoded as constant fields

    Output channels (3):
      - ux: x-velocity at cell centers (ny, nx)
      - uy: y-velocity at cell centers (ny, nx)
      - p: pressure at cell centers (ny, nx)

    Args:
        modes: Number of Fourier modes to keep (spectral resolution).
        width: Hidden channel dimension.
        depth: Number of FNO blocks.
        in_channels: Number of input channels (default 3: mask + 2 BC fields).
        out_channels: Number of output channels (default 3: ux, uy, p).
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 64,
        depth: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.width = width

        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([_FNOBlock(width, modes) for _ in range(depth)])
        self.project = nn.Linear(width, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Predict flow fields from geometry + BC input.

        Args:
            x: Input tensor (B, in_channels, H, W).

        Returns:
            Predicted (ux, uy, p) stacked as (B, 3, H, W).
        """
        # Lift input channels
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.lift(x)            # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (B, width, H, W)

        # FNO blocks
        for block in self.blocks:
            x = block(x) + x  # residual connection

        # Project to output channels
        x = x.permute(0, 2, 3, 1)  # (B, H, W, width)
        x = self.project(x)         # (B, H, W, out_channels)
        x = x.permute(0, 3, 1, 2)  # (B, out_channels, H, W)

        return x


def generate_fno_training_data(
    n_samples: int = 200,
    nx: int = 32,
    ny: int = 16,
    lx: float = 2.0,
    ly: float = 1.0,
    re: float = 100.0,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, Tensor]:
    """Generate training data for FNO using DiffCFD ground-truth solves.

    Varies inlet velocity and cylinder position to create diverse training pairs.

    Returns:
        Dict with 'input' (B, 3, ny, nx) and 'output' (B, 3, ny, nx).
    """
    from diffcfd.geometry.shapes import cylinder_sdf
    from diffcfd.solvers.navier_stokes_2d import NavierStokes2D

    inputs = []
    outputs = []

    for i in range(n_samples):
        # Vary inlet velocity
        u_inlet = 0.5 + 1.5 * (i / max(n_samples - 1, 1))

        # Vary cylinder position slightly
        cx = 0.5 + 0.1 * torch.sin(torch.tensor(i * 0.3)).item()
        cy = ly / 2.0
        radius = 0.05

        solver = NavierStokes2D(
            reynolds_number=re,
            grid=(nx, ny),
            lx=lx, ly=ly,
            device=device,
            backward="unrolled",
            max_iter=1000,
            tol=1e-4,
        )
        sdf = cylinder_sdf(solver.mesh, cx, cy, radius)
        chi = solver.mesh.sdf_to_mask(sdf, epsilon=1e-3)

        ux, uy, p = solver.solve_steady(sdf=sdf, inlet_velocity=u_inlet, case="channel")

        # Cell-center velocities for FNO output
        ux_cc = 0.5 * (ux[:, :-1] + ux[:, 1:])
        uy_cc = 0.5 * (uy[:-1, :] + uy[1:, :])

        # Input: [mask, u_inlet_field, Re_field]
        u_field = torch.full((ny, nx), u_inlet)
        re_field = torch.full((ny, nx), re)
        inp = torch.stack([chi, u_field, re_field], dim=0)  # (3, ny, nx)

        # Output: [ux_cc, uy_cc, p]
        out = torch.stack([ux_cc, uy_cc, p], dim=0)  # (3, ny, nx)

        inputs.append(inp)
        outputs.append(out)

        if verbose and (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")

    return {
        "input": torch.stack(inputs),
        "output": torch.stack(outputs),
    }


def train_fno(
    n_train: int = 100,
    nx: int = 32,
    ny: int = 16,
    modes: int = 8,
    width: int = 32,
    depth: int = 3,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> FNO2D:
    """Train an FNO surrogate on DiffCFD-generated data.

    Returns:
        Trained FNO2D model.
    """
    data = generate_fno_training_data(
        n_samples=n_train, nx=nx, ny=ny, device=device, verbose=verbose,
    )
    x_train = data["input"].to(device)
    y_train = data["output"].to(device)

    model = FNO2D(modes=modes, width=width, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = ((pred - y_train) ** 2).mean()
        loss.backward()
        opt.step()
        scheduler.step()

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            rel_err = ((pred - y_train) ** 2).sum() / (y_train ** 2).sum()
            print(f"Epoch {epoch:4d}: loss={loss.item():.4e}, rel_err={rel_err.item():.4f}")

    return model
