"""FNO with learned geometry deformation for irregular domains.

Wraps the standard FNO2D with a learned deformation map that transforms
an irregular domain (encoded via SDF) to a regular grid. The deformation
is a small CNN that predicts (dx, dy) offsets for each grid point.

Architecture:
  1. Deformation network: SDF -> (dx, dy) grid offsets
  2. Apply forward deformation to input
  3. Run standard FNO on deformed input
  4. Apply inverse deformation to output

This approach leverages the existing FNO implementation while handling
non-rectangular geometries through learned spatial transformations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .fno import FNO2D


class _DeformationNet(nn.Module):
    """Predict per-pixel (dx, dy) deformation offsets from SDF field.

    Args:
        hidden: Hidden channel width.
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )

    def forward(self, sdf: Tensor) -> Tensor:
        """Predict deformation offsets.

        Args:
            sdf: (B, 1, H, W) signed distance field.

        Returns:
            (B, 2, H, W) deformation offsets (dx, dy).
        """
        return self.encoder(sdf)


class GeoFNO(nn.Module):
    """FNO with learned geometry deformation for irregular domains.

    Wraps an existing FNO2D model with a learned spatial deformation.
    When an SDF field is provided, a small CNN predicts per-pixel offsets
    that warp the input toward a regular grid, the FNO processes the
    warped input, and the output is warped back.

    When no SDF is provided, behaves identically to the underlying FNO.

    Args:
        fno: Underlying FNO2D model to wrap.
        deform_hidden: Hidden channel width for the deformation network.
    """

    def __init__(self, fno: FNO2D, deform_hidden: int = 16) -> None:
        super().__init__()
        self.fno = fno
        self.deform_net = _DeformationNet(hidden=deform_hidden)

    def _deform_input(self, x: Tensor, sdf: Tensor) -> Tensor:
        """Apply forward spatial deformation to input using bilinear sampling.

        Args:
            x: (B, C, H, W) input tensor.
            sdf: (B, 1, H, W) signed distance field.

        Returns:
            (B, C, H, W) deformed input.
        """
        offsets = self.deform_net(sdf)  # (B, 2, H, W)
        return self._apply_grid_sample(x, offsets)

    def _deform_output(self, out: Tensor, sdf: Tensor) -> Tensor:
        """Apply inverse deformation to output.

        Uses the same deformation network but negates the offsets
        to approximately invert the spatial transformation.

        Args:
            out: (B, C, H, W) FNO output.
            sdf: (B, 1, H, W) signed distance field.

        Returns:
            (B, C, H, W) deformed output.
        """
        offsets = self.deform_net(sdf)  # (B, 2, H, W)
        return self._apply_grid_sample(out, -offsets)

    @staticmethod
    def _apply_grid_sample(x: Tensor, offsets: Tensor) -> Tensor:
        """Apply grid_sample with predicted offsets.

        Args:
            x: (B, C, H, W) input tensor.
            offsets: (B, 2, H, W) dx, dy offsets in [-1, 1] normalized coords.

        Returns:
            (B, C, H, W) resampled tensor.
        """
        B, _, H, W = x.shape

        # Create base grid in [-1, 1]
        gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        gx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        # Base grid: (1, H, W, 2)
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Scale offsets: small perturbation in normalized coordinates
        # Use tanh to bound offsets to a reasonable range
        scaled_offsets = torch.tanh(offsets) * 0.1  # max 10% grid deformation

        # (B, 2, H, W) -> (B, H, W, 2)
        scaled_offsets = scaled_offsets.permute(0, 2, 3, 1)
        grid = base_grid.expand(B, -1, -1, -1) + scaled_offsets

        return torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def forward(self, x: Tensor, sdf: Tensor | None = None) -> Tensor:
        """Predict flow fields with optional geometry deformation.

        Args:
            x: Input tensor (B, in_channels, H, W).
            sdf: Optional SDF field (B, 1, H, W) for geometry-aware deformation.

        Returns:
            Predicted (ux, uy, p) stacked as (B, 3, H, W).
        """
        if sdf is not None:
            x = self._deform_input(x, sdf)

        out = self.fno(x)

        if sdf is not None:
            out = self._deform_output(out, sdf)

        return out
