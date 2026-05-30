"""Differentiable Photolithography Solver (Exposure and Development).

Implements Dill's exposure model and Mack's development model to calculate
the final developed photoresist profile, taking spin-coating outputs as inputs.

Dill exposure: Beer-Lambert attenuation with PAC bleaching kinetics.
Mack development: Dissolution rate R(z) depends on PAC concentration M(z)
and residual solvent C (plasticizing effect).

Deduplication note (WS-B)
-------------------------
This ``LithoSolver`` is the canonical Dill/Mack physics implementation.
OpenLithoHub wraps it via ``DiffCFDLithoSimulator``
(``openlithohub/plugins/diffcfd_process.py``) and should not reimplement
the exposure/development physics.  The plugin adapter translates
OpenLithoHub's ``SimulatorConfig`` into the parameter surface below and
handles thickness/solvent tensor construction.  Any new Dill/Mack features
should be added here in DiffCFD, then exposed through the plugin adapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LithoSolver(nn.Module):
    """Differentiable Exposure and Development Solver.

    1. Exposure (Dill's Model):
       Calculates the remaining Photoactive Compound (PAC) concentration M(z)
       across the film thickness.

    2. Development (Mack's Model):
       Calculates local dissolution rate R(z) and developed depth as a function
       of time, modified by residual solvent concentration C (plasticizing effect).
    """

    def __init__(
        self,
        dill_A: float = 0.55,
        dill_B: float = 0.05,
        dill_C: float = 0.014,
        r_max: float = 150.0,
        r_min: float = 0.1,
        mack_n: float = 5.0,
        mack_a: float = 0.5,
        gamma_solvent: float = 3.0,
    ) -> None:
        super().__init__()
        self.dill_A = dill_A
        self.dill_B = dill_B
        self.dill_C = dill_C
        self.r_max = r_max
        self.r_min = r_min
        self.mack_n = mack_n
        self.mack_a = mack_a
        self.gamma_solvent = gamma_solvent

    def forward(
        self,
        thickness: Tensor,
        residual_solvent: Tensor,
        exposure_dose: Tensor,
        dev_time: float = 30.0,
        nz: int = 50,
    ) -> Tensor:
        """Run differentiable exposure and development simulation.

        Args:
            thickness: Dry film thickness h (m) from spin coating.
            residual_solvent: Residual solvent fraction C from spin coating.
            exposure_dose: Exposure dose D (mJ/cm2).
            dev_time: Development time in seconds.
            nz: Z-discretization layers.

        Returns:
            remaining_thickness: Developed resist thickness (m).
        """
        device = thickness.device

        z = torch.linspace(0.0, 1.0, nz, device=device) * thickness

        z_um = z * 1e6

        alpha_eff = self.dill_A + self.dill_B
        intensity_ratio = torch.exp(-alpha_eff * z_um)

        local_dose = exposure_dose * intensity_ratio

        M = torch.exp(-self.dill_C * local_dose)

        r_max_eff = self.r_max * torch.exp(self.gamma_solvent * residual_solvent)

        m_term = torch.pow(1.0 - M, self.mack_n)
        R = r_max_eff * ((self.mack_a + 1.0) * m_term) / (self.mack_a + m_term) + self.r_min

        R_m_s = R * 1e-9

        dz = thickness / nz
        time_to_dissolve = dz / R_m_s
        cumulative_time = torch.cumsum(time_to_dissolve, dim=0)

        tau = torch.clamp(time_to_dissolve.mean(), min=0.01)
        dissolved_mask = torch.sigmoid((dev_time - cumulative_time) / tau)
        developed_thickness = torch.sum(dissolved_mask * dz)

        remaining_thickness = torch.clamp(thickness - developed_thickness, min=0.0)
        return remaining_thickness
