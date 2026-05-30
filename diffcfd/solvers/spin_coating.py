"""Differentiable Spin Coating Solver for Photoresist Processing.

Implements two physical models of thin-film flow driven by centrifugal forces
and solvent evaporation, fully compatible with PyTorch autograd:

1. Meyerhofer ODE model (Spatially-uniform approximation)
2. 1D Axisymmetric Radial PDE model (For thickness uniformity & dynamic dispense)

Physical basis: Lubrication approximation (thin-film limit) where
dh/dt = -(2ρω²h³)/(3μ(C)) - E(ω)

With solvent-dependent viscosity μ(C) = μ_s·exp(α·((1-C)/C)^β)
and rotation-enhanced evaporation E = c_e·√ω.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

_MAX_SUBSTEPS = 8


class MeyerhoferSolver(nn.Module):
    """Spatially-uniform thin-film spin coating solver (Meyerhofer theory).

    Solves coupled temporal ODEs for wet film thickness h(t) and solvent mass
    fraction C(t) under centrifugal thinning and evaporation.

    Polymer mass per unit area M_p = ρ(1-C)h is conserved throughout.

    Uses a semi-implicit scheme for the cubic thinning term to ensure
    unconditional stability: h_{n+1} = h_n / (1 + α·h_n²·dt).
    """

    def __init__(
        self,
        rho: float = 1000.0,
        mu_solvent: float = 1e-3,
        alpha_visc: float = 4.5,
        beta_visc: float = 1.5,
        c_evap: float = 1.2e-6,
        c_solid: float = 0.15,
    ) -> None:
        super().__init__()
        self.rho = rho
        self.mu_solvent = mu_solvent
        self.alpha_visc = alpha_visc
        self.beta_visc = beta_visc
        self.c_evap = c_evap
        self.c_solid = c_solid

    @staticmethod
    def _viscosity(mu_solvent: float | Tensor, alpha: float, beta: float, c_val: Tensor) -> Tensor:
        visc_arg = alpha * torch.pow((1.0 - c_val) / c_val, beta)
        visc_arg = torch.nn.functional.softplus(20.0 - visc_arg) + visc_arg - 20.0
        return mu_solvent * torch.exp(visc_arg)

    def _compute_state(self, h: Tensor, m_polymer: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h_min = m_polymer / self.rho + 1e-8
        h_clamped = torch.clamp(h, min=h_min)
        c_val = torch.clamp(1.0 - m_polymer / (self.rho * h_clamped), min=1e-5, max=1.0)
        mu = self._viscosity(self.mu_solvent, self.alpha_visc, self.beta_visc, c_val)
        return h_clamped, c_val, mu, h_min

    def forward(
        self,
        omega_profile: Tensor,
        dt: float,
        h0: Tensor | float,
        c0: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        """Forward temporal integration of the Meyerhofer ODE system.

        Uses a semi-implicit scheme: linearize h³ ≈ h_n² · h_{n+1} for the
        centrifugal thinning term to get an unconditionally stable update.
        Adaptive sub-stepping further protects against stiff transients.

        Args:
            omega_profile: Spin speed over time, shape (N_steps,), in rad/s.
            dt: Time step in seconds.
            h0: Initial wet film thickness in meters.
            c0: Initial solvent mass fraction (0 to 1).

        Returns:
            (h_history, c_history): Wet thickness and solvent fraction over time.
        """
        device = omega_profile.device
        n_steps = omega_profile.shape[0]

        h = torch.as_tensor(h0, dtype=torch.float32, device=device)
        c = torch.as_tensor(c0, dtype=torch.float32, device=device)

        m_polymer = self.rho * (1.0 - c) * h

        h_history = []
        c_history = []

        for t in range(n_steps):
            omega = omega_profile[t]

            h_clamped, c_val, mu, h_min = self._compute_state(h, m_polymer)

            E = self.c_evap * torch.sqrt(torch.clamp(omega, min=0.0) + 1e-5)

            flow_mask = torch.sigmoid(50.0 * (c_val - self.c_solid))

            alpha_thin = (2.0 * self.rho * omega**2 * h_clamped**2) / (3.0 * mu) * flow_mask

            # Adaptive sub-stepping: estimate how many sub-steps needed
            dh_rate = alpha_thin * h_clamped + E
            cfl_ratio = dh_rate * dt / (h_clamped + 1e-12)
            n_sub = torch.clamp((cfl_ratio / 0.5).ceil().long(), min=1, max=_MAX_SUBSTEPS)
            n_sub_int = n_sub.item() if n_sub.numel() == 1 else int(n_sub.max())
            sub_dt = dt / n_sub_int

            for _ in range(n_sub_int):
                h_clamped, c_val, mu, h_min = self._compute_state(h, m_polymer)

                alpha_thin = (2.0 * self.rho * omega**2 * h_clamped**2) / (3.0 * mu) * flow_mask
                E_val = self.c_evap * torch.sqrt(torch.clamp(omega, min=0.0) + 1e-5)

                # Semi-implicit: h_{n+1} = h_n / (1 + alpha*h_n^2*dt + E*dt/h_n)
                denom = 1.0 + alpha_thin * h_clamped * sub_dt + E_val * sub_dt / (h_clamped + 1e-12)
                h = h_clamped / denom
                h = torch.clamp(h, min=h_min)

            # Record C derived from final h
            c_val = torch.clamp(1.0 - m_polymer / (self.rho * torch.clamp(h, min=h_min)), min=1e-5, max=1.0)

            h_history.append(h)
            c_history.append(c_val)

        return torch.stack(h_history), torch.stack(c_history)


class RadialThinFilmSolver(nn.Module):
    """1D Axisymmetric Radial PDE Solver for Spin Coating Uniformity.

    Models radial flow and evaporation on a wafer:
        dh/dt = -(1/r)·d/dr(r²·ρ·ω²·h³/(3μ(C))) - E(ω)

    Uses a 1D Finite Volume Method on a radial grid with upwind fluxes.
    Semi-implicit treatment of the cubic h³ term prevents stiffness blowup.
    """

    def __init__(
        self,
        r_max: float = 0.1,
        nr: int = 50,
        rho: float = 1000.0,
        mu_solvent: float = 1e-3,
        alpha_visc: float = 4.5,
        beta_visc: float = 1.5,
        c_evap: float = 1.2e-6,
        c_solid: float = 0.15,
    ) -> None:
        super().__init__()
        self.r_max = r_max
        self.nr = nr
        self.dr = r_max / nr
        self.rho = rho
        self.mu_solvent = mu_solvent
        self.alpha_visc = alpha_visc
        self.beta_visc = beta_visc
        self.c_evap = c_evap
        self.c_solid = c_solid

        r_centers = torch.linspace(self.dr / 2, r_max - self.dr / 2, nr)
        r_faces = torch.linspace(0.0, r_max, nr + 1)
        self.register_buffer("r_centers", r_centers)
        self.register_buffer("r_faces", r_faces)

    def forward(
        self,
        omega_profile: Tensor,
        dt: float,
        h0_profile: Tensor,
        c0_profile: Tensor,
    ) -> Tensor:
        """Solve the 1D radial PDE over time.

        Args:
            omega_profile: Spin speed over time, shape (N_steps,), in rad/s.
            dt: Time step in seconds.
            h0_profile: Initial wet thickness profile, shape (nr,).
            c0_profile: Initial solvent concentration profile, shape (nr,).

        Returns:
            h_final: Final thickness profile across wafer radius, shape (nr,).
        """
        n_steps = omega_profile.shape[0]
        r_c = self.r_centers
        r_f = self.r_faces

        h = h0_profile.clone().to(r_c.device)
        c = c0_profile.clone().to(r_c.device)

        m_polymer = self.rho * (1.0 - c) * h

        for t in range(n_steps):
            omega = omega_profile[t]

            h_min = m_polymer / self.rho + 1e-8
            h_clamped = torch.clamp(h, min=h_min)
            c_val = torch.clamp(
                1.0 - m_polymer / (self.rho * h_clamped), min=1e-5, max=1.0
            )

            mu = MeyerhoferSolver._viscosity(
                self.mu_solvent, self.alpha_visc, self.beta_visc, c_val
            )

            E = self.c_evap * torch.sqrt(torch.clamp(omega, min=0.0) + 1e-5)

            # Compute explicit divergence for adaptive sub-stepping
            flux = torch.zeros(self.nr + 1, device=r_c.device)
            h_face = h_clamped[:-1]
            mu_face = mu[:-1]
            r_face = r_f[1:-1]
            flux[1:-1] = (
                self.rho * omega**2 * r_face**2 * h_face**3 / (3.0 * mu_face)
            )
            flux[-1] = (
                self.rho
                * omega**2
                * r_f[-1] ** 2
                * h_clamped[-1] ** 3
                / (3.0 * mu[-1])
            )

            flow_mask = torch.sigmoid(50.0 * (c_val - self.c_solid))

            r_flux = r_f * flux
            div = (r_flux[1:] - r_flux[:-1]) / (r_c * self.dr)

            dh_rate = (div * flow_mask + E).abs()
            cfl_ratio = dh_rate * dt / (h_clamped + 1e-12)
            max_cfl = cfl_ratio.max()
            n_sub = max(1, min(int(max_cfl.item() / 0.5) + 1, _MAX_SUBSTEPS))
            sub_dt = dt / n_sub

            for _ in range(n_sub):
                h_min = m_polymer / self.rho + 1e-8
                h_clamped = torch.clamp(h, min=h_min)
                c_val = torch.clamp(
                    1.0 - m_polymer / (self.rho * h_clamped), min=1e-5, max=1.0
                )

                mu = MeyerhoferSolver._viscosity(
                    self.mu_solvent, self.alpha_visc, self.beta_visc, c_val
                )

                E_val = self.c_evap * torch.sqrt(torch.clamp(omega, min=0.0) + 1e-5)

                # Upwind fluxes at cell faces
                flux = torch.zeros(self.nr + 1, device=r_c.device)
                h_face = h_clamped[:-1]
                mu_face = mu[:-1]
                r_face = r_f[1:-1]

                # Semi-implicit: flux uses h_n^2 * h_{n+1} for cubic term
                # We compute an explicit estimate of divergence rate per unit h,
                # then apply the implicit correction.
                flux_coeff = self.rho * omega**2 * r_face**2 * h_face**2 / (3.0 * mu_face)
                flux[1:-1] = flux_coeff
                flux_coeff_bc = self.rho * omega**2 * r_f[-1]**2 * h_clamped[-1]**2 / (3.0 * mu[-1])
                flux[-1] = flux_coeff_bc

                flow_mask = torch.sigmoid(50.0 * (c_val - self.c_solid))

                r_flux = r_f * flux
                div_coeff = (r_flux[1:] - r_flux[:-1]) / (r_c * self.dr)

                # Semi-implicit update: h_{n+1} = h_n / (1 + div_coeff * h_n * flow_mask * sub_dt + E * sub_dt / h_n)
                denom = 1.0 + div_coeff * h_clamped * flow_mask * sub_dt + E_val * sub_dt / (h_clamped + 1e-12)
                h = h_clamped / denom
                h = torch.clamp(h, min=h_min)

        return h
