"""Thermophysical properties: ideal gas (open-source), constant props, abstract interface.

All fluid property calls in diffcfd go through ThermophysicalProps so that the
open-source core never imports a specific implementation.  The sCO2 transcritical
surrogate (C4) lives in a separate private repo (diffcfd-sco2-pro) that subclasses
this interface — allowing dual-licensing without exposing C4 in the open-source package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ThermophysicalProps(ABC):
    """Abstract interface for fluid thermophysical properties.

    All methods accept scalar or tensor inputs and return tensors of the same shape.
    Gradients must flow through all outputs (no detach).
    """

    @abstractmethod
    def density(self, T: Tensor, p: Tensor) -> Tensor:
        """Mass density ρ [kg/m³]."""

    @abstractmethod
    def viscosity(self, T: Tensor, p: Tensor) -> Tensor:
        """Dynamic viscosity μ [Pa·s]."""

    @abstractmethod
    def conductivity(self, T: Tensor, p: Tensor) -> Tensor:
        """Thermal conductivity k [W/(m·K)]."""

    @abstractmethod
    def specific_heat(self, T: Tensor, p: Tensor) -> Tensor:
        """Isobaric specific heat cp [J/(kg·K)]."""


class ConstantProps(ThermophysicalProps):
    """Constant (temperature- and pressure-independent) properties.

    Suitable for incompressible isothermal flow at fixed Re.

    Args:
        rho: Density [kg/m³].
        mu: Dynamic viscosity [Pa·s].
        k: Thermal conductivity [W/(m·K)].
        cp: Specific heat [J/(kg·K)].
    """

    def __init__(
        self,
        rho: float = 1.0,
        mu: float = 1.0,
        k: float = 1.0,
        cp: float = 1.0,
    ) -> None:
        self._rho_val = rho
        self._mu_val = mu
        self._k_val = k
        self._cp_val = cp

    def _tensor(self, val: float, ref: Tensor) -> Tensor:
        return torch.tensor(val, dtype=ref.dtype, device=ref.device).expand_as(ref)

    def density(self, T: Tensor, p: Tensor) -> Tensor:
        return self._tensor(self._rho_val, T)

    def viscosity(self, T: Tensor, p: Tensor) -> Tensor:
        return self._tensor(self._mu_val, T)

    def conductivity(self, T: Tensor, p: Tensor) -> Tensor:
        return self._tensor(self._k_val, T)

    def specific_heat(self, T: Tensor, p: Tensor) -> Tensor:
        return self._tensor(self._cp_val, T)
