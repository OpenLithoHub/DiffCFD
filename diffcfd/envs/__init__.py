"""Gymnasium environments (v0.3+)."""

from diffcfd.envs.base import DiffCFDEnv
from diffcfd.envs.blowing_suction import BlowingSuctionEnv
from diffcfd.envs.cylinder_wake import CylinderWakeEnv
from diffcfd.envs.heat_exchanger import HeatExchangerEnv

__all__ = [
    "BlowingSuctionEnv",
    "CylinderWakeEnv",
    "DiffCFDEnv",
    "HeatExchangerEnv",
]
