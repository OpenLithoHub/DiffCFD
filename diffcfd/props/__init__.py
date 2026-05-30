"""Thermophysical properties: constant, sCO₂ transcritical surrogate."""

from diffcfd.props.ideal_gas import ThermophysicalProps, ConstantProps
from diffcfd.props.sco2 import SCO2Surrogate, train_sco2_surrogate

__all__ = [
    "ThermophysicalProps",
    "ConstantProps",
    "SCO2Surrogate",
    "train_sco2_surrogate",
]
