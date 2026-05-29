"""Tests for sCO₂ property surrogate (C4)."""

import torch


def test_sco2_import():
    from diffcfd.props.sco2 import SCO2Surrogate

    assert SCO2Surrogate is not None


def test_sco2_outputs_positive():
    from diffcfd.props.sco2 import SCO2Surrogate

    model = SCO2Surrogate(hidden_dim=32)
    T = torch.tensor([300.0, 304.0, 310.0, 330.0])
    p = torch.tensor([7.5e6, 8.0e6, 7.0e6, 9.0e6])

    rho = model.density(T, p)
    mu = model.viscosity(T, p)
    k = model.conductivity(T, p)
    cp = model.specific_heat(T, p)

    assert rho.shape == (4,)
    assert (rho > 0).all(), "Density must be positive"
    assert (mu > 0).all(), "Viscosity must be positive"
    assert (k > 0).all(), "Conductivity must be positive"
    assert (cp > 0).all(), "Specific heat must be positive"


def test_sco2_differentiable():
    from diffcfd.props.sco2 import SCO2Surrogate

    model = SCO2Surrogate(hidden_dim=32)
    T = torch.tensor([305.0, 310.0], requires_grad=True)
    p = torch.tensor([7.5e6, 8.0e6], requires_grad=True)

    rho = model.density(T, p)
    loss = rho.sum()
    loss.backward()

    assert T.grad is not None, "Gradient w.r.t. T must exist"
    assert p.grad is not None, "Gradient w.r.t. p must exist"
    assert torch.isfinite(T.grad).all()


def test_sco2_training_runs():
    from diffcfd.props.sco2 import train_sco2_surrogate

    model = train_sco2_surrogate(hidden_dim=16, epochs=5, n_samples=100, verbose=False)
    assert model._trained

    T = torch.tensor([304.0])
    p = torch.tensor([7.5e6])
    rho = model.density(T, p)
    assert rho.item() > 0


def test_sco2_training_data():
    from diffcfd.props.sco2 import generate_training_data

    data = generate_training_data(n_samples=100)
    assert data["rho"].shape == (10000,)
    assert (data["rho"] > 0).all()
    assert (data["mu"] > 0).all()
    assert (data["k"] > 0).all()
    assert (data["cp"] > 0).all()


def test_sco2_inherits_thermophysical_props():
    from diffcfd.props.sco2 import SCO2Surrogate
    from diffcfd.props.ideal_gas import ThermophysicalProps

    model = SCO2Surrogate(hidden_dim=16)
    assert isinstance(model, ThermophysicalProps)
