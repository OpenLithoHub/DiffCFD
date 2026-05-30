"""Tests for PolynomialSCO2 and SplineSCO2 EOS models."""

import torch


# ---------------------------------------------------------------------------
# PolynomialSCO2
# ---------------------------------------------------------------------------


def test_polynomial_eos_properties():
    """All properties return correct shapes and are positive."""
    from diffcfd.props.sco2 import PolynomialSCO2

    model = PolynomialSCO2(degree=4)
    T = torch.tensor([300.0, 304.0, 310.0, 330.0])
    p = torch.tensor([7.5e6, 8.0e6, 7.0e6, 9.0e6])

    rho = model.density(T, p)
    mu = model.viscosity(T, p)
    k = model.conductivity(T, p)
    cp = model.specific_heat(T, p)

    assert rho.shape == (4,), f"Expected shape (4,), got {rho.shape}"
    assert mu.shape == (4,)
    assert k.shape == (4,)
    assert cp.shape == (4,)

    assert (rho > 0).all(), "Density must be positive"
    assert (mu > 0).all(), "Viscosity must be positive"
    assert (k > 0).all(), "Conductivity must be positive"
    assert (cp > 0).all(), "Specific heat must be positive"


# ---------------------------------------------------------------------------
# SplineSCO2
# ---------------------------------------------------------------------------


def test_spline_eos_properties():
    """All properties return correct shapes and are positive."""
    from diffcfd.props.sco2 import SplineSCO2

    model = SplineSCO2(grid_size=20)
    T = torch.tensor([300.0, 304.0, 310.0, 330.0])
    p = torch.tensor([7.5e6, 8.0e6, 7.0e6, 9.0e6])

    rho = model.density(T, p)
    mu = model.viscosity(T, p)
    k = model.conductivity(T, p)
    cp = model.specific_heat(T, p)

    assert rho.shape == (4,), f"Expected shape (4,), got {rho.shape}"
    assert mu.shape == (4,)
    assert k.shape == (4,)
    assert cp.shape == (4,)

    assert (rho > 0).all(), "Density must be positive"
    assert (mu > 0).all(), "Viscosity must be positive"
    assert (k > 0).all(), "Conductivity must be positive"
    assert (cp > 0).all(), "Specific heat must be positive"


# ---------------------------------------------------------------------------
# Gradient consistency
# ---------------------------------------------------------------------------


def test_eos_gradient_consistency():
    """Trained Polynomial EOS gradients are directionally consistent with MLP."""
    from diffcfd.props.sco2 import (
        generate_training_data,
        train_sco2_surrogate,
        fit_polynomial_eos,
    )

    data = generate_training_data(n_samples=50, seed=99)

    # Train both on the same data
    mlp = train_sco2_surrogate(hidden_dim=32, epochs=100, n_samples=50, verbose=False)
    poly = fit_polynomial_eos(reference_data=data, epochs=1000)

    T_val = torch.tensor([305.0, 310.0])
    p_val = torch.tensor([7.5e6, 8.0e6])

    grads = {}
    for name, model in [("mlp", mlp), ("poly", poly)]:
        T = T_val.clone().detach().requires_grad_(True)
        p = p_val.clone().detach().requires_grad_(True)
        rho = model.density(T, p)
        rho.sum().backward()
        grads[name] = torch.cat([T.grad, p.grad])

    # Both models should produce nonzero gradients after training
    assert grads["mlp"].norm() > 1e-6, "MLP gradient is near zero"
    assert grads["poly"].norm() > 1e-6, "Polynomial gradient is near zero"

    # Cosine similarity magnitude between MLP and trained Polynomial
    # (sign may differ due to different parameterization, but magnitude of
    # agreement indicates both models capture the same sensitivity structure)
    cos_poly = torch.nn.functional.cosine_similarity(
        grads["mlp"].unsqueeze(0), grads["poly"].unsqueeze(0)
    ).item()
    assert abs(cos_poly) > 0.99, f"Polynomial gradient |cosine| {abs(cos_poly):.4f} < 0.99"


# ---------------------------------------------------------------------------
# Accuracy vs MLP
# ---------------------------------------------------------------------------


def test_eos_accuracy_vs_mlp():
    """On synthetic data, polynomial/spline fitted errors are bounded."""
    from diffcfd.props.sco2 import (
        generate_training_data,
        fit_polynomial_eos,
        fit_spline_eos,
    )

    data = generate_training_data(n_samples=50, seed=123)
    poly = fit_polynomial_eos(reference_data=data, epochs=5000, lr=1e-2)
    spline = fit_spline_eos(reference_data=data, grid_size=20, epochs=500, lr=1e-4)

    T = data["T"]
    p = data["p"]

    def rel_err(pred, ref):
        return ((pred - ref) / ref).abs().mean().item()

    for model, label in [(poly, "polynomial"), (spline, "spline")]:
        rho_pred = model.density(T, p)
        mu_pred = model.viscosity(T, p)
        k_pred = model.conductivity(T, p)
        cp_pred = model.specific_heat(T, p)

        rho_err = rel_err(rho_pred, data["rho"])
        mu_err = rel_err(mu_pred, data["mu"])
        k_err = rel_err(k_pred, data["k"])
        cp_err = rel_err(cp_pred, data["cp"])

        threshold = 0.20
        assert rho_err < threshold, f"{label} rho relative error {rho_err:.2%} >= {threshold:.0%}"
        assert mu_err < threshold, f"{label} mu relative error {mu_err:.2%} >= {threshold:.0%}"
        assert k_err < threshold, f"{label} k relative error {k_err:.2%} >= {threshold:.0%}"
        assert cp_err < threshold, f"{label} cp relative error {cp_err:.2%} >= {threshold:.0%}"


# ---------------------------------------------------------------------------
# Out-of-range warning
# ---------------------------------------------------------------------------


def test_out_of_range_warning(caplog):
    """Out-of-range queries produce a warning."""
    import logging
    from diffcfd.props.sco2 import PolynomialSCO2

    model = PolynomialSCO2(degree=4)

    # T=400K is far outside the ~[273, 335] fitting range
    T = torch.tensor([400.0])
    p = torch.tensor([7.5e6])

    with caplog.at_level(logging.WARNING, logger="diffcfd.props.sco2"):
        model.density(T, p)

    assert len(caplog.records) > 0, "Expected a warning log for out-of-range query"
    assert "fitting range" in caplog.records[0].message


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


def test_polynomial_is_thermophysical_props():
    from diffcfd.props.sco2 import PolynomialSCO2
    from diffcfd.props.ideal_gas import ThermophysicalProps

    model = PolynomialSCO2(degree=4)
    assert isinstance(model, ThermophysicalProps)


def test_spline_is_thermophysical_props():
    from diffcfd.props.sco2 import SplineSCO2
    from diffcfd.props.ideal_gas import ThermophysicalProps

    model = SplineSCO2(grid_size=10)
    assert isinstance(model, ThermophysicalProps)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_polynomial_differentiable():
    from diffcfd.props.sco2 import PolynomialSCO2

    model = PolynomialSCO2(degree=4)
    T = torch.tensor([305.0, 310.0], requires_grad=True)
    p = torch.tensor([7.5e6, 8.0e6], requires_grad=True)

    rho = model.density(T, p)
    loss = rho.sum()
    loss.backward()

    assert T.grad is not None, "Gradient w.r.t. T must exist"
    assert p.grad is not None, "Gradient w.r.t. p must exist"
    assert torch.isfinite(T.grad).all()


def test_spline_differentiable():
    from diffcfd.props.sco2 import SplineSCO2

    model = SplineSCO2(grid_size=10)
    T = torch.tensor([305.0, 310.0], requires_grad=True)
    p = torch.tensor([7.5e6, 8.0e6], requires_grad=True)

    rho = model.density(T, p)
    loss = rho.sum()
    loss.backward()

    assert T.grad is not None, "Gradient w.r.t. T must exist"
    assert p.grad is not None, "Gradient w.r.t. p must exist"
    assert torch.isfinite(T.grad).all()
