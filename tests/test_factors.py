"""Pruebas de las regresiones multifactoriales con datos sintéticos de alfa y betas conocidas."""

import numpy as np
import pandas as pd
import pytest

from quantmgmt.analytics import rolling_betas, run_factor_regression

ALPHA_MENSUAL = 0.002  # 2,4 % anual
BETAS = {"Mkt-RF": 1.2, "SMB": 0.5, "HML": -0.3}


def make_factors(n: int = 240) -> pd.DataFrame:
    """Factores mensuales aleatorios con los nombres de Ken French."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2005-01-31", periods=n, freq="ME")
    data = rng.normal(0.005, 0.04, size=(n, 5))
    factors = pd.DataFrame(data, index=index, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    factors["Mom"] = rng.normal(0.005, 0.04, n)
    factors["RF"] = 0.001
    return factors


def make_returns(factors: pd.DataFrame, noise: float = 0.001) -> pd.Series:
    """Retornos (no en exceso) generados con un modelo FF3 conocido."""
    rng = np.random.default_rng(1)
    excess = ALPHA_MENSUAL + sum(beta * factors[name] for name, beta in BETAS.items())
    excess = excess + rng.normal(0, noise, len(factors))
    return (excess + factors["RF"]).rename("cartera")


def test_ff3_recovers_alpha_and_betas():
    factors = make_factors()
    result = run_factor_regression(make_returns(factors), factors, model="FF3")

    assert result["alpha"] == pytest.approx(ALPHA_MENSUAL * 12, abs=0.002)
    for name, beta in BETAS.items():
        assert result[f"beta_{name}"] == pytest.approx(beta, abs=0.01)
    assert result["r2_adj"] > 0.99
    assert result["n_obs"] == len(factors)


def test_capm_misses_the_other_factors():
    # Sin SMB ni HML, el CAPM explica peor la cartera.
    factors = make_factors()
    returns = make_returns(factors)
    capm = run_factor_regression(returns, factors, model="CAPM")
    ff3 = run_factor_regression(returns, factors, model="FF3")
    assert capm["r2_adj"] < ff3["r2_adj"]
    assert "beta_SMB" not in capm.index


def test_dataframe_gives_one_row_per_portfolio():
    factors = make_factors()
    returns = pd.DataFrame({"A": make_returns(factors), "B": factors["Mkt-RF"] + factors["RF"]})
    result = run_factor_regression(returns, factors, model="Carhart4")

    assert list(result.index) == ["A", "B"]
    # B es exactamente el mercado: beta 1 y alfa 0.
    assert result.loc["B", "beta_Mkt-RF"] == pytest.approx(1.0)
    assert result.loc["B", "alpha"] == pytest.approx(0.0, abs=1e-12)


def test_newey_west_changes_t_but_not_coefficients():
    factors = make_factors()
    returns = make_returns(factors, noise=0.02)
    nw = run_factor_regression(returns, factors, model="FF5", newey_west=True)
    ols = run_factor_regression(returns, factors, model="FF5", newey_west=False)
    assert nw["alpha"] == pytest.approx(ols["alpha"])
    assert nw["t_alpha"] != pytest.approx(ols["t_alpha"])


def test_rolling_betas_shape_and_values():
    factors = make_factors()
    betas = rolling_betas(make_returns(factors), factors, model="FF3", window=36)

    assert len(betas) == len(factors) - 36 + 1
    assert list(betas.columns) == ["alpha", "Mkt-RF", "SMB", "HML"]
    assert betas["Mkt-RF"].mean() == pytest.approx(1.2, abs=0.02)


def test_invalid_inputs():
    factors = make_factors()
    returns = make_returns(factors)
    with pytest.raises(ValueError, match="model"):
        run_factor_regression(returns, factors, model="FF7")
    with pytest.raises(ValueError, match="Mom"):
        run_factor_regression(returns, factors.drop(columns="Mom"), model="Carhart4")
    with pytest.raises(ValueError, match="ventana"):
        rolling_betas(returns.iloc[:20], factors, window=36)
