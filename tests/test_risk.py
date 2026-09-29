"""Pruebas del módulo de riesgo con valores calculados a mano."""

import numpy as np
import pandas as pd
import pytest

from quantmgmt import risk


def make_series(values: list[float]) -> pd.Series:
    """Serie diaria de ejemplo indexada por fecha."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index, name="A")


# --- Rentabilidad ------------------------------------------------------------


def test_to_returns_simple():
    prices = make_series([100.0, 110.0, 99.0])
    np.testing.assert_allclose(risk.to_returns(prices), [0.10, -0.10])


def test_to_returns_log():
    prices = make_series([100.0, 110.0, 99.0])
    np.testing.assert_allclose(
        risk.to_returns(prices, method="log"), [np.log(1.1), np.log(0.9)]
    )


def test_to_returns_rejects_unknown_method():
    with pytest.raises(ValueError):
        risk.to_returns(make_series([100.0, 110.0]), method="otro")


def test_cumulative_returns():
    # +10 % y después -10 %: 1,1 × 0,9 = 0,99, es decir, -1 % en total.
    returns = make_series([0.10, -0.10])
    np.testing.assert_allclose(risk.cumulative_returns(returns), [0.10, -0.01])


def test_cagr():
    # Dos años al +10 %: crece un 21 % en total, que es un 10 % anual compuesto.
    returns = make_series([0.10, 0.10])
    assert risk.cagr(returns, periods_per_year=1) == pytest.approx(0.10)


def test_annualized_return():
    # Media diaria del 0,2 % × 252 días = 50,4 %.
    returns = make_series([0.001, 0.003])
    assert risk.annualized_return(returns, periods_per_year=252) == pytest.approx(
        0.504
    )


def test_dataframe_gives_one_value_per_column():
    returns = pd.DataFrame({"A": [0.10, 0.10], "B": [0.0, 0.0]})
    result = risk.cagr(returns, periods_per_year=1)
    assert isinstance(result, pd.Series)
    assert result.to_dict() == pytest.approx({"A": 0.10, "B": 0.0})


def test_rejects_non_pandas_input():
    with pytest.raises(TypeError):
        risk.cagr([0.10, 0.10])
