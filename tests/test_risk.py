"""Pruebas del módulo de riesgo con valores calculados a mano."""

import numpy as np
import pandas as pd
import pytest

from quantmgmt import risk
from quantmgmt.risk.dispersion import (
    annualized_volatility,
    downside_deviation,
    rolling_volatility,
)


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


# --- Caídas desde máximos ----------------------------------------------------


def make_drawdown_returns() -> pd.Series:
    """Valor de 1 €: 1,25 (máximo), 1, 0,5 (fondo), 1, 1,5 (nuevo máximo), 1,35."""
    return make_series([0.25, -0.20, -0.50, 1.00, 0.50, -0.10])


def test_drawdown_series():
    np.testing.assert_allclose(
        risk.drawdown_series(make_drawdown_returns()),
        [0.0, 0.20, 0.60, 0.20, 0.0, 0.10],
        atol=1e-12,
    )


def test_max_drawdown():
    # La peor caída va de 1,25 a 0,5: un 60 %.
    assert risk.max_drawdown(make_drawdown_returns()) == pytest.approx(0.60)


def test_drawdown_counts_losses_from_initial_capital():
    # Si el primer día cae un 10 %, ya es una caída desde el capital inicial.
    assert risk.max_drawdown(make_series([-0.10, 0.05])) == pytest.approx(0.10)


def test_time_under_water():
    # Tres periodos seguidos por debajo de 1,25 hasta superarlo con 1,5.
    assert risk.time_under_water(make_drawdown_returns()) == 3


def test_recovery_time():
    # Del fondo (0,5) a superar el máximo anterior (1,5): dos periodos.
    assert risk.recovery_time(make_drawdown_returns()) == 2


def test_recovery_time_is_nan_if_not_recovered():
    # 1,1 → 0,55 → 0,605: no vuelve a 1,1.
    assert np.isnan(risk.recovery_time(make_series([0.10, -0.50, 0.10])))


def test_drawdown_metrics_per_column():
    # Una columna con la caída de ejemplo y otra que solo sube.
    returns = pd.DataFrame(
        {"A": [0.25, -0.20, -0.50, 1.00, 0.50, -0.10], "B": [0.01] * 6}
    )
    expected_max = {"A": 0.60, "B": 0.0}
    assert risk.max_drawdown(returns).to_dict() == pytest.approx(expected_max)
    assert risk.time_under_water(returns).to_dict() == {"A": 3.0, "B": 0.0}
    assert risk.recovery_time(returns).to_dict() == {"A": 2.0, "B": 0.0}


# --- Dispersión --------------------------------------------------------------


def test_annualized_volatility_known_result():
    # Comprueba la desviación típica muestral y su anualización.
    returns = pd.Series([-0.01, 0.0, 0.01])

    result = annualized_volatility(returns, periods_per_year=252)

    # La desviación típica muestral de estos retornos es 0.01.
    expected = 0.01 * np.sqrt(252)
    assert result == pytest.approx(expected)


def test_downside_deviation_counts_all_periods():
    # Comprueba que los retornos sobre el objetivo cuentan como cero, sin excluirse.
    returns = pd.Series([-0.02, 0.01, 0.03])

    result = downside_deviation(
        returns,
        periods_per_year=1,
        target=0.0,
    )

    # Solo un retorno aporta pérdida, pero el divisor incluye los tres.
    expected = 0.02 / np.sqrt(3)
    assert result == pytest.approx(expected)


def test_rolling_volatility_known_windows():
    # Comprueba las ventanas, el valor inicial ausente y las etiquetas.
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    returns = pd.Series(
        [0.0, 0.02, 0.06],
        index=dates,
        name="Activo",
    )

    result = rolling_volatility(
        returns,
        window=2,
        periods_per_year=1,
    )

    # La primera ventana está incompleta; las siguientes contienen dos datos.
    expected = pd.Series(
        [np.nan, 0.02 / np.sqrt(2), 0.04 / np.sqrt(2)],
        index=dates,
        name="Activo",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
    )


def test_downside_deviation_nonzero_target():
    # Comprueba que las desviaciones se calculan respecto al objetivo indicado.
    returns = pd.Series([-0.01, 0.02, 0.04])

    result = downside_deviation(
        returns,
        periods_per_year=1,
        target=0.01,
    )

    # Las desviaciones desfavorables son -0.02, 0 y 0.
    expected = 0.02 / np.sqrt(3)
    assert result == pytest.approx(expected)


def test_dispersion_multiple_assets():
    # Comprueba resultados por activo, anualización y conservación de etiquetas.
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {
            "Activo A": [-0.01, 0.0, 0.01],
            "Activo B": [-0.02, 0.0, 0.02],
        },
        index=dates,
    )

    annualized = annualized_volatility(
        returns,
        periods_per_year=12,
    )
    downside = downside_deviation(
        returns,
        periods_per_year=12,
    )
    rolling = rolling_volatility(
        returns,
        window=2,
        periods_per_year=12,
    )

    expected_annualized = pd.Series(
        [0.01 * np.sqrt(12), 0.02 * np.sqrt(12)],
        index=returns.columns,
    )
    expected_downside = pd.Series(
        [0.02, 0.04],
        index=returns.columns,
    )
    expected_rolling = pd.DataFrame(
        {
            "Activo A": [
                np.nan,
                0.01 * np.sqrt(6),
                0.01 * np.sqrt(6),
            ],
            "Activo B": [
                np.nan,
                0.02 * np.sqrt(6),
                0.02 * np.sqrt(6),
            ],
        },
        index=dates,
    )

    pd.testing.assert_series_equal(
        annualized,
        expected_annualized,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        downside,
        expected_downside,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        rolling,
        expected_rolling,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
    )


def test_dispersion_constant_positive_returns():
    # Comprueba volatilidad cero y ausencia de desviaciones bajo un objetivo cero.
    returns = pd.Series([0.02, 0.02, 0.02])

    assert annualized_volatility(returns) == pytest.approx(0.0)
    assert downside_deviation(returns) == pytest.approx(0.0)

    result = rolling_volatility(returns, window=2)
    expected = pd.Series([np.nan, 0.0, 0.0])

    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
    )


def test_dispersion_missing_values():
    # Comprueba que se omiten los ausentes, pero rolling exige ventanas completas.
    returns = pd.Series([-0.01, np.nan, 0.0, 0.01])

    annualized = annualized_volatility(
        returns,
        periods_per_year=1,
    )
    downside = downside_deviation(
        returns,
        periods_per_year=1,
    )
    rolling = rolling_volatility(
        returns,
        window=2,
        periods_per_year=1,
    )

    assert annualized == pytest.approx(0.01)
    assert downside == pytest.approx(0.01 / np.sqrt(3))

    # Solo la última ventana contiene dos observaciones válidas.
    expected = pd.Series(
        [np.nan, np.nan, np.nan, 0.01 / np.sqrt(2)]
    )

    pd.testing.assert_series_equal(
        rolling,
        expected,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
    )


def test_volatility_insufficient_observations():
    # Comprueba que una observación válida no permite estimar volatilidad muestral.
    returns = pd.Series([np.nan, 0.01])

    assert np.isnan(annualized_volatility(returns))

    result = rolling_volatility(returns, window=2)
    assert result.isna().all()


@pytest.mark.parametrize(
    "function",
    [
        annualized_volatility,
        downside_deviation,
        rolling_volatility,
    ],
)
@pytest.mark.parametrize(
    "periods_per_year",
    [0, -252, np.nan, np.inf, -np.inf],
)
def test_dispersion_invalid_periods_per_year(function, periods_per_year):
    # Comprueba que las tres funciones rechazan una anualización inválida.
    returns = pd.Series([-0.01, 0.0, 0.01])

    with pytest.raises(ValueError, match="periods_per_year"):
        function(returns, periods_per_year=periods_per_year)


@pytest.mark.parametrize("window", [0, 1, -2, 2.5])
def test_rolling_volatility_invalid_window(window):
    # Comprueba que la ventana sea un entero de al menos dos observaciones.
    returns = pd.Series([-0.01, 0.0, 0.01])

    with pytest.raises(ValueError, match="window"):
        rolling_volatility(returns, window=window)


@pytest.mark.parametrize("target", [np.nan, np.inf, -np.inf])
def test_downside_deviation_invalid_target(target):
    # Comprueba que el objetivo sea un número finito.
    returns = pd.Series([-0.01, 0.0, 0.01])

    with pytest.raises(ValueError, match="target"):
        downside_deviation(returns, target=target)
