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

from quantmgmt.risk.ratios import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    tracking_error,
)

from quantmgmt.risk.tail import (
    var_historical,
    cvar_historical,
    var_parametric,
    cvar_parametric,
    skewness,
    kurtosis,
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


# --- Métrica propia: probabilidad de susto ------------------------------------


def make_scare_returns() -> pd.Series:
    """Valor al cierre: 1, 0,9, 0,72, 1,08, 1,08."""
    return make_series([0.0, -0.10, -0.20, 0.50, 0.0])


def test_worst_loss_ahead():
    # Con horizonte 2: quien entra en 1 cae hasta 0,72 (-28 %); en 0,9, hasta
    # 0,72 (-20 %); en 0,72 ya solo sube (0 %).
    worst = risk.worst_loss_ahead(make_scare_returns(), horizon=2)
    np.testing.assert_allclose(worst, [0.28, 0.20, 0.0])


def test_scare_probability():
    returns = make_scare_returns()
    # Pérdidas de 28 %, 20 % y 0 %: dos de tres pasan del 15 %, una del 25 %.
    assert risk.scare_probability(returns, 0.15, horizon=2) == pytest.approx(2 / 3)
    assert risk.scare_probability(returns, 0.25, horizon=2) == pytest.approx(1 / 3)


def test_scare_probability_per_column():
    returns = pd.DataFrame(
        {"A": [0.0, -0.10, -0.20, 0.50, 0.0], "B": [0.01] * 5}
    )
    result = risk.scare_probability(returns, 0.15, horizon=2)
    assert result.to_dict() == pytest.approx({"A": 2 / 3, "B": 0.0})


def test_scare_probability_without_full_horizon_is_nan():
    assert np.isnan(risk.scare_probability(make_scare_returns(), horizon=10))


def test_scare_probability_rejects_invalid_limit():
    with pytest.raises(ValueError, match="loss_limit"):
        risk.scare_probability(make_scare_returns(), loss_limit=1.5)


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


# --- Ratios ----------------------------------------------------


def test_sharpe_ratio_known_result():
    # Comprueba la resta del tipo libre de riesgo y la anualización.
    returns = pd.Series([-0.01, 0.0, 0.01])

    result = sharpe_ratio(
        returns,
        periods_per_year=12,
        risk_free_rate=0.01,
    )

    # Media en exceso -0.01 y desviación típica 0.01 por período.
    expected = -np.sqrt(12)
    assert result == pytest.approx(expected)


def test_sortino_ratio_nonzero_target():
    # Comprueba que el objetivo se aplique una sola vez y se anualice el ratio.
    returns = pd.Series([-0.01, 0.02, 0.04])

    result = sortino_ratio(
        returns,
        periods_per_year=12,
        target=0.01,
    )

    # Media en exceso anual 0.08 y desviación a la baja anual 0.04.
    assert result == pytest.approx(2.0)


def test_calmar_ratio_multiple_assets():
    # Comprueba el CAGR, el signo del drawdown y los resultados por columna.
    dates = pd.to_datetime(["2025-04-30", "2025-08-31", "2025-12-31"])
    returns = pd.DataFrame(
        {
            "Activo A": [0.10, -0.20, 0.25],
            "Activo B": [0.20, -0.50, 1.00],
        },
        index=dates,
    )

    result = calmar_ratio(returns, periods_per_year=3)

    # Un año: A gana 10 % y cae 20 %; B gana 20 % y cae 50 %.
    expected = pd.Series(
        [0.10 / 0.20, 0.20 / 0.50],
        index=returns.columns,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-7,
        atol=1e-12,
    )


def test_tracking_error_aligns_dates():
    # Comprueba que se comparen fechas comunes y se anualicen las diferencias.
    returns = pd.Series(
        [0.50, 0.01, 0.03, 0.05],
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )
    benchmark = pd.Series(
        [0.02, 0.03, 0.04, -0.50],
        index=pd.date_range("2026-01-02", periods=4, freq="D"),
    )

    result = tracking_error(
        returns,
        benchmark,
        periods_per_year=252,
    )

    # En las tres fechas comunes, las diferencias son -0.01, 0 y 0.01.
    expected = 0.01 * np.sqrt(252)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "function",
    [sharpe_ratio, sortino_ratio, calmar_ratio],
)
def test_ratios_zero_denominator(function):
    # Comprueba que un denominador cero devuelva NaN, sin dividir entre cero.
    returns = pd.Series(
        [0.0, 0.0, 0.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    result = function(returns, periods_per_year=252)

    assert np.isnan(result)


# --- Riesgo de Cola ----------------------------------------------------


@pytest.mark.parametrize(
    "confidence, expected_var, expected_cvar",
    [
        (0.75, 0.04, 0.06),
        (0.80, 0.048, 0.08),
    ],
)
def test_historical_tail_known_results(
    confidence, expected_var, expected_cvar
):
    # Comprueba cuantiles, selección de cola, valores ausentes y varias columnas.
    returns = pd.DataFrame(
        {
            "Activo A": [-0.08, -0.04, 0.0, 0.02, 0.04, np.nan],
            "Activo B": [-0.16, -0.08, 0.0, 0.04, 0.08, np.nan],
        }
    )

    result_var = var_historical(returns, confidence)
    result_cvar = cvar_historical(returns, confidence)

    # El segundo activo duplica los retornos y también ambas medidas.

    expected_var = pd.Series(
    [expected_var, 2 * expected_var],
    index=returns.columns,
    name=1.0 - confidence,
    )
    
    expected_cvar = pd.Series(
        [expected_cvar, 2 * expected_cvar],
        index=returns.columns,
    )

    pd.testing.assert_series_equal(
        result_var, expected_var,
        check_exact=False, rtol=1e-7, atol=1e-12,
    )
    pd.testing.assert_series_equal(
        result_cvar, expected_cvar,
        check_exact=False, rtol=1e-7, atol=1e-12,
    )


def test_skewness_and_excess_kurtosis():
    # Comprueba los estimadores muestrales y que la curtosis sea exceso de curtosis.
    returns = pd.Series([0.0, 0.0, 0.0, 0.04])

    assert skewness(returns) == pytest.approx(2.0)
    assert kurtosis(returns) == pytest.approx(4.0)


@pytest.mark.parametrize("method", ["gaussian", "cornish_fisher"])
def test_var_parametric_known_result(method):
    # Comprueba ambos métodos con media, volatilidad y momentos conocidos.
    returns = pd.Series([0.0, 0.0, 0.0, 0.04])

    result = var_parametric(
        returns,
        confidence=0.95,
        method=method,
    )

    # Media 0.01, desviación típica 0.02 y cuantil normal del 5 %.
    z = -1.6448536269514722

    if method == "cornish_fisher":
        # Esta muestra tiene asimetría 2 y exceso de curtosis 4.
        z = (
            z
            + (z**2 - 1) * 2 / 6
            + (z**3 - 3 * z) * 4 / 24
            - (2 * z**3 - 5 * z) * 4 / 36
        )

    expected = -(0.01 + 0.02 * z)
    assert result == pytest.approx(expected)


def test_cvar_parametric_known_result():
    # Comprueba el CVaR normal al 95 % con media cero y volatilidad 0.01.
    returns = pd.Series([-0.01, 0.0, 0.01])

    result = cvar_parametric(returns, confidence=0.95)

    # La pérdida media de la cola normal es aproximadamente 2.0627 veces sigma.
    expected = 0.02062712807507425
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "function",
    [
        var_historical,
        cvar_historical,
        var_parametric,
        cvar_parametric,
    ],
)
@pytest.mark.parametrize("confidence", [0.0, 1.0, np.nan, np.inf])
def test_tail_invalid_confidence(function, confidence):
    # Comprueba que las cuatro funciones rechacen niveles de confianza inválidos.
    returns = pd.Series([-0.02, 0.0, 0.01, 0.03])

    with pytest.raises(ValueError, match="confidence"):
        function(returns, confidence=confidence)


def test_var_parametric_invalid_method():
    # Comprueba que un método desconocido produzca un error explícito.
    returns = pd.Series([-0.02, 0.0, 0.01, 0.03])

    with pytest.raises(ValueError, match="method"):
        var_parametric(returns, method="otro")


# --- Tabla resumen --------------------------------------------------------------


def test_risk_summary_has_one_row_per_asset():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.015, -0.005, 0.02] * 60,
            "B": [0.002, -0.001, 0.003, -0.002, 0.001] * 60,
        },
        index=pd.date_range("2024-01-01", periods=300, freq="D"),
    )
    summary = risk.risk_summary(returns, benchmark=returns["A"])

    # Una fila por cartera y los mismos valores que las funciones sueltas.
    assert list(summary.index) == ["A", "B"]
    assert summary.loc["B", "CAGR"] == pytest.approx(risk.cagr(returns["B"]))
    assert summary.loc["A", "Máx. drawdown"] == pytest.approx(
        risk.max_drawdown(returns["A"])
    )
    assert summary.loc["A", "Tracking error"] == pytest.approx(0.0)
    assert not summary.drop(columns="Tiempo de recuperación").isna().any().any()
