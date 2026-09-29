"""Ratios de rentabilidad ajustada al riesgo."""

import numpy as np
import pandas as pd
from quantmgmt.risk.returns import annualized_return, cagr
from quantmgmt.risk.drawdown import max_drawdown

from quantmgmt.risk._checks import (
    PandasData,
    check_finite,
    check_pandas,
    check_periods_per_year,
)
from quantmgmt.risk.returns import annualized_return
from quantmgmt.risk.dispersion import (
    annualized_volatility,
    downside_deviation,
)


def sharpe_ratio(
    returns: PandasData,
    periods_per_year: float = 252,
    risk_free_rate: float = 0.0,
) -> float | pd.Series:
    """Calcula el ratio de Sharpe anualizado.

    risk_free_rate es una tasa constante por período, en formato decimal.
    Devuelve NaN cuando la volatilidad es cero o no puede estimarse.
    """
    check_pandas(returns)
    check_periods_per_year(periods_per_year)
    check_finite(risk_free_rate, "risk_free_rate")

    # Restamos la rentabilidad libre de riesgo de cada período.
    excess_returns = returns - risk_free_rate

    # Calculamos la media y la volatilidad anualizadas de los excesos.
    mean_excess = annualized_return(excess_returns, periods_per_year)
    volatility = annualized_volatility(excess_returns, periods_per_year)

    # Evitamos dividir entre cero cuando recibimos varios activos.
    if isinstance(volatility, pd.Series):
        return mean_excess / volatility.replace(0.0, np.nan)

    # Aplicamos la misma convención cuando recibimos un solo activo.
    if volatility == 0 or pd.isna(volatility):
        return np.nan

    return float(mean_excess / volatility)


def sortino_ratio(
    returns: PandasData,
    periods_per_year: float = 252,
    target: float = 0.0,
) -> float | pd.Series:
    """Calcula el ratio de Sortino anualizado.

    target es la rentabilidad objetivo por período, en formato decimal.
    Devuelve NaN cuando la desviación a la baja es cero o no puede calcularse.
    """
    check_pandas(returns)
    check_periods_per_year(periods_per_year)
    check_finite(target, "target")

    # Calculamos la rentabilidad media anualizada por encima del objetivo.
    excess_returns = returns - target
    mean_excess = annualized_return(excess_returns, periods_per_year)

    # Medimos las desviaciones desfavorables respecto al mismo objetivo.
    downside = downside_deviation(
        returns,
        periods_per_year=periods_per_year,
        target=target,
    )

    # Evitamos dividir entre cero cuando recibimos varios activos.
    if isinstance(downside, pd.Series):
        return mean_excess / downside.replace(0.0, np.nan)

    # Aplicamos la misma convención cuando recibimos un solo activo.
    if downside == 0 or pd.isna(downside):
        return np.nan

    return float(mean_excess / downside)


def tracking_error(
    returns: PandasData,
    benchmark: pd.Series,
    periods_per_year: float = 252,
) -> float | pd.Series:
    """Calcula el tracking error anualizado frente a un benchmark común.

    Ambos contienen retornos por período en formato decimal.
    Utiliza fechas comunes y omite pares con valores ausentes.
    """
    check_pandas(returns)
    check_pandas(benchmark, "benchmark")
    check_periods_per_year(periods_per_year)

    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark debe ser una pd.Series.")

    if not returns.index.is_unique or not benchmark.index.is_unique:
        raise ValueError("Los índices no deben contener fechas duplicadas.")

    # Alineamos las carteras y el benchmark por sus fechas comunes.
    aligned_returns, aligned_benchmark = returns.align(
        benchmark,
        join="inner",
        axis=0,
    )

    # Restamos el benchmark a cada cartera, haciendo coincidir las fechas.
    active_returns = aligned_returns.sub(aligned_benchmark, axis=0)
    check_pandas(active_returns, "active_returns")

    # Anualizamos la volatilidad de las diferencias de rentabilidad.
    return annualized_volatility(active_returns, periods_per_year)



def calmar_ratio(
    returns: PandasData,
    periods_per_year: float = 252,
) -> float | pd.Series:
    """Calcula el ratio de Calmar sobre todo el histórico recibido.

    Divide la rentabilidad anual compuesta entre la máxima caída.
    Devuelve NaN cuando la caída es cero o no puede calcularse.
    """
    check_pandas(returns)
    check_periods_per_year(periods_per_year)

    # Calculamos la rentabilidad anual compuesta del histórico.
    annual_growth = cagr(returns, periods_per_year)

    # La función del módulo drawdown devuelve la caída en positivo.
    maximum_drawdown = max_drawdown(returns)

    # Evitamos dividir entre cero cuando recibimos varios activos.
    if isinstance(maximum_drawdown, pd.Series):
        return annual_growth / maximum_drawdown.replace(0.0, np.nan)

    # Aplicamos la misma convención cuando recibimos un solo activo.
    if maximum_drawdown == 0 or pd.isna(maximum_drawdown):
        return np.nan

    return float(annual_growth / maximum_drawdown)


