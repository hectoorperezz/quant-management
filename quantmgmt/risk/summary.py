"""Tabla resumen con todas las métricas de riesgo para comparar carteras."""

import pandas as pd

from quantmgmt.risk._checks import PandasData, check_pandas
from quantmgmt.risk.custom import scare_probability
from quantmgmt.risk.dispersion import annualized_volatility, downside_deviation
from quantmgmt.risk.drawdown import max_drawdown, recovery_time, time_under_water
from quantmgmt.risk.ratios import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)
from quantmgmt.risk.returns import annualized_return, cagr
from quantmgmt.risk.tail import (
    cvar_historical,
    cvar_parametric,
    kurtosis,
    skewness,
    var_historical,
    var_parametric,
)


def risk_summary(
    returns: PandasData,
    benchmark: pd.Series | None = None,
    periods_per_year: float = 252,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calcula todas las métricas de riesgo en una tabla, una fila por cartera.

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos simples en decimal, una columna por activo o cartera.
    benchmark : pd.Series, opcional
        Retornos del índice de referencia. Si se indica, añade el tracking
        error de cada cartera frente a él.
    periods_per_year : float
        Periodos por año: 252 si son diarios.
    risk_free_rate : float
        Tipo libre de riesgo por periodo, usado en Sharpe y Sortino.
    confidence : float
        Nivel de confianza del VaR y el CVaR.

    Devuelve
    --------
    pd.DataFrame
        Una fila por cartera y una columna por métrica. Las pérdidas (caídas,
        VaR y CVaR) van en positivo y los tiempos, en periodos.
    """
    check_pandas(returns)
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    level = f"{confidence:.0%}"
    metrics = {
        "Rentabilidad anualizada": annualized_return(returns, periods_per_year),
        "CAGR": cagr(returns, periods_per_year),
        "Volatilidad anualizada": annualized_volatility(returns, periods_per_year),
        "Desviación a la baja": downside_deviation(
            returns, periods_per_year, target=risk_free_rate
        ),
        "Máx. drawdown": max_drawdown(returns),
        "Tiempo bajo el agua": time_under_water(returns),
        "Tiempo de recuperación": recovery_time(returns),
        "Sharpe": sharpe_ratio(returns, periods_per_year, risk_free_rate),
        "Sortino": sortino_ratio(returns, periods_per_year, target=risk_free_rate),
        "Calmar": calmar_ratio(returns, periods_per_year),
        f"VaR histórico {level}": var_historical(returns, confidence),
        f"VaR gaussiano {level}": var_parametric(returns, confidence),
        f"VaR Cornish-Fisher {level}": var_parametric(
            returns, confidence, method="cornish_fisher"
        ),
        f"CVaR histórico {level}": cvar_historical(returns, confidence),
        f"CVaR gaussiano {level}": cvar_parametric(returns, confidence),
        "Asimetría": skewness(returns),
        "Curtosis (exceso)": kurtosis(returns),
        "Probabilidad de susto": scare_probability(returns),
    }
    if benchmark is not None:
        metrics["Tracking error"] = tracking_error(
            returns, benchmark, periods_per_year
        )

    return pd.DataFrame(metrics)
