"""Métricas de rentabilidad: retornos, retorno acumulado, CAGR y retorno anualizado."""

from typing import Literal

import numpy as np
import pandas as pd

from quantmgmt.risk._checks import PandasData, check_pandas, check_periods_per_year


def to_returns(
    prices: PandasData, method: Literal["simple", "log"] = "simple"
) -> PandasData:
    """Convierte precios (o NAV) en retornos periódicos.

    - ``"simple"``: r_t = P_t / P_{t-1} - 1
    - ``"log"``: r_t = ln(P_t / P_{t-1})

    Parámetros
    ----------
    prices : pd.Series o pd.DataFrame
        Precios positivos indexados por fecha, un activo por columna.
    method : {"simple", "log"}
        Tipo de retorno. El resto del módulo trabaja con retornos simples.

    Devuelve
    --------
    pd.Series o pd.DataFrame
        Retornos en decimal (0.01 = 1 %). Se elimina la primera fecha, que no
        tiene precio anterior.
    """
    check_pandas(prices, "prices")
    if (prices.to_numpy() <= 0).any():
        raise ValueError("prices solo puede contener precios positivos")
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices).diff()
    else:
        raise ValueError(f"method debe ser 'simple' o 'log', no {method!r}")
    return returns.iloc[1:]


def cumulative_returns(returns: PandasData) -> PandasData:
    """Retorno acumulado en cada fecha desde el inicio.

    R_t = (1 + r_1)(1 + r_2)...(1 + r_t) - 1

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos simples en decimal.

    Devuelve
    --------
    pd.Series o pd.DataFrame
        Retorno acumulado con el mismo índice que ``returns``. Por ejemplo,
        0.25 significa que 1 € invertido al inicio vale 1,25 €.
    """
    check_pandas(returns)
    return (1 + returns).cumprod() - 1


def cagr(returns: PandasData, periods_per_year: float = 252) -> float | pd.Series:
    """Tasa de crecimiento anual compuesta (CAGR).

    Es el retorno anual constante que lleva del valor inicial al final:
    CAGR = (1 + retorno total) ** (1 / años) - 1, con años = n / periods_per_year.

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos simples en decimal.
    periods_per_year : float
        Periodos por año: 252 si son diarios, 52 semanales, 12 mensuales.

    Devuelve
    --------
    float o pd.Series
        Un número, o un valor por columna si ``returns`` es un DataFrame.
    """
    check_pandas(returns)
    check_periods_per_year(periods_per_year)
    years = returns.count() / periods_per_year
    growth = (1 + returns).prod(min_count=1)
    return growth ** (1 / years) - 1


def annualized_return(
    returns: PandasData, periods_per_year: float = 252
) -> float | pd.Series:
    """Retorno medio anualizado (aritmético): media por periodo × periods_per_year.

    Es el que se usa en ratios como el de Sharpe. No recoge el efecto de la
    capitalización compuesta: para lo que realmente ganó la inversión, usa
    :func:`cagr`.

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos simples en decimal.
    periods_per_year : float
        Periodos por año: 252 si son diarios, 52 semanales, 12 mensuales.

    Devuelve
    --------
    float o pd.Series
        Un número, o un valor por columna si ``returns`` es un DataFrame.
    """
    check_pandas(returns)
    check_periods_per_year(periods_per_year)
    return returns.mean() * periods_per_year
