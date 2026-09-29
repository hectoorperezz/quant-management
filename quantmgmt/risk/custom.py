"""Métrica de riesgo propia: la probabilidad de susto.

Responde a la pregunta que se haría un inversor antes de entrar: "si invierto
un día cualquiera, ¿qué probabilidad hay de que en el año siguiente llegue a
ir perdiendo más de lo que soporto?".

A diferencia de la volatilidad o del max drawdown, que miran un único número
de toda la historia, recorre todas las fechas de entrada posibles y depende
del perfil del inversor: cuánta pérdida aguanta (``loss_limit``) y durante
cuánto tiempo mira (``horizon``). Así se pueden comparar carteras para un
cliente concreto, como el que dice "no quiero perder más de un 20 %".
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quantmgmt.risk._checks import PandasData, check_pandas, check_window


def worst_loss_ahead(returns: PandasData, horizon: int = 252) -> PandasData:
    """Calcula, para cada fecha de entrada, la peor pérdida en los periodos siguientes.

    Para quien compra al cierre de la fecha t, es la mayor caída de su
    inversión respecto al precio de compra durante los ``horizon`` periodos
    siguientes, en positivo (0.25 = llegó a perder un 25 %). Es 0 si nunca
    bajó del precio de compra. Solo incluye fechas con el horizonte completo.
    """
    check_pandas(returns)
    check_window(horizon, min_size=1)
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda column: _worst_loss_ahead(column, horizon))
    return _worst_loss_ahead(returns, horizon)


def scare_probability(
    returns: PandasData,
    loss_limit: float = 0.20,
    horizon: int = 252,
) -> float | pd.Series:
    """Calcula la probabilidad de susto.

    Es la proporción de fechas de entrada en las que la inversión llegó a
    perder más de ``loss_limit`` durante los ``horizon`` periodos siguientes.
    Por defecto: perder más de un 20 % en el año siguiente con datos diarios.

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos simples en decimal.
    loss_limit : float
        Pérdida máxima que soporta el inversor, en positivo (0.20 = 20 %).
    horizon : int
        Periodos que mira hacia delante: 252 es un año de datos diarios.

    Devuelve
    --------
    float o pd.Series
        Probabilidad entre 0 y 1, o un valor por columna si ``returns`` es un
        DataFrame. NaN si no hay datos para un horizonte completo.
    """
    if not 0 < loss_limit < 1:
        raise ValueError(f"loss_limit debe estar entre 0 y 1, no {loss_limit}")
    worst = worst_loss_ahead(returns, horizon)

    # Cada fecha de entrada cuenta como 1 si supera el límite y 0 si no.
    scares = (worst > loss_limit).where(worst.notna())
    return scares.mean()


def _worst_loss_ahead(returns: pd.Series, horizon: int) -> pd.Series:
    """Calcula worst_loss_ahead para una sola Serie."""
    returns = returns.dropna()

    # Valor de la inversión al cierre de cada fecha.
    wealth = (1 + returns).cumprod().to_numpy()
    n_entries = len(wealth) - horizon
    if n_entries <= 0:
        return pd.Series(dtype=float, name=returns.name)

    # Para cada entrada, el valor más bajo de los periodos siguientes.
    windows = sliding_window_view(wealth[1:], horizon)
    lowest_ahead = windows.min(axis=1)

    worst = np.maximum(0.0, 1 - lowest_ahead / wealth[:n_entries])
    return pd.Series(worst, index=returns.index[:n_entries], name=returns.name)
