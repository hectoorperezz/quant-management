"""Métricas de caídas desde máximos (drawdown).

Las caídas se expresan en positivo, como en las diapositivas de la sesión 2:
0.35 significa que el valor está un 35 % por debajo de su máximo anterior.
El capital inicial cuenta como primer máximo.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantmgmt.risk._checks import PandasData, check_pandas

# Margen para no confundir el ruido de coma flotante con una caída real.
_TOLERANCE = 1e-12


def drawdown_series(returns: PandasData) -> PandasData:
    """Calcula la caída desde el máximo anterior en cada fecha.

    D_t = 1 - V_t / max(V_0, ..., V_t), donde V es el valor de 1 € invertido
    al inicio (V_0 = 1). Conserva el índice y las columnas de ``returns``.
    """
    check_pandas(returns)
    return _drawdown(returns)


def max_drawdown(returns: PandasData) -> float | pd.Series:
    """Devuelve la mayor caída desde un máximo (0.35 = un 35 %)."""
    check_pandas(returns)
    return _drawdown(returns).max()


def time_under_water(returns: PandasData) -> float | pd.Series:
    """Cuenta el tramo más largo de periodos seguidos por debajo del máximo.

    Se mide en observaciones (días si los datos son diarios) e incluye el
    tramo que siga abierto al final de la serie.
    """
    check_pandas(returns)
    return _by_column(returns, _longest_under_water)


def recovery_time(returns: PandasData) -> float | pd.Series:
    """Cuenta los periodos que tarda en recuperarse de la peor caída.

    Se mide en observaciones, desde el fondo del max drawdown hasta volver al
    máximo anterior. Devuelve 0 si nunca ha caído y NaN si aún no se ha
    recuperado.
    """
    check_pandas(returns)
    return _by_column(returns, _periods_to_recover)


def _drawdown(returns: PandasData) -> PandasData:
    """Calcula la serie de caídas sin repetir las comprobaciones."""
    # Valor de 1 € invertido al inicio.
    wealth = (1 + returns).cumprod()

    # Máximo alcanzado hasta cada fecha, contando el capital inicial.
    peak = wealth.cummax().clip(lower=1.0)

    return 1 - wealth / peak


def _by_column(
    returns: PandasData,
    metric: Callable[[pd.Series], float],
) -> float | pd.Series:
    """Aplica una métrica de Serie a cada columna si ``returns`` es un DataFrame."""
    if isinstance(returns, pd.DataFrame):
        return returns.apply(metric)
    return metric(returns)


def _longest_under_water(returns: pd.Series) -> float:
    """Calcula time_under_water para una sola Serie."""
    drawdown = _drawdown(returns.dropna())
    if drawdown.empty:
        return np.nan

    underwater = drawdown > _TOLERANCE

    # Los periodos seguidos bajo el agua comparten el mismo número de tramo.
    spell = (~underwater).cumsum()
    return float(underwater.groupby(spell).sum().max())


def _periods_to_recover(returns: pd.Series) -> float:
    """Calcula recovery_time para una sola Serie."""
    drawdown = _drawdown(returns.dropna()).to_numpy()
    if drawdown.size == 0:
        return np.nan

    # Posición del fondo de la peor caída.
    trough = int(drawdown.argmax())
    if drawdown[trough] <= _TOLERANCE:
        return 0.0

    # Primer periodo, contado desde el fondo, sin caída.
    recovered = np.flatnonzero(drawdown[trough:] <= _TOLERANCE)
    if recovered.size == 0:
        return np.nan
    return float(recovered[0])
