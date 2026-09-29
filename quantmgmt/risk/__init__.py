"""Medidas de riesgo para la práctica 1.

Convenciones comunes a todas las funciones:

- Reciben retornos simples en decimal (0.01 = 1 %) indexados por fecha. Para
  pasar de precios o NAV a retornos, usa :func:`to_returns`.
- Una ``pd.Series`` devuelve un número. Un ``pd.DataFrame`` (un activo o una
  cartera por columna) devuelve una ``pd.Series`` con un valor por columna,
  para poder comparar carteras.
- ``periods_per_year`` indica la frecuencia de los datos en las funciones que
  anualizan: 252 para datos diarios, 52 semanales y 12 mensuales.
- Los ``NaN`` se ignoran columna a columna.
- Las caídas (drawdown) van en positivo: 0.35 significa un 35 % por debajo
  del máximo anterior.
"""

from quantmgmt.risk.drawdown import (
    drawdown_series,
    max_drawdown,
    recovery_time,
    time_under_water,
)
from quantmgmt.risk.returns import (
    annualized_return,
    cagr,
    cumulative_returns,
    to_returns,
)

__all__ = [
    "annualized_return",
    "cagr",
    "cumulative_returns",
    "drawdown_series",
    "max_drawdown",
    "recovery_time",
    "time_under_water",
    "to_returns",
]
