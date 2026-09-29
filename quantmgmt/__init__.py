"""Medidas de riesgo para la práctica 1.

Convenciones comunes:
- Retornos simples en decimal, con fechas en filas y activos en columnas.
- periods_per_year indica la frecuencia: 252 diaria, 52 semanal y 12 mensual.
- Los valores ausentes se omiten; rolling exige ventanas completas.
- Las caídas se expresan en positivo: 0.25 representa una caída del 25 %.
- Las tasas de referencia y los objetivos se indican por período.
"""

from quantmgmt.risk.returns import (
    annualized_return,
    cagr,
    cumulative_returns,
    to_returns,
)
from quantmgmt.risk.dispersion import (
    annualized_volatility,
    downside_deviation,
    rolling_volatility,
)
from quantmgmt.risk.drawdown import (
    drawdown_series,
    max_drawdown,
    recovery_time,
    time_under_water,
)
from quantmgmt.risk.ratios import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)


# Funciones públicas del módulo de riesgo.
__all__ = [
    "annualized_return",
    "cagr",
    "cumulative_returns",
    "to_returns",
    "annualized_volatility",
    "downside_deviation",
    "rolling_volatility",
    "drawdown_series",
    "max_drawdown",
    "recovery_time",
    "time_under_water",
    "calmar_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
]