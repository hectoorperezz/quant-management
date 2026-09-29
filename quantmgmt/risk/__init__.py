"""Medidas de riesgo para la práctica 1.

Convenciones comunes:
- Retornos simples en decimal, con fechas en filas y activos en columnas.
- periods_per_year indica la frecuencia: 252 diaria, 52 semanal y 12 mensual.
- Los valores ausentes se omiten; rolling exige ventanas completas.
- Las caídas se expresan en positivo: 0.25 representa una caída del 25 %.
- Las tasas de referencia y los objetivos se indican por período.
- VaR y CVaR se calculan por período, sin anualización automática.
- VaR y CVaR expresan pérdidas en positivo y pueden ser negativos.
- La curtosis se expresa como exceso de curtosis, con referencia normal cero.
- La métrica propia, scare_probability, es la probabilidad de perder más de
  un límite tras entrar en una fecha cualquiera (ver quantmgmt.risk.custom).
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
from quantmgmt.risk.tail import (
    var_historical,
    cvar_historical,
    var_parametric,
    cvar_parametric,
    skewness,
    kurtosis,
)
from quantmgmt.risk.custom import scare_probability, worst_loss_ahead
from quantmgmt.risk.summary import risk_summary


# Funciones públicas del módulo de riesgo.
__all__ = [
    # Rentabilidad.
    "annualized_return",
    "cagr",
    "cumulative_returns",
    "to_returns",
    # Dispersión.
    "annualized_volatility",
    "downside_deviation",
    "rolling_volatility",
    # Caídas desde máximos.
    "drawdown_series",
    "max_drawdown",
    "recovery_time",
    "time_under_water",
    # Ratios y comparación.
    "calmar_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
    # Riesgo de cola y forma de la distribución.
    "var_historical",
    "cvar_historical",
    "var_parametric",
    "cvar_parametric",
    "skewness",
    "kurtosis",
    # Métrica propia.
    "scare_probability",
    "worst_loss_ahead",
    # Tabla resumen.
    "risk_summary",
]