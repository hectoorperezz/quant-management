"""Métricas de dispersión de rentabilidades."""

import numpy as np
import pandas as pd

from quantmgmt.risk._checks import (
    PandasData,
    check_finite,
    check_pandas,
    check_periods_per_year,
    check_window,
)


def annualized_volatility(
    returns: PandasData,
    periods_per_year: float = 252,
) -> float | pd.Series:
    """Anualiza la volatilidad de retornos decimales, omitiendo valores ausentes."""
    check_pandas(returns)
    check_periods_per_year(periods_per_year)

    # Calculamos la desviación típica muestral por activo.
    period_volatility = returns.std(axis=0, ddof=1)

    # Anualizamos según el número de períodos por año.
    return period_volatility * np.sqrt(periods_per_year)


def downside_deviation(
    returns: PandasData,
    periods_per_year: float = 252,
    target: float = 0.0,
) -> float | pd.Series:
    """Calcula la desviación a la baja anualizada.

    target es la rentabilidad objetivo por período, en formato decimal.
    Los valores ausentes se omiten; los retornos sobre el objetivo cuentan como cero.
    """
    check_pandas(returns)
    check_periods_per_year(periods_per_year)
    check_finite(target, "target")

    # Conservamos las desviaciones negativas respecto al objetivo.
    shortfalls = (returns - target).clip(upper=0.0)

    # Promediamos los cuadrados sobre todos los períodos válidos.
    mean_squared_shortfall = shortfalls.pow(2).mean(axis=0)

    # Calculamos la raíz y anualizamos.
    return np.sqrt(mean_squared_shortfall * periods_per_year)


def rolling_volatility(
    returns: PandasData,
    window: int = 63,
    periods_per_year: float = 252,
) -> PandasData:
    """Calcula la volatilidad anualizada sobre una ventana móvil.

    window indica el número de observaciones, no días de calendario.
    Exige una ventana completa de valores válidos por activo.
    """
    check_pandas(returns)
    check_window(window)
    check_periods_per_year(periods_per_year)

    # Calculamos la desviación típica en cada ventana de observaciones.
    period_volatility = returns.rolling(
        window=window,
        min_periods=window,
    ).std(ddof=1)

    # Anualizamos conservando las fechas y columnas.
    return period_volatility * np.sqrt(periods_per_year)
