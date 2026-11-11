"""Backtest fuera de muestra con rebalanceo periódico y ventana móvil de estimación."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, years: int = 3, freq: str = "QE") -> pd.DatetimeIndex:
    """Último día con datos de cada periodo que ya tiene `years` años completos de historia detrás."""
    fechas = pd.DatetimeIndex(index.to_series().resample(freq).last().dropna())
    inicio_ventana = fechas - pd.DateOffset(years=years) + pd.Timedelta(days=1)
    return fechas[inicio_ventana >= index[0]]


def run_backtest(
    returns: pd.DataFrame,
    strategy: Callable[[pd.DataFrame], pd.Series],
    years: int = 3,
    freq: str = "QE",
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Evalúa una estrategia de pesos fuera de muestra.

    En cada rebalanceo, strategy recibe solo los retornos de los `years` años
    anteriores y devuelve los pesos objetivo. Se invierte al cierre de ese día
    y los pesos derivan con los precios hasta el siguiente rebalanceo.

    Devuelve los retornos diarios de la cartera, los pesos objetivo de cada
    rebalanceo y su rotación, ½ Σ |w_objetivo − w_antes|. El primer
    rebalanceo no tiene rotación porque parte de liquidez.
    """
    fechas = rebalance_dates(returns.index, years, freq)
    fechas = fechas[fechas < returns.index[-1]]   # el último día ya no tiene retornos después
    if fechas.empty:
        raise ValueError(f"No hay ninguna fecha con {years} años de historia y retornos después")

    retornos, pesos, rotacion = [], {}, {}
    w_antes = None
    for i, fecha in enumerate(fechas):
        inicio = fecha - pd.DateOffset(years=years) + pd.Timedelta(days=1)
        w = strategy(returns.loc[inicio:fecha]).reindex(returns.columns).fillna(0.0)
        if not np.isclose(w.sum(), 1, atol=1e-6):
            raise ValueError(f"Los pesos del {fecha.date()} suman {w.sum():.4f} y deben sumar 1")

        # Del día siguiente al rebalanceo hasta el siguiente, ambos incluidos
        fin = fechas[i + 1] if i + 1 < len(fechas) else returns.index[-1]
        periodo = returns.loc[fecha:fin].iloc[1:]
        valor = (1 + periodo).cumprod() * w            # valor de cada posición, partiendo de 1
        total = valor.sum(axis=1)
        retornos.append(total / total.shift(fill_value=1.0) - 1)

        rotacion[fecha] = np.nan if w_antes is None else 0.5 * (w - w_antes).abs().sum()
        w_antes = valor.iloc[-1] / total.iloc[-1]       # pesos que han dejado los precios
        pesos[fecha] = w

    return (
        pd.concat(retornos).rename("cartera"),
        pd.DataFrame(pesos).T,
        pd.Series(rotacion, name="rotacion"),
    )


def effective_assets(weights: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Número efectivo de activos, 1 / Σ w²: N con pesos iguales y 1 con todo en un activo."""
    if isinstance(weights, pd.Series):
        return float(1 / (weights**2).sum())
    return 1 / (weights**2).sum(axis=1)
