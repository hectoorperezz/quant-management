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

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quantmgmt.risk._checks import (
    PandasData,
    check_finite,
    check_pandas,
    check_window,
)


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


def tail_weighted_downside_risk(
    returns: PandasData,
    target: float = 0.0,
    tail_fraction: float = 0.05,
    tail_weight: float = 0.5,
) -> float | pd.Series:
    """Calcula el riesgo de incumplimiento ponderado por la cola.

    Para cada período, el incumplimiento es d_t = min(r_t - target, 0).
    Combinamos la media de d_t² en todos los períodos con la media de d_t²
    en la cola de los k peores, con k = ceil(tail_fraction × n):

        riesgo = sqrt((1 - tail_weight) × media(d²) + tail_weight × media_cola(d²))

    Con tail_weight = 0 coincide con la desviación a la baja sin anualizar.

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos en decimal, con fechas en filas y activos en columnas.
    target : float
        Rentabilidad mínima por período; por debajo hay incumplimiento.
    tail_fraction : float
        Proporción de períodos que forman la cola, entre 0 (excluido) y 1.
    tail_weight : float
        Peso de la cola en la combinación, entre 0 y 1.

    Devuelve
    --------
    float o pd.Series
        Riesgo por período, sin anualizar, en las unidades de los retornos.
        Cero si nunca se incumple; NaN si una columna no tiene datos válidos.
    """
    check_pandas(returns)
    check_finite(target, "target")
    check_finite(tail_fraction, "tail_fraction")
    check_finite(tail_weight, "tail_weight")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError(
            f"tail_fraction debe estar en (0, 1], no {tail_fraction}"
        )
    if not 0.0 <= tail_weight <= 1.0:
        raise ValueError(f"tail_weight debe estar en [0, 1], no {tail_weight}")
    if np.isinf(returns.to_numpy()).any():
        raise ValueError("returns no puede contener valores infinitos.")

    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            lambda column: _tail_weighted_downside_risk(
                column, target, tail_fraction, tail_weight
            )
        )
    return _tail_weighted_downside_risk(returns, target, tail_fraction, tail_weight)


def _tail_weighted_downside_risk(
    returns: pd.Series,
    target: float,
    tail_fraction: float,
    tail_weight: float,
) -> float:
    """Calcula tail_weighted_downside_risk para una sola Serie."""
    returns = returns.dropna()
    if returns.empty:
        return float("nan")

    squared_shortfalls = np.minimum(returns.to_numpy() - target, 0.0) ** 2

    # Los peores incumplimientos son los de mayor cuadrado.
    # Redondeamos antes del techo: 0.07 × 100 da 7.000000000000001 en coma flotante.
    n_tail = math.ceil(round(tail_fraction * len(squared_shortfalls), 9))
    tail = np.sort(squared_shortfalls)[-n_tail:]

    combined = (1 - tail_weight) * squared_shortfalls.mean() + tail_weight * tail.mean()
    return float(np.sqrt(combined))


def tail_weighted_downside_ratio(
    returns: PandasData,
    benchmark: float | pd.Series = 0.0,
    tail_fraction: float = 0.05,
    tail_weight: float = 0.5,
) -> float | pd.Series:
    """Calcula la rentabilidad relativa por unidad de riesgo de incumplimiento.

    Filosofía
    ---------
    Nuestra idea de riesgo se centra en los rendimientos que no superan a un determinado
    target, por ejemplo el tipo libre de riesgo:
    superar a esa referencia no debe penalizar a la estrategia. Nos preocupa
    con qué frecuencia quedamos por debajo y cuánto perdemos respecto a ella.

    Para recoger ambos aspectos, los períodos que superan la referencia
    cuentan como cero incumplimiento y los que quedan por debajo aportan
    su diferencia al cuadrado. Así, los incumplimientos grandes pesan
    proporcionalmente más que los pequeños.

    Además, damos importancia adicional a los peores incumplimientos.
    Combinamos la media cuadrática de todos los períodos con la media
    cuadrática de una cola de los peores casos y aplicamos la raíz.
    La cola ya está incluida en el conjunto general: su peso adicional
    expresa nuestra preocupación por las pérdidas relativas más graves.

    Finalmente, dividimos el retorno medio sobre la referencia entre ese
    riesgo. Buscamos estrategias que generen más rentabilidad adicional
    por cada unidad de esta nueva medida de riesgo, considerando todos los retornos en
    el numerador, tanto favorables como desfavorables.

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos en decimal, con fechas en filas y activos en columnas.
    benchmark : float o pd.Series
        Referencia con la misma frecuencia y moneda que los retornos.
        Cero evalúa pérdidas nominales; el libre de riesgo, incumplimientos
        frente a esa alternativa; SPY, resultados inferiores al mercado.
        Con SPY podemos penalizar un día positivo si el mercado ganó más.
    tail_fraction : float
        Proporción de períodos que seleccionamos como los peores casos.
        Por ejemplo, 0.05 selecciona los 5 peores de cada 100 observaciones.
        El número seleccionado se redondea hacia arriba.
    tail_weight : float
        Importancia de la cola en la combinación.
        Con 0.5, el componente general y el de cola pesan lo mismo.
        Con 0, solo cuenta el general; con 1, solo cuenta la cola.
        Tanto este peso como el tamaño de cola son decisiones de diseño.


    Devuelve
    --------
    float o pd.Series
        Ratio por período, sin anualizar. NaN si el riesgo es cero
        o una columna no tiene observaciones válidas compartidas.
    """
    check_pandas(returns)

    if np.isinf(returns.to_numpy()).any():
        raise ValueError("returns no puede contener valores infinitos.")

    if isinstance(benchmark, pd.Series):
        check_pandas(benchmark, "benchmark")

        if np.isinf(benchmark.to_numpy()).any():
            raise ValueError("benchmark no puede contener valores infinitos.")

        if not returns.index.is_unique or not benchmark.index.is_unique:
            raise ValueError("Los índices no pueden contener fechas duplicadas.")

        # Compara únicamente fechas presentes en ambas series.
        aligned_returns, aligned_benchmark = returns.align(
            benchmark,
            join="inner",
            axis=0,
        )

        # Resta la referencia de cada fecha a todos los activos.
        excess_returns = aligned_returns.sub(aligned_benchmark, axis=0)

    else:
        check_finite(benchmark, "benchmark")
        excess_returns = returns - benchmark

    # Exige al menos una comparación válida con la referencia.
    check_pandas(excess_returns, "excess_returns")

    # Conserva excesos positivos y negativos al calcular la rentabilidad.
    mean_excess_return = excess_returns.mean()

    # La referencia ya está descontada, por lo que el objetivo es cero.
    downside_risk = tail_weighted_downside_risk(
        excess_returns,
        target=0.0,
        tail_fraction=tail_fraction,
        tail_weight=tail_weight,
    )

    # Evita interpretar la ausencia de incumplimientos como un ratio infinito.
    if isinstance(downside_risk, pd.Series):
        return mean_excess_return / downside_risk.where(downside_risk > 0)

    if downside_risk > 0:
        return float(mean_excess_return / downside_risk)

    return float("nan")