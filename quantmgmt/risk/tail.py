"""Métricas de riesgo de cola y forma de la distribución."""

import pandas as pd
from statistics import NormalDist

from quantmgmt.risk._checks import (
    PandasData,
    check_confidence,
    check_pandas,
)


def skewness(returns: PandasData) -> float | pd.Series:
    """Calcula la asimetría muestral por activo, omitiendo valores ausentes."""
    check_pandas(returns)
    return returns.skew(axis=0)


def kurtosis(returns: PandasData) -> float | pd.Series:
    """Calcula el exceso de curtosis muestral: la normal tiene referencia cero."""
    check_pandas(returns)
    return returns.kurt(axis=0)


def var_historical(
    returns: PandasData,
    confidence: float = 0.95,
) -> float | pd.Series:
    """Calcula el VaR histórico por período mediante interpolación lineal.

    Omite valores ausentes y devuelve un resultado por activo.
    Expresa pérdidas en positivo; puede ser negativo si el cuantil es positivo.
    """
    check_pandas(returns)
    check_confidence(confidence)

    # Obtenemos el cuantil de la cola inferior de los retornos.
    tail_quantile = returns.quantile(
        1.0 - confidence,
        interpolation="linear",
    )

    # Cambiamos el signo para expresar el resultado como pérdida.
    return -tail_quantile


def cvar_historical(
    returns: PandasData,
    confidence: float = 0.95,
) -> float | pd.Series:
    """Calcula la pérdida media de la cola histórica por período.

    Incluye los retornos menores o iguales al cuantil, incluidos los empates.
    Omite valores ausentes y devuelve un resultado por activo.
    """
    # Obtenemos el umbral de retorno; var_historical valida las entradas.
    threshold = -var_historical(returns, confidence)

    # Conservamos los retornos de la cola y marcamos los demás como ausentes.
    tail_returns = returns.where(returns <= threshold)

    # Promediamos únicamente la cola y expresamos el resultado como pérdida.
    return -tail_returns.mean(axis=0)


def var_parametric(
    returns: PandasData,
    confidence: float = 0.95,
    method: str = "gaussian",
) -> float | pd.Series:
    """Calcula el VaR por período con normal o aproximación Cornish-Fisher.

    method admite "gaussian" y "cornish_fisher".
    Omite valores ausentes y expresa las pérdidas en positivo.
    Cornish-Fisher devuelve NaN si faltan datos para estimar los momentos.
    """
    check_pandas(returns)
    check_confidence(confidence)

    if method not in ("gaussian", "cornish_fisher"):
        raise ValueError("method debe ser 'gaussian' o 'cornish_fisher'.")

    # Estimamos la media y la desviación típica por activo.
    mean_return = returns.mean(axis=0)
    volatility = returns.std(axis=0, ddof=1)

    # Partimos del cuantil inferior de la normal estándar.
    z = NormalDist().inv_cdf(1.0 - confidence)

    if method == "cornish_fisher":
        asymmetry = skewness(returns)
        excess_kurtosis = kurtosis(returns)

        # Ajustamos el cuantil por asimetría y exceso de curtosis.
        z = (
            z
            + (z**2 - 1.0) * asymmetry / 6.0
            + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
            - (2.0 * z**3 - 5.0 * z) * asymmetry**2 / 36.0
        )

    # Convertimos el cuantil de retorno en un umbral de pérdida.
    return -(mean_return + volatility * z)


def cvar_parametric(
    returns: PandasData,
    confidence: float = 0.95,
) -> float | pd.Series:
    """Calcula el CVaR gaussiano por período.

    Estima la pérdida media de la cola inferior de una distribución normal.
    Omite valores ausentes y expresa las pérdidas en positivo.
    """
    check_pandas(returns)
    check_confidence(confidence)

    # Estimamos la media y la desviación típica por activo.
    mean_return = returns.mean(axis=0)
    volatility = returns.std(axis=0, ddof=1)

    # Calculamos el cuantil normal que delimita la cola inferior.
    normal = NormalDist()
    tail_probability = 1.0 - confidence
    z = normal.inv_cdf(tail_probability)

    # Obtenemos la pérdida media de la cola bajo el modelo normal.
    return -mean_return + volatility * normal.pdf(z) / tail_probability