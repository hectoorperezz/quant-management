"""Comprobaciones comunes a las funciones del módulo de riesgo."""

import numpy as np
import pandas as pd

PandasData = pd.Series | pd.DataFrame


def check_pandas(data: object, name: str = "returns") -> None:
    """Comprueba que ``data`` es una Serie o un DataFrame con algún dato válido."""
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"{name} debe ser pd.Series o pd.DataFrame, no {type(data).__name__}"
        )
    if not data.notna().to_numpy().any():
        raise ValueError(f"{name} no contiene datos válidos")


def check_periods_per_year(periods_per_year: float) -> None:
    """Comprueba que ``periods_per_year`` es finito y positivo."""
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year debe ser finito y positivo, no {periods_per_year}"
        )


def check_finite(value: float, name: str) -> None:
    """Comprueba que ``value`` es un número finito."""
    if not np.isfinite(value):
        raise ValueError(f"{name} debe ser finito, no {value}")


def check_window(window: int, min_size: int = 2) -> None:
    """Comprueba que ``window`` es un entero mayor o igual que ``min_size``."""
    if not isinstance(window, (int, np.integer)) or window < min_size:
        raise ValueError(
            f"window debe ser un entero mayor o igual que {min_size}, no {window}"
        )


def check_confidence(confidence: float) -> None:
    """Comprueba que el nivel de confianza esté entre cero y uno."""
    check_finite(confidence, "confidence")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence debe estar entre 0 y 1, sin incluirlos.")