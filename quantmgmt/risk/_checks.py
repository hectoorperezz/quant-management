"""Comprobaciones comunes a las funciones del módulo de riesgo."""

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
    """Comprueba que ``periods_per_year`` es positivo."""
    if periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year debe ser positivo, no {periods_per_year}"
        )
