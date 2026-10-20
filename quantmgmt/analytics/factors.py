"""Regresiones multifactoriales: CAPM, Fama-French y Carhart.

Explican el exceso de rentabilidad de una cartera sobre el tipo libre de
riesgo con primas de riesgo conocidas (mercado, tamaño, valor, momentum,
rentabilidad e inversión). La constante de la regresión es el alfa: la
parte de la rentabilidad que los factores no explican.

Los factores son los de la Ken French Data Library
(``quantmgmt.data.factors.download_ff_factors``): mensuales, en decimal y
ya en exceso sobre ``RF``.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from quantmgmt.risk._checks import (
    PandasData,
    check_pandas,
    check_periods_per_year,
    check_window,
)

# Factores de cada especificación, con los nombres de columna de Ken French.
MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart4": ["Mkt-RF", "SMB", "HML", "Mom"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


def run_factor_regression(
    returns: PandasData,
    factors: pd.DataFrame,
    model: str = "FF3",
    periods_per_year: float = 12,
    newey_west: bool = True,
) -> pd.Series | pd.DataFrame:
    """Estima por MCO el modelo de factores sobre el exceso de rentabilidad.

    r_t - RF_t = alfa + sum_j beta_j * F_j,t + e_t

    Parámetros
    ----------
    returns : pd.Series o pd.DataFrame
        Retornos simples en decimal (no en exceso), una columna por cartera.
    factors : pd.DataFrame
        Factores en decimal con la columna ``RF`` y las del modelo.
    model : {"CAPM", "FF3", "Carhart4", "FF5"}
        Especificación del modelo.
    periods_per_year : float
        Periodos por año para anualizar el alfa: 12 con datos mensuales.
    newey_west : bool
        Si es True, los t-estadísticos usan errores de Newey-West, robustos a
        heterocedasticidad y autocorrelación (como en el notebook de Buffett).
        Los coeficientes no cambian.

    Devuelve
    --------
    pd.Series o pd.DataFrame
        ``alpha`` (anualizada), ``t_alpha``, ``beta_<factor>`` y ``t_<factor>``
        de cada factor, ``r2_adj`` y ``n_obs``. Un DataFrame con una fila por
        cartera si ``returns`` es un DataFrame.
    """
    check_pandas(returns)
    check_periods_per_year(periods_per_year)
    names = _model_factors(model, factors)

    if isinstance(returns, pd.DataFrame):
        rows = {
            column: _regression(returns[column], factors, names, periods_per_year, newey_west)
            for column in returns.columns
        }
        return pd.DataFrame(rows).T
    return _regression(returns, factors, names, periods_per_year, newey_west)


def rolling_betas(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "FF3",
    window: int = 36,
    periods_per_year: float = 12,
) -> pd.DataFrame:
    """Estima alfa (anualizada) y betas en una ventana móvil de ``window`` periodos.

    Cada fila corresponde a la fecha final de su ventana. Las primeras fechas,
    sin una ventana completa, no se incluyen.
    """
    check_pandas(returns)
    check_window(window, min_size=2)
    check_periods_per_year(periods_per_year)
    if not isinstance(returns, pd.Series):
        raise TypeError("rolling_betas recibe una sola cartera (pd.Series).")
    names = _model_factors(model, factors)

    excess, X = _excess_and_design(returns, factors, names)
    if len(excess) < window:
        raise ValueError(f"Hay {len(excess)} observaciones, menos que la ventana ({window}).")

    params = RollingOLS(excess, X, window=window).fit().params.dropna()
    params["const"] *= periods_per_year
    return params.rename(columns={"const": "alpha"})


def _model_factors(model: str, factors: pd.DataFrame) -> list[str]:
    """Devuelve los factores del modelo y comprueba que están en ``factors``."""
    if model not in MODELS:
        raise ValueError(f"model debe ser uno de {list(MODELS)}, no {model!r}")
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors debe ser un pd.DataFrame.")
    missing = [name for name in [*MODELS[model], "RF"] if name not in factors.columns]
    if missing:
        raise ValueError(f"Faltan columnas en factors: {missing}")
    return MODELS[model]


def _excess_and_design(
    returns: pd.Series,
    factors: pd.DataFrame,
    names: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Alinea por fecha, resta RF y añade la constante (el alfa)."""
    data = pd.concat([returns.rename("_r"), factors[[*names, "RF"]]], axis=1, join="inner")
    data = data.dropna()
    excess = (data["_r"] - data["RF"]).rename(returns.name)
    return excess, sm.add_constant(data[names])


def _regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    names: list[str],
    periods_per_year: float,
    newey_west: bool,
) -> pd.Series:
    """Regresión de una sola cartera."""
    excess, X = _excess_and_design(returns, factors, names)
    n_obs = len(excess)
    if n_obs <= len(names) + 1:
        raise ValueError(f"Hay {n_obs} observaciones, insuficientes para {len(names)} factores.")

    if newey_west:
        # Regla habitual de rezagos, la misma del notebook de Buffett.
        lags = int(4 * (n_obs / 100) ** (2 / 9))
        fit = sm.OLS(excess, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
        )
    else:
        fit = sm.OLS(excess, X).fit()

    result = {
        "alpha": fit.params["const"] * periods_per_year,
        "t_alpha": fit.tvalues["const"],
    }
    for name in names:
        result[f"beta_{name}"] = fit.params[name]
        result[f"t_{name}"] = fit.tvalues[name]
    result["r2_adj"] = fit.rsquared_adj
    result["n_obs"] = n_obs
    return pd.Series(result, name=returns.name)
