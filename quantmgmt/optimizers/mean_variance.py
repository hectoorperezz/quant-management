"""Optimización media-varianza (Markowitz) con restricciones realistas."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantmgmt.risk._checks import check_periods_per_year


def estimate_inputs(
    returns: pd.DataFrame, periods_per_year: float = 12
) -> tuple[pd.Series, pd.DataFrame]:
    """Media y covarianza muestrales, anualizadas.

    Exige meses comunes: si hay huecos, falla en vez de estimar una
    covarianza incoherente.
    """
    check_periods_per_year(periods_per_year)
    if returns.isna().any().any():
        raise ValueError("returns tiene huecos: usa una muestra con meses comunes")
    mu = returns.mean() * periods_per_year
    sigma = returns.cov() * periods_per_year
    return mu, sigma


def portfolio_stats(w: np.ndarray, mu: pd.Series, sigma: pd.DataFrame, rf: float = 0.0) -> dict:
    """Rentabilidad, volatilidad y Sharpe esperados de una cartera."""
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(w @ sigma.to_numpy() @ w))
    return {"ret": ret, "vol": vol, "sharpe": (ret - rf) / vol}


def _constraints(
    n: int,
    long_only: bool,
    max_weight: float | None,
    prev_weights: Sequence[float] | None,
    max_turnover: float | None,
) -> tuple[list, list]:
    """Límites por activo y restricciones comunes a todos los optimizadores."""
    low = 0.0 if long_only else -1.0
    high = max_weight if max_weight is not None else 1.0
    bounds = [(low, high)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]          # suma 1
    if max_turnover is not None:
        if prev_weights is None:
            raise ValueError("max_turnover necesita prev_weights")
        prev = np.asarray(prev_weights, dtype=float)
        cons.append({"type": "ineq",                                   # >= 0
                     "fun": lambda w: max_turnover - np.abs(w - prev).sum()})
    return bounds, cons


def _solve(objective, n, bounds, cons) -> np.ndarray:
    w0 = np.full(n, 1.0 / n)                     # arranca en equiponderada
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"SLSQP no ha convergido: {res.message}")
    return res.x


def min_variance(
    sigma: pd.DataFrame,
    long_only: bool = True,
    max_weight: float | None = None,
    prev_weights: Sequence[float] | None = None,
    max_turnover: float | None = None,
) -> pd.Series:
    """Cartera de mínima varianza: minimiza w' Σ w."""
    n = len(sigma)
    S = sigma.to_numpy()
    bounds, cons = _constraints(n, long_only, max_weight, prev_weights, max_turnover)
    w = _solve(lambda w: w @ S @ w, n, bounds, cons)
    return pd.Series(w, index=sigma.index, name="min_variance")



def max_sharpe(
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float | None = None,
    prev_weights: Sequence[float] | None = None,
    max_turnover: float | None = None,
) -> pd.Series:
    """Cartera tangente: maximiza (w'μ - rf) / sqrt(w'Σw)."""
    n = len(mu)
    m, S = mu.to_numpy(), sigma.to_numpy()
    bounds, cons = _constraints(n, long_only, max_weight, prev_weights, max_turnover)

    def neg_sharpe(w):
        return -(w @ m - rf) / np.sqrt(w @ S @ w)   # SLSQP minimiza: cambiamos el signo

    w = _solve(neg_sharpe, n, bounds, cons)
    return pd.Series(w, index=mu.index, name="max_sharpe")


def _max_return(mu, sigma, **kw) -> float:
    """Máxima rentabilidad alcanzable con las restricciones dadas."""
    n, m = len(mu), mu.to_numpy()
    bounds, cons = _constraints(n, **kw)
    w = _solve(lambda w: -(w @ m), n, bounds, cons)
    return float(w @ m)


def efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    n_points: int = 50,
    long_only: bool = True,
    max_weight: float | None = None,
    prev_weights: Sequence[float] | None = None,
    max_turnover: float | None = None,
) -> pd.DataFrame:
    """Frontera eficiente: para cada rentabilidad objetivo, la cartera de mínima varianza.

    Devuelve una fila por punto con ret, vol y los pesos de cada activo.
    """
    kw = dict(long_only=long_only, max_weight=max_weight,
              prev_weights=prev_weights, max_turnover=max_turnover)
    n, m, S = len(mu), mu.to_numpy(), sigma.to_numpy()
    r_min = float(min_variance(sigma, **kw) @ m)      # empieza en la mínima varianza
    r_max = _max_return(mu, sigma, **kw)              # acaba en la máxima rentabilidad posible
    filas = []
    for target in np.linspace(r_min, r_max, n_points):
        bounds, cons = _constraints(n, **kw)
        cons.append({"type": "eq", "fun": lambda w, t=target: w @ m - t})
        w = _solve(lambda w: w @ S @ w, n, bounds, cons)
        filas.append({"ret": target, "vol": float(np.sqrt(w @ S @ w)), **dict(zip(mu.index, w))})
    return pd.DataFrame(filas)


def target_volatility(
    mu: pd.Series,
    sigma: pd.DataFrame,
    target_vol: float,
    long_only: bool = True,
    max_weight: float | None = None,
    prev_weights: Sequence[float] | None = None,
    max_turnover: float | None = None,
) -> pd.Series:
    """Máxima rentabilidad con volatilidad <= target_vol."""
    n, m, S = len(mu), mu.to_numpy(), sigma.to_numpy()
    bounds, cons = _constraints(n, long_only, max_weight, prev_weights, max_turnover)
    cons.append({"type": "ineq", "fun": lambda w: target_vol - np.sqrt(w @ S @ w)})
    w = _solve(lambda w: -(w @ m), n, bounds, cons)
    return pd.Series(w, index=mu.index, name=f"vol_{target_vol:.0%}")