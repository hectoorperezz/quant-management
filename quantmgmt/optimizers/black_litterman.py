"""Black-Litterman: equilibrio de mercado y visiones del gestor."""

import numpy as np
import pandas as pd


def risk_aversion(w_mkt: pd.Series, sigma: pd.DataFrame, sharpe: float) -> float:
    """Aversión al riesgo implícita en el Sharpe del mercado: δ = Sharpe / σ_m."""
    w = _align(w_mkt, sigma)
    vol = float(np.sqrt(w @ sigma.to_numpy() @ w))
    return sharpe / vol


def implied_returns(w_mkt: pd.Series, sigma: pd.DataFrame, delta: float) -> pd.Series:
    """Rentabilidades de equilibrio en exceso: Π = δ Σ w_mkt."""
    w = _align(w_mkt, sigma)
    return pd.Series(delta * sigma.to_numpy() @ w, index=sigma.index, name="pi")


def _align(w_mkt: pd.Series, sigma: pd.DataFrame) -> np.ndarray:
    """Pesos en el mismo orden que Σ. Falla si falta algún activo."""
    w = w_mkt.reindex(sigma.index)
    if w.isna().any():
        raise ValueError(f"w_mkt no tiene peso para: {list(w[w.isna()].index)}")
    return w.to_numpy()
