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


def posterior(
    pi: pd.Series,
    sigma: pd.DataFrame,
    P: pd.DataFrame,
    Q: pd.Series,
    omega: pd.DataFrame,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Rentabilidad esperada y covarianza posteriores de Black-Litterman.

    μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]
    Σ_BL = Σ + M, con M = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ la incertidumbre de μ_BL.
    """
    activos = sigma.index
    S = sigma.to_numpy()
    p = P.reindex(columns=activos).fillna(0.0).to_numpy()
    q = Q.reindex(P.index).to_numpy()
    om = omega.reindex(index=P.index, columns=P.index).to_numpy()
    prior = np.linalg.inv(tau * S)
    m = np.linalg.inv(prior + p.T @ np.linalg.inv(om) @ p)
    mu = m @ (prior @ pi.reindex(activos).to_numpy() + p.T @ np.linalg.inv(om) @ q)
    mu_bl = pd.Series(mu, index=activos, name="mu_bl")
    sigma_bl = pd.DataFrame(S + m, index=activos, columns=activos)
    return mu_bl, sigma_bl


def _align(w_mkt: pd.Series, sigma: pd.DataFrame) -> np.ndarray:
    """Pesos en el mismo orden que Σ. Falla si falta algún activo."""
    w = w_mkt.reindex(sigma.index)
    if w.isna().any():
        raise ValueError(f"w_mkt no tiene peso para: {list(w[w.isna()].index)}")
    return w.to_numpy()
