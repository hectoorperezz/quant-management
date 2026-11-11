"""Hierarchical Risk Parity (López de Prado, 2016) y contribución al riesgo."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Distancia entre activos a partir de su correlación: d = √((1 − ρ) / 2)."""
    # clip evita raíces de negativos minúsculos en la diagonal por redondeo
    return np.sqrt(((1 - corr) / 2).clip(lower=0))


def cluster(cov: pd.DataFrame, method: str = "single") -> np.ndarray:
    """Árbol jerárquico de los activos a partir de Σ (matriz de enlace de scipy).

    Cada fila es un paso: [elemento a, elemento b, distancia, nº de activos].
    Los índices menores que el nº de activos son activos; los demás, grupos ya formados.
    """
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)
    dist = correlation_distance(corr)
    # squareform pasa la matriz cuadrada al vector del triángulo superior que espera linkage
    return linkage(squareform(dist.to_numpy(), checks=False), method=method)


def quasi_diagonal(link: np.ndarray, assets: list[str]) -> list[str]:
    """Orden de los activos según las hojas del árbol: los parecidos quedan juntos."""
    return [assets[i] for i in leaves_list(link)]


def _cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
    """Varianza de un grupo con sus pesos repartidos por la inversa de la varianza."""
    sub = cov.loc[assets, assets].to_numpy()
    w = 1 / np.diag(sub)
    w /= w.sum()
    return float(w @ sub @ w)


def recursive_bisection(cov: pd.DataFrame, order: list[str]) -> pd.Series:
    """Pesos HRP: se parte la lista ordenada en mitades y se reparte según su varianza."""
    w = pd.Series(1.0, index=order)
    grupos = [order]
    while grupos:
        # Cada grupo con más de un activo se parte en dos mitades por posición
        grupos = [g[i:j] for g in grupos if len(g) > 1
                  for i, j in ((0, len(g) // 2), (len(g) // 2, len(g)))]
        for izq, der in zip(grupos[::2], grupos[1::2]):
            v_izq, v_der = _cluster_variance(cov, izq), _cluster_variance(cov, der)
            alpha = 1 - v_izq / (v_izq + v_der)
            w[izq] *= alpha
            w[der] *= 1 - alpha
    return w.reindex(cov.index)


def hrp_weights(cov: pd.DataFrame, method: str = "single") -> pd.Series:
    """Cartera HRP: clustering, cuasi-diagonalización y bisección recursiva."""
    order = quasi_diagonal(cluster(cov, method), list(cov.index))
    return recursive_bisection(cov, order).rename("hrp")


def inverse_volatility(cov: pd.DataFrame) -> pd.Series:
    """Cartera naive: pesos proporcionales a la inversa de la volatilidad."""
    inv = 1 / np.sqrt(np.diag(cov))
    return pd.Series(inv / inv.sum(), index=cov.index, name="inv_vol")


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Contribución de cada activo a la volatilidad de la cartera. Suman σ_p."""
    w = weights.reindex(cov.index).to_numpy()
    marginal = cov.to_numpy() @ w
    vol = np.sqrt(w @ marginal)
    return pd.Series(w * marginal / vol, index=cov.index, name="rc")
