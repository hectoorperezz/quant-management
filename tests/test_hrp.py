"""Pruebas de HRP con correlaciones conocidas."""

import numpy as np
import pandas as pd

from quantmgmt.optimizers.hrp import (
    cluster,
    correlation_distance,
    hrp_weights,
    inverse_volatility,
    quasi_diagonal,
    recursive_bisection,
    risk_contribution,
)


def test_distance_limits():
    # ρ = 1 → 0, ρ = 0 → √0,5, ρ = −1 → 1
    corr = pd.DataFrame([[1.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0],
                         [-1.0, 0.0, 1.0]],
                        index=list("ABC"), columns=list("ABC"))
    d = correlation_distance(corr)
    assert np.allclose(np.diag(d), 0)
    assert np.isclose(d.loc["A", "B"], np.sqrt(0.5))
    assert np.isclose(d.loc["A", "C"], 1)


def test_identical_assets_join_first():
    # A y B son el mismo activo (ρ = 1): el primer paso los une con distancia 0
    cov = pd.DataFrame([[0.04, 0.04, 0.00],
                        [0.04, 0.04, 0.00],
                        [0.00, 0.00, 0.09]],
                       index=list("ABC"), columns=list("ABC"))
    link = cluster(cov)
    assert set(link[0, :2]) == {0, 1}
    assert link[0, 2] < 1e-6   # ρ sale 0,9999999999999998 y la raíz deja un resto de 1e-8


def test_similar_assets_end_up_together():
    # A y C son casi el mismo activo, igual que B y D: en el orden deben quedar juntos
    cov = pd.DataFrame([[0.04, 0.00, 0.039, 0.00],
                        [0.00, 0.09, 0.00, 0.088],
                        [0.039, 0.00, 0.04, 0.00],
                        [0.00, 0.088, 0.00, 0.09]],
                       index=list("ABCD"), columns=list("ABCD"))
    orden = quasi_diagonal(cluster(cov), list(cov.index))
    pos = {a: i for i, a in enumerate(orden)}
    assert abs(pos["A"] - pos["C"]) == 1
    assert abs(pos["B"] - pos["D"]) == 1


def test_uncorrelated_assets_get_inverse_variance():
    # Sin correlaciones, HRP coincide con la inversa de la varianza
    var = np.array([0.01, 0.04, 0.09, 0.16])
    cov = pd.DataFrame(np.diag(var), index=list("ABCD"), columns=list("ABCD"))
    w = recursive_bisection(cov, list("ABCD"))
    esperado = (1 / var) / (1 / var).sum()
    assert np.allclose(w.to_numpy(), esperado)


def test_weights_sum_one_and_are_positive():
    cov = pd.DataFrame([[0.04, 0.01, 0.00],
                        [0.01, 0.09, 0.02],
                        [0.00, 0.02, 0.16]],
                       index=list("ABC"), columns=list("ABC"))
    w = recursive_bisection(cov, quasi_diagonal(cluster(cov), list(cov.index)))
    assert np.isclose(w.sum(), 1)
    assert (w > 0).all()


def test_hrp_weights_chains_the_steps():
    cov = pd.DataFrame([[0.04, 0.01, 0.00],
                        [0.01, 0.09, 0.02],
                        [0.00, 0.02, 0.16]],
                       index=list("ABC"), columns=list("ABC"))
    orden = quasi_diagonal(cluster(cov, "ward"), list(cov.index))
    assert hrp_weights(cov, "ward").equals(recursive_bisection(cov, orden).rename("hrp"))


def test_risk_contributions_add_up_to_volatility():
    cov = pd.DataFrame([[0.04, 0.01, 0.00],
                        [0.01, 0.09, 0.02],
                        [0.00, 0.02, 0.16]],
                       index=list("ABC"), columns=list("ABC"))
    w = pd.Series([0.5, 0.3, 0.2], index=list("ABC"))
    rc = risk_contribution(w, cov)
    assert np.isclose(rc.sum(), np.sqrt(w @ cov @ w))


def test_inverse_volatility_equalises_weight_times_vol():
    # Con la inversa de la volatilidad, peso × volatilidad es igual en todos los activos
    vol = np.array([0.1, 0.2, 0.4])
    cov = pd.DataFrame(np.diag(vol**2), index=list("ABC"), columns=list("ABC"))
    w = inverse_volatility(cov)
    assert np.allclose(w * vol, (w * vol).iloc[0])
    assert np.isclose(w.sum(), 1)
