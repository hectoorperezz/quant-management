"""Pruebas del equilibrio de Black-Litterman con una covarianza conocida."""

import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers.black_litterman import implied_returns, posterior, risk_aversion

ACTIVOS = ["A", "B", "C"]
SIGMA = pd.DataFrame(
    [[0.04, 0.01, 0.00],
     [0.01, 0.09, 0.02],
     [0.00, 0.02, 0.16]],
    index=ACTIVOS, columns=ACTIVOS,
)
W_MKT = pd.Series([0.5, 0.3, 0.2], index=ACTIVOS)


def test_implied_returns_is_delta_sigma_w():
    pi = implied_returns(W_MKT, SIGMA, delta=2.5)
    esperado = 2.5 * SIGMA.to_numpy() @ W_MKT.to_numpy()
    assert np.allclose(pi.to_numpy(), esperado)
    assert list(pi.index) == ACTIVOS


def test_market_sharpe_is_recovered():
    # Con el δ implícito, el mercado valorado con Π tiene el Sharpe de partida.
    delta = risk_aversion(W_MKT, SIGMA, sharpe=0.3)
    pi = implied_returns(W_MKT, SIGMA, delta)
    vol = np.sqrt(W_MKT @ SIGMA @ W_MKT)
    assert (W_MKT @ pi) / vol == pytest.approx(0.3)


def test_weights_are_aligned_by_name():
    desordenados = W_MKT[["C", "A", "B"]]
    assert implied_returns(desordenados, SIGMA, 2.5).equals(implied_returns(W_MKT, SIGMA, 2.5))


def test_missing_weight_fails():
    with pytest.raises(ValueError, match="C"):
        implied_returns(W_MKT.drop("C"), SIGMA, 2.5)


# Una visión relativa: A batirá a B.
P_AB = pd.DataFrame([[1.0, -1.0, 0.0]], index=["V1"], columns=ACTIVOS)
TAU = 0.05


def _omega(var):
    return pd.DataFrame([[var]], index=["V1"], columns=["V1"])


def test_view_equal_to_equilibrium_changes_nothing():
    # Si Q coincide con lo que ya implica Π, el posterior es el equilibrio.
    pi = implied_returns(W_MKT, SIGMA, 2.5)
    q = pd.Series(P_AB.to_numpy() @ pi.to_numpy(), index=["V1"])
    mu, _ = posterior(pi, SIGMA, P_AB, q, _omega(0.001), TAU)
    assert np.allclose(mu.to_numpy(), pi.to_numpy())


def test_full_confidence_meets_the_view():
    # Con Ω casi cero la visión se cumple exactamente.
    pi = implied_returns(W_MKT, SIGMA, 2.5)
    q = pd.Series([0.03], index=["V1"])
    mu, _ = posterior(pi, SIGMA, P_AB, q, _omega(1e-12), TAU)
    assert mu["A"] - mu["B"] == pytest.approx(0.03, abs=1e-6)


def test_more_confidence_moves_more():
    pi = implied_returns(W_MKT, SIGMA, 2.5)
    q = pd.Series([0.03], index=["V1"])
    poca, _ = posterior(pi, SIGMA, P_AB, q, _omega(0.01), TAU)
    mucha, _ = posterior(pi, SIGMA, P_AB, q, _omega(0.0001), TAU)
    equilibrio = pi["A"] - pi["B"]
    assert abs(mucha["A"] - mucha["B"] - 0.03) < abs(poca["A"] - poca["B"] - 0.03)
    assert poca["A"] - poca["B"] > equilibrio


def test_posterior_covariance_is_larger():
    # Σ_BL = Σ + M: la incertidumbre del estimador se suma a la de los activos.
    pi = implied_returns(W_MKT, SIGMA, 2.5)
    q = pd.Series([0.03], index=["V1"])
    _, sigma_bl = posterior(pi, SIGMA, P_AB, q, _omega(0.001), TAU)
    assert np.all(np.diag(sigma_bl) > np.diag(SIGMA))
