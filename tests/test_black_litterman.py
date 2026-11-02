"""Pruebas del equilibrio de Black-Litterman con una covarianza conocida."""

import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers.black_litterman import implied_returns, risk_aversion

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
