"""Pruebas del backtest con retornos conocidos."""

import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtest.engine import effective_assets, rebalance_dates, run_backtest

DIAS = pd.bdate_range("2010-01-04", "2014-12-31")


def retornos(valores: dict[str, float]) -> pd.DataFrame:
    """Retornos diarios constantes por activo."""
    return pd.DataFrame(valores, index=DIAS)


def equiponderada(r: pd.DataFrame) -> pd.Series:
    return pd.Series(1 / r.shape[1], index=r.columns)


def test_dates_are_quarter_ends_with_full_window():
    fechas = rebalance_dates(DIAS, years=3)
    # 2012-12-31 aún no tiene tres años completos detrás; 2013-03-29 ya sí
    assert fechas[0] == pd.Timestamp("2013-03-29")
    assert all(f.month in (3, 6, 9, 12) for f in fechas)
    assert fechas[-1] == pd.Timestamp("2014-12-31")


def test_strategy_only_sees_the_past():
    # La ventana que recibe la estrategia termina el día del rebalanceo y dura tres años
    vistas = []

    def espia(r):
        vistas.append((r.index[0], r.index[-1]))
        return equiponderada(r)

    run_backtest(retornos({"A": 0.001, "B": 0.0}), espia)
    fechas = rebalance_dates(DIAS)
    assert [fin for _, fin in vistas] == list(fechas[fechas < DIAS[-1]])
    assert vistas[0][0] >= pd.Timestamp("2010-03-30")


def test_returns_start_the_day_after_the_first_rebalance():
    r, _, _ = run_backtest(retornos({"A": 0.001, "B": 0.0}), equiponderada)
    assert r.index[0] == pd.Timestamp("2013-04-01")
    assert r.index[-1] == DIAS[-1]
    assert r.index.is_unique


def test_identical_assets_give_the_asset_return_and_no_turnover():
    r, _, rot = run_backtest(retornos({"A": 0.001, "B": 0.001}), equiponderada)
    assert np.allclose(r, 0.001)
    assert np.isnan(rot.iloc[0])
    assert np.allclose(rot.iloc[1:], 0)


def test_weights_drift_between_rebalances():
    # Sin rebalanceos intermedios, la cartera es un buy and hold de los pesos iniciales
    datos = retornos({"A": 0.002, "B": 0.0})
    r, _, _ = run_backtest(datos, equiponderada, years=1, freq="YE")
    primer_año = r.loc["2012"]
    valor = (1 + primer_año).prod()
    dias = len(primer_año)
    assert valor == pytest.approx(0.5 * 1.002**dias + 0.5)


def test_turnover_restores_the_target():
    # Tras un año A pesa más que B; volver al 50/50 mueve la mitad de la diferencia
    datos = retornos({"A": 0.002, "B": 0.0})
    r, _, rot = run_backtest(datos, equiponderada, years=1, freq="YE")
    crecimiento = 1.002 ** len(r.loc["2012"])
    w_a = 0.5 * crecimiento / (0.5 * crecimiento + 0.5)
    assert rot.iloc[1] == pytest.approx(w_a - 0.5)


def test_weights_must_add_up_to_one():
    with pytest.raises(ValueError, match="suman"):
        run_backtest(retornos({"A": 0.001, "B": 0.0}), lambda r: pd.Series(0.4, index=r.columns))


def test_effective_assets():
    w = pd.DataFrame([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]], columns=list("ABCD"))
    assert np.allclose(effective_assets(w), [4, 1])
    assert effective_assets(w.iloc[0]) == pytest.approx(4)
