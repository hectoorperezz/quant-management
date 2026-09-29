"""Pruebas de la descarga de precios, sin conexión a internet."""

import pandas as pd
import pytest

from quantmgmt.data import prices


def fake_download(tickers, **kwargs):
    """Imita la respuesta de yfinance: columnas (campo, ticker) ordenadas."""
    index = pd.date_range("2024-01-01", periods=2, freq="D", name="Date")
    data = {
        ("Close", ticker): [1.0 + i, 1.5 + i]
        for i, ticker in enumerate(sorted(tickers))
    }
    result = pd.DataFrame(data, index=index)
    result.columns.names = ["Price", "Ticker"]
    return result


def test_download_prices_uses_cache(tmp_path, monkeypatch):
    calls = []

    def counting_download(tickers, **kwargs):
        calls.append(tickers)
        return fake_download(tickers, **kwargs)

    monkeypatch.setattr(prices.yf, "download", counting_download)

    first = prices.download_prices(["BBB", "AAA"], "2024-01-01", cache_dir=tmp_path)
    second = prices.download_prices(["BBB", "AAA"], "2024-01-01", cache_dir=tmp_path)

    # La segunda llamada lee el CSV de la caché en lugar de descargar.
    assert len(calls) == 1
    assert list(first.columns) == ["BBB", "AAA"]
    assert first.loc["2024-01-02", "AAA"] == pytest.approx(1.5)
    pd.testing.assert_frame_equal(first, second)


def test_download_prices_fails_without_data(tmp_path, monkeypatch):
    def empty_download(tickers, **kwargs):
        return fake_download(tickers).iloc[0:0]

    monkeypatch.setattr(prices.yf, "download", empty_download)

    with pytest.raises(ValueError, match="AAA"):
        prices.download_prices(["AAA"], "2024-01-01", cache_dir=tmp_path)
