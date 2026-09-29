"""Descarga de precios diarios con yfinance y caché local en ``data/cache/``."""

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yfinance as yf

# Carpeta data/cache/ en la raíz del repositorio (excluida de Git).
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def download_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Descarga precios de cierre diarios, ajustados por dividendos y splits.

    La primera descarga se guarda como CSV en ``cache_dir`` y las siguientes
    llamadas con los mismos argumentos la leen de ahí. Así el análisis se
    puede repetir sin depender de Yahoo Finance.

    Parámetros
    ----------
    tickers : lista de str
        Símbolos de Yahoo Finance, por ejemplo ``["SPY", "TLT"]``.
    start, end : str
        Fechas ``"AAAA-MM-DD"``. ``end`` no se incluye; si es None, la
        descarga llega hasta hoy.
    cache_dir : str o Path
        Carpeta de la caché. Por defecto, ``data/cache/`` del repositorio.
    refresh : bool
        Si es True, ignora la caché y vuelve a descargar.

    Devuelve
    --------
    pd.DataFrame
        Precios indexados por fecha, una columna por ticker en el orden pedido.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers debe contener al menos un símbolo")

    cache_file = Path(cache_dir) / _cache_name(tickers, start, end)
    if refresh or not cache_file.exists():
        _download_to_csv(tickers, start, end, cache_file)

    # Siempre se devuelve lo que hay en la caché, para que todas las llamadas
    # den exactamente el mismo resultado.
    return pd.read_csv(cache_file, index_col="Date", parse_dates=True)


def _download_to_csv(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_file: Path,
) -> None:
    """Descarga los precios con yfinance y los guarda en ``cache_file``."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # Nos quedamos con el cierre ajustado, en el orden pedido.
    prices = data["Close"].reindex(columns=tickers).dropna(how="all")
    missing = [ticker for ticker in tickers if prices[ticker].isna().all()]
    if missing:
        raise ValueError(f"Yahoo Finance no ha devuelto precios para: {missing}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.rename_axis(index="Date", columns=None).to_csv(cache_file)


def _cache_name(tickers: list[str], start: str, end: str | None) -> str:
    """Construye el nombre del CSV de caché a partir de la descarga pedida."""
    name = "_".join([*tickers, start, end or "hoy"])
    return re.sub(r"[^A-Za-z0-9._-]", "-", name) + ".csv"
