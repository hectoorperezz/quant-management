"""Descarga de factores Fama-French y momentum (Ken French Data Library) con caché."""

from pathlib import Path

import pandas as pd
import pandas_datareader.data as web

from quantmgmt.data.prices import DEFAULT_CACHE_DIR

FF5 = "F-F_Research_Data_5_Factors_2x3"
MOM = "F-F_Momentum_Factor"


def download_ff_factors(
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Factores mensuales Mkt-RF, SMB, HML, RMW, CMA, Mom y RF.

    Devuelve rentabilidades simples en DECIMAL (Ken French publica en %),
    indexadas por fin de mes para alinearse con ``resample("ME")``.
    Mkt-RF y el resto de factores ya están en exceso sobre RF.
    """
    cache_file = Path(cache_dir) / f"ff5_mom_{start}_{end or 'hoy'}.csv"    
    if refresh or not cache_file.exists():
        ff5 = web.DataReader(FF5, "famafrench", start=start, end=end)[0]    
        mom = web.DataReader(MOM, "famafrench", start=start, end=end)[0]
        mom.columns = mom.columns.str.strip()
        factors = ff5.join(mom, how="inner") / 100
        factors.index = factors.index.to_timestamp(how="end").normalize()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        factors.rename_axis("Date").to_csv(cache_file)
    return pd.read_csv(cache_file, index_col="Date", parse_dates=True)