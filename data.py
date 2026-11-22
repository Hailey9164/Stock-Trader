"""Data loading: price file loader and basic info (copied from Assignment 2 core.py).

TODO (later checklist item): add load_news() for the headline source.
"""
import io

import pandas as pd

_ALIASES = {
    "date": "Date", "symbol": "Symbol", "ticker": "Symbol", "stock": "Symbol",
    "close": "Close", "adj close": "Close", "adj_close": "Close", "adjclose": "Close",
    "price": "Close", "volume": "Volume",
}


# --------------------------------------------------------------------------- data
def load_prices(source):
    """Load a long-format price file: Date, Symbol, Close, Volume (one row per stock per day).
    `source` is a path or raw bytes. Column names are matched case-insensitively
    ('Ticker' and 'Adj Close' are accepted)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    df = df.rename(columns={c: _ALIASES.get(c.strip().lower(), c) for c in df.columns})
    missing = {"Date", "Symbol", "Close", "Volume"} - set(df.columns)
    if missing:
        raise ValueError(
            "Price file needs columns Date, Symbol, Close, Volume "
            f"(missing: {', '.join(sorted(missing))}). Found: {', '.join(df.columns)}"
        )
    df = df[["Date", "Symbol", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna().sort_values(["Symbol", "Date"]).drop_duplicates(["Symbol", "Date"])
    if df.empty:
        raise ValueError("Price file has no usable rows.")
    return df.reset_index(drop=True)


def data_info(df):
    return {
        "symbols": int(df["Symbol"].nunique()),
        "rows": int(len(df)),
        "date_min": df["Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Date"].max().strftime("%Y-%m-%d"),
        "symbol_list": sorted(df["Symbol"].unique().tolist()),
    }
