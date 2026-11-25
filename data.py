"""Data loading: prices and basic info (from Assignment 2 core.py), plus the news/headline source.

News is timestamped text, one row per headline: Timestamp, Symbol, Headline. The only way the
rest of the app should read headlines is headlines_before(), which enforces "strictly before the
decision date" so a headline from the future cannot reach the LLM.
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


# --------------------------------------------------------------------------- news
_NEWS_ALIASES = {
    "timestamp": "Timestamp", "datetime": "Timestamp", "date": "Timestamp",
    "published": "Timestamp", "time": "Timestamp",
    "symbol": "Symbol", "ticker": "Symbol", "stock": "Symbol",
    "headline": "Headline", "title": "Headline", "text": "Headline",
}


def load_news(source):
    """Load a headline file: Timestamp, Symbol, Headline (one row per headline).
    `source` is a path or raw bytes. Column names are matched case-insensitively
    ('Date', 'Published', 'Ticker', 'Title' ... are accepted). Timestamps with a time zone are
    converted to UTC and stored without one, so every comparison is between naive datetimes."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    df = df.rename(columns={c: _NEWS_ALIASES.get(c.strip().lower(), c) for c in df.columns})
    missing = {"Timestamp", "Symbol", "Headline"} - set(df.columns)
    if missing:
        raise ValueError(
            "News file needs columns Timestamp, Symbol, Headline "
            f"(missing: {', '.join(sorted(missing))}). Found: {', '.join(df.columns)}"
        )
    df = df[["Timestamp", "Symbol", "Headline"]].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Headline"] = df["Headline"].astype(str).str.strip()
    df = df[df["Timestamp"].notna() & (df["Headline"] != "") & (df["Headline"].str.lower() != "nan")]
    df = df.sort_values(["Symbol", "Timestamp"]).drop_duplicates(["Symbol", "Timestamp", "Headline"])
    if df.empty:
        raise ValueError("News file has no usable rows.")
    return df.reset_index(drop=True)


def news_info(df):
    counts = df.groupby("Symbol").size()
    return {
        "symbols": int(df["Symbol"].nunique()),
        "rows": int(len(df)),
        "date_min": df["Timestamp"].min().strftime("%Y-%m-%d"),
        "date_max": df["Timestamp"].max().strftime("%Y-%m-%d"),
        "per_symbol": {s: int(n) for s, n in counts.items()},
    }


def headlines_before(news, symbol, decision_date, window_days=30, max_n=12):
    """Headlines for `symbol` published STRICTLY BEFORE the decision date, newest `max_n` of those
    inside the last `window_days` days, oldest first, as [(Timestamp, text), ...].

    Strictly before means before midnight at the START of the decision date: a headline from the
    afternoon of the decision day is already the future."""
    cutoff = pd.Timestamp(decision_date).normalize()
    sub = news[(news["Symbol"] == symbol)
               & (news["Timestamp"] < cutoff)
               & (news["Timestamp"] >= cutoff - pd.Timedelta(days=int(window_days)))]
    sub = sub.sort_values("Timestamp").tail(int(max_n))
    return list(zip(sub["Timestamp"], sub["Headline"]))
