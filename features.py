"""Features and date split (copied unchanged from Assignment 2 core.py).

Still to change in later checklist items:
  - build_features(): label is still the 0/1 up-or-down flag; it becomes the forward return
  - split_by_date():  still a 2-way train/test split; it becomes train/val/test with embargo
  - normalization:    not here yet; fit on training rows only
"""
import numpy as np
import pandas as pd

FEATURES = [
    "ret5",          # return over last 5 days
    "ret20",         # return over last 20 days
    "px_ma50",       # price / 50-day moving average
    "ma20_ma50",     # 20-day MA / 50-day MA
    "vol_ratio",     # volume / average volume of last 20 days
    "volat20",       # std-dev of daily returns, last 20 days
]


# ----------------------------------------------------------------------- features
def build_features(prices, horizon=20):
    """One row per (Symbol, Date).

    NO LOOK-AHEAD: every feature on date t is computed from closes/volumes of
    days strictly BEFORE t (we compute on the raw series, then shift by one day).
    Label on date t: 1 if Close[t+horizon] > Close[t], else 0. It is the only column
    that looks forward, and it is NaN where the future does not exist yet.
    """
    out = []
    for sym, g in prices.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        c, v = g["Close"], g["Volume"]
        daily = c.pct_change()
        f = pd.DataFrame({
            "ret5": c / c.shift(5) - 1,
            "ret20": c / c.shift(20) - 1,
            "px_ma50": c / c.rolling(50).mean(),
            "ma20_ma50": c.rolling(20).mean() / c.rolling(50).mean(),
            "vol_ratio": v / v.rolling(20).mean(),
            "volat20": daily.rolling(20).std(),
        })
        f = f.shift(1)  # <- the leak guard: row t only sees data through t-1
        f["Date"], f["Symbol"], f["Close"] = g["Date"], sym, c
        fwd = c.shift(-horizon)
        f["label"] = np.where(fwd.isna(), np.nan, (fwd > c).astype(float))
        out.append(f)
    feats = pd.concat(out, ignore_index=True)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return feats.sort_values(["Date", "Symbol"]).reset_index(drop=True)


# ------------------------------------------------------------------------- split
def split_by_date(feats, split_date=None, split_pct=80, horizon=20, embargo=None):
    """Sort by date; train on dates < split_date, test on dates >= split_date.
    The last `embargo` (default = horizon) training DATES are dropped so no training label
    reaches into the test period."""
    embargo = int(horizon) if embargo in (None, "") else int(embargo)
    labelled = feats.dropna(subset=["label"])
    dates = np.sort(labelled["Date"].unique())
    if len(dates) < 10:
        raise ValueError("Not enough dated rows to split. Use a longer price history.")
    if split_date:
        sd = pd.Timestamp(split_date)
    else:
        sd = pd.Timestamp(dates[int(len(dates) * split_pct / 100)])
    train_dates = dates[dates < np.datetime64(sd)]
    test_dates = dates[dates >= np.datetime64(sd)]
    if len(train_dates) <= embargo + 5 or len(test_dates) < 5:
        raise ValueError(
            f"Split date {sd.date()} leaves too little data (train dates {len(train_dates)}, "
            f"test dates {len(test_dates)}, embargo {embargo}). Pick a date inside the history."
        )
    kept_train = train_dates[: len(train_dates) - embargo] if embargo else train_dates
    train = labelled[labelled["Date"].isin(kept_train)]
    test = labelled[labelled["Date"].isin(test_dates)]
    return train, test, {
        "split_date": sd.strftime("%Y-%m-%d"),
        "embargo": embargo,
        "train_start": str(kept_train[0])[:10], "train_end": str(kept_train[-1])[:10],
        "test_start": str(test_dates[0])[:10], "test_end": str(test_dates[-1])[:10],
    }
