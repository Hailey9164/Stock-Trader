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
def split_by_date(feats, val_date=None, test_date=None, train_pct=70, val_pct=15,
                  horizon=20, embargo=None):
    """Chronological train / validation / test split, with an embargo gap.

        |<------ TRAIN ------>|<-gap->|<-- VAL -->|<-gap->|<---- TEST ---->|
                              ^ embargo           ^ embargo

    Blocks are cut by date: train < val_date <= val < test_date <= test.
    If the dates are not given they are picked from train_pct / val_pct of the
    labelled dates (test gets the remainder).

    A row's label looks `horizon` days ahead, so the last `embargo` dates (default =
    horizon) of the training block AND of the validation block are dropped. That way no
    label reaches into the next block. The test block is never touched except at the end.

    Returns (train, val, test, info). `info` holds the date ranges to display in the GUI.
    """
    embargo = int(horizon) if embargo in (None, "") else int(embargo)
    labelled = feats.dropna(subset=["label"])
    dates = np.sort(labelled["Date"].unique())
    if len(dates) < 30:
        raise ValueError("Not enough dated rows to split. Use a longer price history.")

    if val_date:
        vd = pd.Timestamp(val_date)
    else:
        vd = pd.Timestamp(dates[int(len(dates) * train_pct / 100)])
    if test_date:
        td = pd.Timestamp(test_date)
    else:
        td = pd.Timestamp(dates[int(len(dates) * (train_pct + val_pct) / 100)])
    if not vd < td:
        raise ValueError(f"Validation start {vd.date()} must come before test start {td.date()}.")

    train_dates = dates[dates < np.datetime64(vd)]
    val_dates = dates[(dates >= np.datetime64(vd)) & (dates < np.datetime64(td))]
    test_dates = dates[dates >= np.datetime64(td)]
    if len(train_dates) <= embargo + 5 or len(val_dates) <= embargo + 5 or len(test_dates) < 5:
        raise ValueError(
            f"Split dates {vd.date()} / {td.date()} leave too little data "
            f"(train dates {len(train_dates)}, val dates {len(val_dates)}, "
            f"test dates {len(test_dates)}, embargo {embargo}). Pick dates inside the history."
        )

    kept_train = train_dates[: len(train_dates) - embargo]
    kept_val = val_dates[: len(val_dates) - embargo]
    train = labelled[labelled["Date"].isin(kept_train)]
    val = labelled[labelled["Date"].isin(kept_val)]
    test = labelled[labelled["Date"].isin(test_dates)]

    def rng(d):
        return str(d[0])[:10], str(d[-1])[:10]

    info = {
        "val_date": vd.strftime("%Y-%m-%d"),
        "test_date": td.strftime("%Y-%m-%d"),
        "embargo": embargo,
        "train_start": rng(kept_train)[0], "train_end": rng(kept_train)[1],
        "val_start": rng(kept_val)[0], "val_end": rng(kept_val)[1],
        "test_start": rng(test_dates)[0], "test_end": rng(test_dates)[1],
        "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
    }
    return train, val, test, info


# ------------------------------------------------------------------ normalization
def fit_scaler(train, cols=None):
    """Mean and standard deviation of each feature, computed on TRAINING rows only.
    
    Returns (mu, sigma) as pandas Series indexed by column name. A column with zero
    spread gets sigma = 1 so we never divide by zero. Pass cols=FEATURES + ["sentiment"]
    later to include the LLM score as the seventh input.
    """
    cols = list(FEATURES if cols is None else cols)
    mu = train[cols].mean()
    sigma = train[cols].std(ddof=0).replace(0, 1.0)
    return mu, sigma

def apply_scaler(df, mu, sigma):
    """Return a copy of df with the scaler's columns converted to (x - mu) / sigma.
    
    Always call this with the mu/sigma from fit_scaler(train): the same numbers go to
    train, validation, and test. Never fit on val or test.
    """
    out = df.copy()
    cols = list(mu.index)
    out[cols] = (df[cols] - mu) / sigma
    return out

def normalize(train, val, test, cols=None):
    """Fit on train only, apply to all three. Returns (train_n, val_n, test_n, mu, sigma)."""
    mu, sigma = fit_scaler(train, cols)
    return (apply_scaler(train, mu, sigma), apply_scaler(val, mu, sigma), apply_scaler(test, mu, sigma), mu, sigma)