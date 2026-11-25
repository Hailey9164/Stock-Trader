"""Features, forward-return label, 3-way date split with embargo, training-only normalization.

Base: teammate's features.py (split_by_date, fit_scaler, apply_scaler, normalize), kept as written.
Changes and additions are marked in the code:
  # CHANGED  build_features(): the label is now the forward RETURN, not the 0/1 up-or-down flag
  # ADDED    label_date column, an exact leak check and the embargo guard in split_by_date(),
  #          the gap date ranges in `info`, and format_report() / the __main__ demo that print
  #          the date ranges and the means and stds

Run `python features.py [prices.csv]` to print the split ranges and the normalization stats.
"""
import sys

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
    Label on date t: the forward return Close[t+horizon] / Close[t] - 1, as a decimal
    (e.g. +0.027). `label_date` is the date that future close comes from. Both look forward,
    and both are NaN where the future does not exist yet.
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
        f["label"] = fwd / c - 1                      # CHANGED: continuous forward return (was a 0/1 flag)
        f["label_date"] = g["Date"].shift(-horizon)   # ADDED: when that future close happens
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
    if int(horizon) < 1 or embargo < int(horizon):   # ADDED: a shorter gap lets labels peek ahead
        raise ValueError(f"Embargo ({embargo}) must be at least the label horizon ({horizon}) "
                         "and the horizon at least 1; a shorter gap lets labels peek into the next block.")
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
    gap1 = train_dates[len(train_dates) - embargo:]   # ADDED: the dropped dates, so the GUI can show the gap
    gap2 = val_dates[len(val_dates) - embargo:]
    train = labelled[labelled["Date"].isin(kept_train)]
    val = labelled[labelled["Date"].isin(kept_val)]
    test = labelled[labelled["Date"].isin(test_dates)]

    # ADDED: exact leak check. No label may be computed from a price on or after the next block starts.
    if "label_date" in labelled.columns:
        if train["label_date"].max() >= val["Date"].min():
            raise AssertionError("LEAK: a training label looks into the validation period.")
        if val["label_date"].max() >= test["Date"].min():
            raise AssertionError("LEAK: a validation label looks into the test period.")

    def rng(d):
        return str(d[0])[:10], str(d[-1])[:10]

    info = {
        "val_date": vd.strftime("%Y-%m-%d"),
        "test_date": td.strftime("%Y-%m-%d"),
        "embargo": embargo,
        "train_start": rng(kept_train)[0], "train_end": rng(kept_train)[1],
        "val_start": rng(kept_val)[0], "val_end": rng(kept_val)[1],
        "test_start": rng(test_dates)[0], "test_end": rng(test_dates)[1],
        "gap1_start": rng(gap1)[0], "gap1_end": rng(gap1)[1],     # ADDED
        "gap2_start": rng(gap2)[0], "gap2_end": rng(gap2)[1],     # ADDED
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


# ------------------------------------------------------------------------- report
# ADDED: text for the "display the ranges" and "show that you normalized" requirements.
def format_report(info, mu, sigma, train_n, val_n, test_n):
    """Date ranges (with the embargo gaps) and the normalization stats.

    The last three columns are the mean and std of each block AFTER scaling. Train comes out at
    about (0, 1) by construction; val and test do not, because they were scaled with TRAIN's
    mu and sigma, which is the point."""
    split = [
        f"SPLIT  chronological, embargo {info['embargo']} dates at each boundary",
        f"  TRAIN    {info['train_start']} .. {info['train_end']}   {info['train_rows']:>7,} rows",
        f"  embargo  {info['gap1_start']} .. {info['gap1_end']}   (dropped)",
        f"  VAL      {info['val_start']} .. {info['val_end']}   {info['val_rows']:>7,} rows",
        f"  embargo  {info['gap2_start']} .. {info['gap2_end']}   (dropped)",
        f"  TEST     {info['test_start']} .. {info['test_end']}   {info['test_rows']:>7,} rows",
    ]
    cell = lambda df, c: f"({df[c].mean():+.2f}, {df[c].std(ddof=0):.2f})".ljust(15)  # noqa: E731
    norm = [
        f"NORMALIZATION  fitted on TRAIN rows only: {info['train_rows']:,} rows, "
        f"{info['train_start']} .. {info['train_end']}",
        f"  {'feature':<10}{'train mean':>12}{'train std':>11}  | after scaling, (mean, std) of:  "
        f"{'train':<15}{'val':<15}{'test':<15}",
    ]
    for c in mu.index:
        norm.append(f"  {c:<10}{mu[c]:>12.5f}{sigma[c]:>11.5f}  | {'':<31}"
                    f"{cell(train_n, c)}{cell(val_n, c)}{cell(test_n, c)}")
    return "\n".join(split) + "\n\n" + "\n".join(norm)


if __name__ == "__main__":
    import os
    from data import load_prices

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_market_A.csv")
    feats = build_features(load_prices(path), 20)
    tr, va, te, info = split_by_date(feats)
    tr_n, va_n, te_n, mu, sigma = normalize(tr, va, te)
    print(format_report(info, mu, sigma, tr_n, va_n, te_n))
