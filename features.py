"""Features, forward-return label, 3-way date split with embargo, training-only normalization.

Modified from Assignment 2 (core.py):
  - label is the forward RETURN (a decimal such as +0.027), not a 0/1 up-or-down flag
  - split is train / validation / test, chronological, with an embargo at BOTH boundaries
  - features are normalized with mean/std fitted on TRAINING rows only

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

    Label on date t: the forward return Close[t+horizon] / Close[t] - 1, as a decimal.
    `label_date` is the date that future close comes from. Both look forward, and both are
    NaN where the future does not exist yet (the most recent `horizon` rows per symbol).
    Those rows are kept so the final date can still be ranked, but they have no label.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
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
        f["label"] = c.shift(-horizon) / c - 1            # continuous forward return
        f["label_date"] = g["Date"].shift(-horizon)       # when that future close happens
        out.append(f)
    feats = pd.concat(out, ignore_index=True)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return feats.sort_values(["Date", "Symbol"]).reset_index(drop=True)


# ------------------------------------------------------------------------- split
def split_by_date(feats, val_start=None, test_start=None, train_pct=70, val_pct=15,
                  horizon=20, embargo=None):
    """Chronological train / validation / test split with an embargo at both boundaries.

        |<------ TRAIN ------>|<-embargo->|<--- VAL --->|<-embargo->|<----- TEST ----->|

    Blocks are cut by DATE: train < val_start <= val < test_start <= test. If the dates are not
    given they are picked from train_pct / val_pct (the rest is test).

    The last `embargo` dates of train and of validation are dropped. A row's label looks
    `horizon` days ahead, so without the gap the last training labels would be computed from
    prices inside the validation period (and the last validation labels from the test period).
    The embargo may not be shorter than the horizon.

    Returns (train, val, test, info). `info` holds the ranges for display.
    """
    horizon = int(horizon)
    embargo = horizon if embargo in (None, "") else int(embargo)
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    if embargo < horizon:
        raise ValueError(f"Embargo ({embargo}) must be at least the label horizon ({horizon}); "
                         "a shorter gap lets labels peek into the next block.")
    if not (0 < train_pct and 0 < val_pct and train_pct + val_pct < 100):
        raise ValueError("train_pct and val_pct must be positive and add up to less than 100.")

    labelled = feats.dropna(subset=["label"])
    dates = np.sort(labelled["Date"].unique())
    n = len(dates)
    if n < 3 * (embargo + 5):
        raise ValueError(f"Only {n} labelled dates: not enough for three blocks with a "
                         f"{embargo}-day embargo. Use a longer price history.")

    vs = pd.Timestamp(val_start) if val_start else pd.Timestamp(dates[int(n * train_pct / 100)])
    ts = pd.Timestamp(test_start) if test_start else pd.Timestamp(dates[int(n * (train_pct + val_pct) / 100)])
    if not vs < ts:
        raise ValueError(f"Validation start ({vs.date()}) must be before test start ({ts.date()}).")

    train_all = dates[dates < np.datetime64(vs)]
    val_all = dates[(dates >= np.datetime64(vs)) & (dates < np.datetime64(ts))]
    test_dates = dates[dates >= np.datetime64(ts)]
    for name, d in (("train", train_all), ("validation", val_all)):
        if len(d) <= embargo + 5:
            raise ValueError(f"The {name} block has only {len(d)} dates, which is too few after "
                             f"dropping the {embargo}-day embargo. Move the split dates.")
    if len(test_dates) < 5:
        raise ValueError(f"The test block has only {len(test_dates)} dates. Move the test start earlier.")

    kept_train, gap1 = train_all[: len(train_all) - embargo], train_all[len(train_all) - embargo:]
    kept_val, gap2 = val_all[: len(val_all) - embargo], val_all[len(val_all) - embargo:]
    train = labelled[labelled["Date"].isin(kept_train)]
    val = labelled[labelled["Date"].isin(kept_val)]
    test = labelled[labelled["Date"].isin(test_dates)]

    # Exact leak check: no label may be computed from a price on or after the next block starts.
    if train["label_date"].max() >= val["Date"].min():
        raise AssertionError("LEAK: a training label looks into the validation period.")
    if val["label_date"].max() >= test["Date"].min():
        raise AssertionError("LEAK: a validation label looks into the test period.")

    day = lambda x: str(x)[:10]  # noqa: E731
    return train, val, test, {
        "horizon": horizon, "embargo": embargo,
        "val_start": vs.strftime("%Y-%m-%d"), "test_start": ts.strftime("%Y-%m-%d"),
        "train": {"start": day(kept_train[0]), "end": day(kept_train[-1]), "rows": int(len(train)), "dates": int(len(kept_train))},
        "gap1": {"start": day(gap1[0]), "end": day(gap1[-1]), "dates": int(len(gap1))},
        "val": {"start": day(kept_val[0]), "end": day(kept_val[-1]), "rows": int(len(val)), "dates": int(len(kept_val))},
        "gap2": {"start": day(gap2[0]), "end": day(gap2[-1]), "dates": int(len(gap2))},
        "test": {"start": day(test_dates[0]), "end": day(test_dates[-1]), "rows": int(len(test)), "dates": int(len(test_dates))},
    }


# ----------------------------------------------------------------------- normalize
class Normalizer:
    """z-score scaler: (x - mean) / std, with mean and std taken from TRAINING rows only.

    Fit once on the training block, then call transform() on train, validation and test with
    those same numbers. Uses the population std (ddof=0), like sklearn's StandardScaler.
    Pass cols=FEATURES + ["sentiment"] later to scale a 7th input the same way."""

    def __init__(self, cols=FEATURES):
        self.cols = list(cols)
        self.mean = None
        self.std = None
        self.n_rows = 0
        self.fit_start = self.fit_end = None

    def fit(self, train):
        if len(train) == 0:
            raise ValueError("Cannot fit a normalizer on zero rows.")
        self.mean = train[self.cols].mean()
        std = train[self.cols].std(ddof=0)
        self.std = std.where(std > 1e-12, 1.0)      # a constant column would divide by zero
        self.n_rows = int(len(train))
        if "Date" in train:
            self.fit_start, self.fit_end = str(train["Date"].min())[:10], str(train["Date"].max())[:10]
        return self

    def transform(self, df):
        if self.mean is None:
            raise RuntimeError("Call fit() on the training rows first.")
        out = df.copy()
        out[self.cols] = (df[self.cols] - self.mean) / self.std
        return out

    def table(self):
        return pd.DataFrame({"mean": self.mean, "std": self.std})


# ------------------------------------------------------------------------- report
def format_split_report(info):
    def row(label, b, extra=""):
        return f"  {label:<9}{b['start']} .. {b['end']}   {b['dates']:>5} dates" + (f"  {b['rows']:>7,} rows" if "rows" in b else "") + extra
    return "\n".join([
        f"SPLIT  chronological, label horizon {info['horizon']}d, embargo {info['embargo']} dates at each boundary",
        row("TRAIN", info["train"]),
        row("embargo", info["gap1"], "   (dropped)"),
        row("VAL", info["val"]),
        row("embargo", info["gap2"], "   (dropped)"),
        row("TEST", info["test"]),
    ])


def format_norm_report(norm, train, val, test):
    """Means/stds used, plus the normalized mean/std of each block. Train comes out at about
    (0, 1) by construction; val and test do not, because they were scaled with TRAIN's numbers."""
    tn, vn, sn = norm.transform(train), norm.transform(val), norm.transform(test)
    cell = lambda df, c: f"({df[c].mean():+.2f}, {df[c].std(ddof=0):.2f})".ljust(15)  # noqa: E731
    lines = [
        f"NORMALIZATION  fitted on TRAIN rows only: {norm.n_rows:,} rows, {norm.fit_start} .. {norm.fit_end}",
        f"  {'feature':<10}{'train mean':>12}{'train std':>11}  | after scaling, (mean, std) of:  "
        f"{'train':<15}{'val':<15}{'test':<15}",
    ]
    for c in norm.cols:
        lines.append(f"  {c:<10}{norm.mean[c]:>12.5f}{norm.std[c]:>11.5f}  | {'':<31}"
                     f"{cell(tn, c)}{cell(vn, c)}{cell(sn, c)}")
    return "\n".join(lines)


def describe(feats, **split_kwargs):
    """Split, normalize, and return (train_n, val_n, test_n, norm, report_text)."""
    train, val, test, info = split_by_date(feats, **split_kwargs)
    norm = Normalizer(FEATURES).fit(train)
    report = format_split_report(info) + "\n\n" + format_norm_report(norm, train, val, test)
    return norm.transform(train), norm.transform(val), norm.transform(test), norm, report


if __name__ == "__main__":
    import os
    from data import load_prices

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_market_A.csv")
    feats = build_features(load_prices(path), 20)
    print(describe(feats)[-1])
