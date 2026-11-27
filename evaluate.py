"""Evaluation: the one-time test-set comparison, and the with / without sentiment ablation.

    python evaluate.py [prices.csv] [news.csv] [--llm] [--every 5] [--cache sentiment_cache.json] [--out results]

Honest evaluation rules this module follows:
  * The test block is used for exactly one thing: scoring models that are already fully trained.
    Training, early stopping, the scaler and every setting are fixed before the test block is looked at.
  * Each configuration is scored on the test block once. Nothing is tuned afterwards.
  * Every model is judged by the same four numbers: MAE, RMSE, direction accuracy, R-squared.
  * If a result looks too good (direction accuracy above 80%, R-squared above 0.5), a warning says so.
    The walkthrough calls that a bug, not a discovery.

"Up" means a strictly positive forward return; direction is +1 for up, -1 otherwise.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from features import FEATURES, normalize
from models import ForestRegressor, MeanBaseline, NeuralNet, _torch, format_history, plot_loss_curves, to_xy

METRICS = ["mae", "rmse", "direction_acc", "r2"]
_HEADINGS = {"mae": "test MAE", "rmse": "test RMSE", "direction_acc": "direction acc", "r2": "R-squared"}


# ----------------------------------------------------------------------- metrics
def direction_of(x):
    return np.where(np.asarray(x, dtype=float) > 0, 1, -1)


def metrics(pred, y, pred_direction=None):
    """MAE, RMSE, direction accuracy and R-squared. `pred_direction` overrides sign(pred), which is
    how the baseline's "always the training majority direction" is scored."""
    pred, y = np.asarray(pred, dtype=float), np.asarray(y, dtype=float)
    if pred.shape != y.shape or y.size == 0:
        raise ValueError(f"pred and y must be the same non-empty shape (got {pred.shape} and {y.shape}).")
    err = pred - y
    d = direction_of(pred) if pred_direction is None else np.asarray(pred_direction)
    sst = float(((y - y.mean()) ** 2).sum())
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "direction_acc": float((d == direction_of(y)).mean()),
            "r2": 1.0 - float((err ** 2).sum()) / sst if sst > 0 else float("nan")}


def compare_on_test(models, X_test, y_test):
    """One row per (already fitted) model: its test-block metrics. This is the only place the test
    block is scored."""
    rows = []
    for m in models:
        d = m.predict_direction(X_test) if hasattr(m, "predict_direction") else None
        rows.append({"model": m.name, **metrics(m.predict(X_test), y_test, d)})
    return pd.DataFrame(rows, columns=["model", *METRICS])


def format_table(df):
    head = f"{'model':<30}" + "".join(f"{_HEADINGS[c]:>15}" for c in METRICS)
    lines = [head]
    for _, r in df.iterrows():
        lines.append(f"{r['model']:<30}{r['mae']:>15.5f}{r['rmse']:>15.5f}{r['direction_acc']:>15.1%}{r['r2']:>15.4f}")
    return "\n".join(lines)


def _too_good(table):
    """Walkthrough tell: direction accuracy above 80%, or R-squared near 1, means a leak. Direction accuracy
    is compared with the baseline's so a test period that is simply 85% up days does not raise a false alarm."""
    base = table[table["model"].str.startswith("Baseline")]["direction_acc"]
    base_acc = float(base.iloc[0]) if len(base) else 0.5
    out = []
    for _, r in table.iterrows():
        if r["model"].startswith("Baseline"):
            continue
        if (r["direction_acc"] > 0.80 and r["direction_acc"] > base_acc + 0.05) or r["r2"] > 0.5:
            out.append(f"{r['model']}: direction accuracy {r['direction_acc']:.1%} (baseline {base_acc:.1%}), "
                       f"R-squared {r['r2']:.3f} on the test block is too good to be real. The walkthrough calls "
                       "that a bug, not a discovery. Check, in order: normalization fitted on training rows only, a "
                       "date-based split, the embargo, and headlines dated before the decision date.")
    return out


# --------------------------------------------------------------------- comparison
def run_comparison(train, val, test, cols=None, ann_params=None, forest_params=None, on_epoch=None):
    """Fit baseline, forest and network on TRAIN (the network early-stops on VAL), then score all three
    on TEST, once. The frames must already be split by date (features.split_by_date).

    Returns a dict: table (DataFrame), models, ann (the NeuralNet, or None), cols, mu, sigma,
    sizes, notes, warnings."""
    cols = list(FEATURES if cols is None else cols)
    tr_n, va_n, te_n, mu, sigma = normalize(train, val, test, cols)     # scaler fitted on train only
    (Xtr, ytr), (Xva, yva), (Xte, yte) = to_xy(tr_n, cols), to_xy(va_n, cols), to_xy(te_n, cols)

    baseline = MeanBaseline().fit(Xtr, ytr)
    forest = ForestRegressor(**(forest_params or {})).fit(Xtr, ytr)
    fitted, notes, ann = [baseline, forest], [], None
    try:
        _torch()
    except ImportError as e:
        notes.append(f"Neural network skipped: {e}")
    else:
        ann = NeuralNet(**(ann_params or {})).fit(Xtr, ytr, Xva, yva, on_epoch=on_epoch)
        fitted.append(ann)

    # ---- everything above is fixed. The next line is the one and only look at the test block. ----
    table = compare_on_test(fitted, Xte, yte)
    return {"table": table, "models": {"baseline": baseline, "forest": forest, "ann": ann}, "ann": ann,
            "cols": cols, "mu": mu, "sigma": sigma, "notes": notes, "warnings": _too_good(table),
            "sizes": {"train": len(train), "val": len(val), "test": len(test)}}


# ----------------------------------------------------------------------- ablation
def run_ablation(train, val, test, ann_params=None, forest_params=None, on_epoch=None, progress=None):
    """Run the whole comparison twice, identically except for the sentiment column:
        WITH sentiment   the real scores from sentiment.sentiment_for_rows
        ZEROED           the same 7th input, set to 0.0 on every row
    The frames need a 'sentiment' column. Seeds and settings are the same in both runs, so a difference
    comes from sentiment alone. Returns both results, the difference, and a plain-language summary."""
    for name, f in (("train", train), ("val", val), ("test", test)):
        if "sentiment" not in f.columns:
            raise ValueError(f"The {name} frame has no 'sentiment' column. Add one (all 0.0 for LLM-off).")
    cols = FEATURES + ["sentiment"]
    say = progress or (lambda msg: None)
    kw = dict(cols=cols, ann_params=ann_params, forest_params=forest_params, on_epoch=on_epoch)

    say("Run 1 of 2: WITH sentiment")
    with_s = run_comparison(train, val, test, **kw)
    say("Run 2 of 2: sentiment ZEROED")
    zero = run_comparison(train.assign(sentiment=0.0), val.assign(sentiment=0.0), test.assign(sentiment=0.0), **kw)

    a, b = with_s["table"].set_index("model")[METRICS], zero["table"].set_index("model")[METRICS]
    delta = (a - b).reset_index()                        # with minus zeroed: negative MAE / RMSE = sentiment helped
    all_zero = bool(all((f["sentiment"] == 0).all() for f in (train, val, test)))
    return {"with_sentiment": with_s, "zeroed": zero, "delta": delta, "sentiment_all_zero": all_zero,
            "summary": _summary(a, b, all_zero)}


def _summary(a, b, all_zero):
    if all_zero:
        return ["Sentiment is 0.0 on every row (LLM off, or no headlines), so the two runs are the same experiment. "
                "Any difference below is run-to-run noise, and there is no sentiment finding to report."]
    lines = []
    for model in a.index:
        if model.startswith("Baseline"):
            continue                                     # the baseline ignores features, so it cannot change
        x, y = a.loc[model], b.loc[model]
        lines.append(f"{model}: test MAE {x['mae']:.5f} with sentiment vs {y['mae']:.5f} zeroed "
                     f"({x['mae'] - y['mae']:+.5f}, {(x['mae'] - y['mae']) / y['mae']:+.1%}); direction accuracy "
                     f"{x['direction_acc']:.1%} vs {y['direction_acc']:.1%} ({(x['direction_acc'] - y['direction_acc']) * 100:+.1f} points).")
    lines.append("Lower MAE is better. This is one seed per run on one split, so treat small differences as noise: "
                 "'removing sentiment changes nothing' is a legitimate finding.")
    return lines


# ----------------------------------------------------------------------- reporting
def _split_lines(info):
    return ["SPLIT  chronological, embargo %d dates at each boundary" % info["embargo"],
            f"  TRAIN    {info['train_start']} .. {info['train_end']}   {info['train_rows']:>7,} rows",
            f"  VAL      {info['val_start']} .. {info['val_end']}   {info['val_rows']:>7,} rows   (early stopping only)",
            f"  TEST     {info['test_start']} .. {info['test_end']}   {info['test_rows']:>7,} rows   (scored once per run)"]


def format_report(abl, info=None):
    """Everything the write-up needs, as text: split, both tables, the difference, per-epoch loss."""
    out = []
    if info:
        out += _split_lines(info) + [""]
    out += ["The test block is scored once per configuration, after all training and early stopping. Nothing is tuned on it.", ""]
    for key, title in (("with_sentiment", "WITH SENTIMENT (7 inputs)"), ("zeroed", "SENTIMENT ZEROED (same 7 inputs, last one 0.0)")):
        res = abl[key]
        out += [f"=== {title} ===", format_table(res["table"])]
        out += [f"note: {n}" for n in res["notes"]] + [f"WARNING: {w}" for w in res["warnings"]]
        out.append("")
    out += ["=== DIFFERENCE (with sentiment minus zeroed; direction in percentage points) ===", _format_delta(abl["delta"]), ""]
    out += abl["summary"] + [""]
    for key, title in (("with_sentiment", "with sentiment"), ("zeroed", "sentiment zeroed")):
        ann = abl[key]["ann"]
        if ann is not None:
            out += [f"=== network training, per epoch ({title}) ===", format_history(ann), ""]
    return "\n".join(out)


def _format_delta(d):
    head = f"{'model':<30}" + "".join(f"{_HEADINGS[c]:>15}" for c in METRICS)
    rows = [head]
    for _, r in d.iterrows():
        rows.append(f"{r['model']:<30}{r['mae']:>+15.5f}{r['rmse']:>+15.5f}{r['direction_acc'] * 100:>+14.1f}p{r['r2']:>+15.4f}")
    return "\n".join(rows)


def save_results(abl, out_dir="results", info=None):
    """Keep both runs for the write-up. Writes (and returns the paths of):
        comparison_with_sentiment.csv, comparison_zeroed.csv, ablation_difference.csv,
        loss_history_with_sentiment.csv, loss_history_zeroed.csv, loss_curves.png, report.txt."""
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def put(name, writer):
        path = os.path.join(out_dir, name)
        writer(path)
        written.append(path)

    put("comparison_with_sentiment.csv", lambda p: abl["with_sentiment"]["table"].to_csv(p, index=False))
    put("comparison_zeroed.csv", lambda p: abl["zeroed"]["table"].to_csv(p, index=False))
    put("ablation_difference.csv", lambda p: abl["delta"].to_csv(p, index=False))
    histories = {}
    for key, name in (("with_sentiment", "loss_history_with_sentiment.csv"), ("zeroed", "loss_history_zeroed.csv")):
        ann = abl[key]["ann"]
        if ann is not None:
            histories[key] = ann
            put(name, lambda p, a=ann: pd.DataFrame(a.history_).to_csv(p, index=False))
    if histories:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            def draw(path):
                fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 3.6), squeeze=False)
                for ax, (key, ann) in zip(axes[0], histories.items()):
                    plot_loss_curves(ax, ann.history_, ann.best_epoch_)
                    ax.set_title("with sentiment" if key == "with_sentiment" else "sentiment zeroed")
                fig.tight_layout()
                fig.savefig(path, dpi=130)
                plt.close(fig)
            put("loss_curves.png", draw)
    put("report.txt", lambda p: open(p, "w", encoding="utf-8").write(format_report(abl, info) + "\n"))
    return written


# ---------------------------------------------------------------------------- CLI
def main(argv=None):
    import argparse

    from data import load_news, load_prices
    from features import build_features, split_by_date
    from sentiment import MODEL_ID, load_qwen_generator, sentiment_for_rows

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Compare baseline, forest and network on the test block, with and without sentiment.")
    ap.add_argument("prices", nargs="?", default=os.path.join(here, "sample_market_A.csv"))
    ap.add_argument("news", nargs="?", default=os.path.join(here, "sample_news.csv"))
    ap.add_argument("--llm", action="store_true", help="score sentiment with Qwen (slow; needs torch + transformers). Default: LLM off.")
    ap.add_argument("--every", type=int, default=5, help="score each stock every N trading days and carry the score forward (default 5)")
    ap.add_argument("--cache", default=os.path.join(here, "sentiment_cache.json"), help="where LLM scores are saved so a run can resume")
    ap.add_argument("--out", default=os.path.join(here, "results"), help="folder for the saved results")
    args = ap.parse_args(argv)

    feats = build_features(load_prices(args.prices), 20)
    news = load_news(args.news) if os.path.exists(args.news) else None
    generate_fn, status = None, "LLM off: sentiment is 0.0 on every row (offline)."
    if args.llm:
        try:
            generate_fn, status = load_qwen_generator(), f"LLM on: {MODEL_ID}"
        except Exception as e:                           # noqa: BLE001
            status = f"LLM unavailable ({type(e).__name__}: {str(e)[:90]}). Sentiment is 0.0 on every row (offline)."
    print(status)
    sent, st = sentiment_for_rows(
        feats, news, generate_fn, every_n_days=args.every, cache_path=args.cache if generate_fn else None,
        progress=lambda done, total, sym: print(f"  scoring sentiment {done}/{total} ({sym})", flush=True))
    if generate_fn is not None:
        print(f"sentiment: {st['llm_calls']} LLM calls, {st['cache_hits']} reused, {st['no_news']} with no headlines, {st['failed']} failed")
    feats["sentiment"] = sent

    train, val, test, info = split_by_date(feats)
    abl = run_ablation(train, val, test, progress=lambda m: print(f"\n{m} ..."))
    print("\n" + format_report(abl, info))
    paths = save_results(abl, args.out, info)
    print(f"saved {len(paths)} files in {args.out}")
    return abl


if __name__ == "__main__":
    main()
