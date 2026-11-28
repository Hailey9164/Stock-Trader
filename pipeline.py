"""Runs every stage in order and reports progress through callbacks. NO Tkinter here: the GUI imports this file,
and so does the command line (`python pipeline.py --help`).

    load -> features -> sentiment -> train -> predict -> rank -> plan -> explain

    result = run_pipeline(PipelineConfig(cash=10000, top_n=3), progress=cb, on_epoch=cb2)

    progress(stage, message, fraction)   stage is one of STAGES, fraction is overall progress 0..1
    on_epoch(epoch, train_loss, val_loss) called after every network epoch; return True to cancel training

LLM off (the default) -> every sentiment score is 0.0 with reason "offline", and the explanation is the template.
LLM on but it cannot load / fails -> the same offline behaviour, with a line saying so in result.notes.
Anything that is a look-ahead bug (a headline on/after the decision date) is NOT swallowed.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import evaluate as ev
import explain as ex
import features as F
import models as M
import portfolio as P
import sentiment as S
from data import data_info, load_news, load_prices, news_info

STAGES = ["load", "features", "sentiment", "train", "predict", "rank", "plan", "explain"]
# share of the progress bar each stage takes (sentiment and training are the slow ones)
_WEIGHTS = {"load": 0.04, "features": 0.04, "sentiment": 0.40, "train": 0.30, "predict": 0.03,
            "rank": 0.02, "plan": 0.10, "explain": 0.07}
_START = {s: sum(_WEIGHTS[t] for t in STAGES[:i]) for i, s in enumerate(STAGES)}
HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PipelineConfig:
    prices_path: str = os.path.join(HERE, "sample_market_A.csv")
    news_path: Optional[str] = os.path.join(HERE, "sample_news.csv")
    cash: float = 10000.0
    holdings: Dict[str, int] = field(default_factory=dict)
    top_n: int = 3
    fee: float = 5.0
    lots: Tuple[int, ...] = P.DEFAULT_LOTS
    weight: float = P.DEFAULT_WEIGHT
    require_positive: bool = False
    use_llm: bool = False                    # LLM off by default: this is what a grader without a GPU gets
    every_n_days: int = 5                    # sentiment is scored every N trading days (see sentiment_for_rows)
    cache_path: Optional[str] = os.path.join(HERE, "sentiment_cache.json")
    decision_date: Optional[str] = None      # default: the last date in the price file
    model: str = "ann"                       # which model's predictions are ranked: "ann" or "forest"
    ann_params: Optional[dict] = None
    horizon: int = 20


@dataclass
class PipelineResult:
    config: PipelineConfig
    decision_date: pd.Timestamp
    data_text: str
    split_text: str
    comparison: dict                         # evaluate.run_comparison output (table, ann history, ...)
    sentiment_table: pd.DataFrame            # per stock at the decision date: sentiment, reason, n_headlines
    sentiment_status: str
    ranked: pd.DataFrame
    picks: pd.DataFrame
    plan: P.TradePlan
    trades_text: str
    explanation: str
    explain_prompt: str
    explain_source: str
    model_used: str
    notes: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def llm_on(self):
        return self.sentiment_status.startswith("LLM on")

    def to_text(self):
        t = ev.format_table(self.comparison["table"])
        parts = [self.data_text, self.split_text, "", "Held-out test scores (scored once):", t]
        for w in self.comparison["warnings"] + self.comparison["notes"]:
            parts.append("! " + w)
        parts += ["", f"Decision date {self.decision_date.date()}   sentiment: {self.sentiment_status}",
                  f"Ranking by the {self.model_used} (final score = predicted + {self.config.weight} x sentiment):",
                  P.format_ranking(self.ranked, self.picks, self.plan.allocation),
                  "", f"Plan ({self.plan.method}):", self.trades_text]
        parts += ["! " + n for n in self.notes + self.plan.notes]
        parts += ["", f"Explanation [{self.explain_source}]:", self.explanation, "", "--- prompt sent to the explainer ---",
                  self.explain_prompt]
        return "\n".join(parts)


def _noop(*_a, **_k):
    return None


def run_pipeline(cfg: PipelineConfig, progress: Optional[Callable] = None, on_epoch: Optional[Callable] = None,
                 generate_fn: Optional[Callable] = None) -> PipelineResult:
    """Run all stages. `generate_fn` (prompt -> text) replaces the Qwen model; used by tests. When it is None and
    cfg.use_llm is True, Qwen is loaded once and shared by the sentiment scorer and the explainer."""
    progress = progress or _noop
    notes: List[str] = []
    timings: Dict[str, float] = {}
    clock = {"t": time.time()}

    def at(stage, message, frac=0.0):
        progress(stage, message, min(1.0, _START[stage] + _WEIGHTS[stage] * frac))

    def done(stage):
        now = time.time()
        timings[stage] = now - clock["t"]
        clock["t"] = now

    if cfg.model not in ("ann", "forest"):
        raise ValueError("model must be 'ann' or 'forest'")
    if cfg.cash < 0 or cfg.top_n < 1:
        raise ValueError("cash must be >= 0 and top_n >= 1")

    # ---------------------------------------------------------------- 1. load
    at("load", "Loading prices and news")
    prices = load_prices(cfg.prices_path)
    if cfg.decision_date is not None:                   # a past decision date: the whole run sees only the past
        prices = prices[prices["Date"] <= pd.Timestamp(cfg.decision_date)]
    if prices.empty:
        raise ValueError("no price rows on or before the decision date")
    decision_date = prices["Date"].max()
    news = load_news(cfg.news_path) if cfg.news_path and os.path.exists(cfg.news_path) else None
    data_text = "PRICES  " + str(data_info(prices)) + ("\nNEWS    " + str(news_info(news)) if news is not None
                                                      else "\nNEWS    none loaded (sentiment will be 0.0)")
    if news is None:
        notes.append("No news file loaded: sentiment is 0.0 for every stock.")
    done("load")

    # ------------------------------------------------------------- 2. features
    at("features", "Building features and the forward-return label")
    feats = F.build_features(prices, cfg.horizon)
    done("features")

    # ------------------------------------------------------------ 3. sentiment
    at("sentiment", "Preparing sentiment scorer")
    gen, status = generate_fn, ("LLM on: custom generator" if generate_fn is not None else "LLM off: every sentiment score is 0.0 (offline)")
    if generate_fn is None and cfg.use_llm:
        try:
            gen = S.load_qwen_generator(max_new_tokens=200)     # room for the explainer paragraph
            status = f"LLM on: {S.MODEL_ID}"
        except Exception as e:                                  # noqa: BLE001
            status = f"LLM unavailable ({type(e).__name__}: {str(e)[:90]}). Sentiment is 0.0 on every row (offline)."
    use_gen = gen if (cfg.use_llm or generate_fn is not None) else None

    def sent_progress(i, total, sym):
        at("sentiment", f"Scoring news sentiment {i}/{total} ({sym})", 0.9 * i / max(total, 1))

    sent_col, st = S.sentiment_for_rows(feats, news, use_gen, every_n_days=cfg.every_n_days,
                                        cache_path=cfg.cache_path if use_gen is not None else None,
                                        progress=sent_progress)
    feats["sentiment"] = sent_col
    symbols = sorted(feats["Symbol"].unique())
    at("sentiment", "Scoring each stock on the decision date", 0.92)
    sent_table, status2 = S.score_all(symbols, news, decision_date, use_llm=use_gen is not None, generate_fn=use_gen)
    if use_gen is not None:
        n_bad = int(st["failed"]) + int(((sent_table["reason"] == "offline") & (sent_table["n_headlines"] > 0)).sum())
        if n_bad:
            notes.append(f"{n_bad} sentiment call(s) failed and were scored 0.0 (offline).")
    done("sentiment")

    # --------------------------------------------------------------- 4. train
    at("train", "Splitting by date and training the models")
    labelled_feats = feats
    train, val, test, info = F.split_by_date(labelled_feats, horizon=cfg.horizon)
    split_text = "\n".join(ev._split_lines(info))
    cols = F.FEATURES + ["sentiment"]
    n_epochs_seen = {"n": 0}

    def epoch_cb(epoch, tr, va):
        n_epochs_seen["n"] = epoch
        at("train", f"Network epoch {epoch}: train {tr:.5f}  val {va:.5f}", 1.0 - 1.0 / (1 + epoch / 10))
        return bool(on_epoch(epoch, tr, va)) if on_epoch else False

    comp = ev.run_comparison(train, val, test, cols, ann_params=cfg.ann_params, on_epoch=epoch_cb)
    done("train")

    # ------------------------------------------------------------- 5. predict
    at("predict", "Predicting each stock's next-20-day return")
    dec = feats[feats["Date"] == decision_date].copy()
    missing = sorted(set(symbols) - set(dec["Symbol"]))
    if missing:
        notes.append("Not enough history to score: " + ", ".join(missing))
    if dec.empty:
        raise ValueError("no stock has a feature row on the decision date")
    dec = dec.drop(columns="sentiment").merge(sent_table[["Symbol", "sentiment"]], on="Symbol", how="left")
    dec["sentiment"] = dec["sentiment"].fillna(0.0)              # the fresh decision-date score replaces the carried one
    dec_n = F.apply_scaler(dec, comp["mu"], comp["sigma"])
    model_name = cfg.model
    if model_name == "ann" and comp["models"]["ann"] is None:
        model_name = "forest"
        notes.append("The neural network is not available (PyTorch missing); ranking with the random forest instead.")
    model = comp["models"][model_name]
    preds = model.predict(M.to_x(dec_n, cols))
    predicted = dict(zip(dec["Symbol"], map(float, preds)))
    px = {r.Symbol: float(r.Close) for r in dec.itertuples()}
    done("predict")

    # ---------------------------------------------------------------- 6. rank
    at("rank", "Ranking and picking")
    sent_map = dict(zip(sent_table["Symbol"], sent_table["sentiment"]))
    reasons = dict(zip(sent_table["Symbol"], sent_table["reason"]))
    ranked = P.score_stocks(predicted, sent_map, cfg.weight)
    picks = P.pick_top_n(ranked, cfg.top_n, cfg.require_positive)
    if len(picks) < cfg.top_n:
        notes.append(f"Only {len(picks)} stock(s) available for top {cfg.top_n}.")
    if picks.empty:
        raise ValueError("no stock qualified (none has a positive final score); try without 'require positive'")
    done("rank")

    # ----------------------------------------------------------------- 7. plan
    at("plan", "Building the trade plan")
    on_day = prices[prices["Date"] == decision_date].set_index("Symbol")["Close"]
    hold_px = {**{s: float(p) for s, p in on_day.items()}, **px}       # every stock priced on the decision date
    unknown = sorted(s for s, q in cfg.holdings.items() if q and s not in hold_px)
    if unknown:
        raise ValueError("holdings not in the price data on " + str(decision_date.date()) + ": " + ", ".join(unknown)
                         + ". Known symbols: " + ", ".join(sorted(hold_px)))
    plan = P.plan_trades(cfg.holdings, cfg.cash, picks, hold_px, cfg.fee, cfg.lots)
    trades_text = P.format_trades(plan)
    done("plan")

    # -------------------------------------------------------------- 8. explain
    at("explain", "Writing the explanation")
    facts = ex.build_facts(ranked, picks, plan, {s: q for s, q in cfg.holdings.items() if q}, reasons, cfg.top_n)
    text, prompt, source = ex.explain(facts, use_gen)
    done("explain")
    at("explain", "Done", 1.0)

    return PipelineResult(cfg, decision_date, data_text, split_text, comp, sent_table, status, ranked, picks, plan,
                          trades_text, text, prompt, source, model_name, notes, timings)


# ------------------------------------------------------------------------------ command line
def _parse_holdings(text):
    out = {}
    for part in (text or "").split(","):
        if part.strip():
            sym, _, q = part.partition("=")
            out[sym.strip().upper()] = int(q)
    return out


def main(argv=None, generate_fn=None):
    import argparse

    ap = argparse.ArgumentParser(description="Run the whole stock-trader pipeline from the command line.")
    ap.add_argument("prices", nargs="?", default=PipelineConfig.prices_path)
    ap.add_argument("news", nargs="?", default=PipelineConfig.news_path)
    ap.add_argument("--cash", type=float, default=10000.0)
    ap.add_argument("--holdings", default="", help="e.g. ALFA=100,BETA=50")
    ap.add_argument("--top", type=int, default=3, help="how many stocks to buy (top N)")
    ap.add_argument("--fee", type=float, default=5.0)
    ap.add_argument("--lots", default="1,10,100")
    ap.add_argument("--weight", type=float, default=P.DEFAULT_WEIGHT, help="sentiment weight in the final score")
    ap.add_argument("--llm", action="store_true", help="use the Qwen model for sentiment and the explanation (default: off)")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--cache", default=PipelineConfig.cache_path)
    ap.add_argument("--date", default=None, help="decision date (default: last date in the price file)")
    ap.add_argument("--model", choices=["ann", "forest"], default="ann")
    ap.add_argument("--epochs", type=int, default=None, help="cap network epochs (quick tests)")
    a = ap.parse_args(argv)

    cfg = PipelineConfig(
        prices_path=a.prices, news_path=a.news, cash=a.cash, holdings=_parse_holdings(a.holdings), top_n=a.top, fee=a.fee,
        lots=tuple(int(x) for x in a.lots.split(",")), weight=a.weight, use_llm=a.llm, every_n_days=a.every,
        cache_path=a.cache, decision_date=a.date, model=a.model,
        ann_params={"epochs": a.epochs} if a.epochs else None)
    last = {"stage": None}

    def progress(stage, message, frac):
        if stage != last["stage"] or stage == "sentiment":
            if stage != last["stage"]:
                print(f"[{frac * 100:3.0f}%] {stage:<9} {message}", flush=True)
            last["stage"] = stage

    res = run_pipeline(cfg, progress, generate_fn=generate_fn)
    print("\n" + res.to_text())
    return res


if __name__ == "__main__":
    main()
