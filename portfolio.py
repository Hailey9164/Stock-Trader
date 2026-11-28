"""Step 3, first half: score, rank, pick the top N, and allocate cash. No Tkinter, no model code.

    final score = predicted return + weight * sentiment
    top N       = the N highest scores (ties broken by symbol so the result is repeatable)
    allocation  = equal dollars per pick, whole shares only, never more than the cash

Money is handled in integer cents so "total <= cash" is exact (no float drift).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_WEIGHT = 0.02      # a sentiment of +1 is worth +2 percentage points of expected 20-day return. Team's choice.


def _cents(dollars):
    return int(round(float(dollars) * 100))


def score_stocks(predicted, sentiment=None, weight=DEFAULT_WEIGHT):
    """predicted: {symbol: expected return}; sentiment: {symbol: score in [-1, 1]} (missing -> 0.0).
    Returns a DataFrame ranked best first with columns symbol, predicted, sentiment, score, rank."""
    if not predicted:
        raise ValueError("no predictions to rank")
    sentiment = sentiment or {}
    rows = []
    for sym, p in predicted.items():
        p = float(p)
        s = float(sentiment.get(sym, 0.0))
        if not (math.isfinite(p) and math.isfinite(s)):
            raise ValueError(f"{sym}: prediction or sentiment is not a finite number")
        if not -1.0 <= s <= 1.0:
            raise ValueError(f"{sym}: sentiment {s} is outside [-1, 1]")
        rows.append((sym, p, s, p + weight * s))
    df = pd.DataFrame(rows, columns=["symbol", "predicted", "sentiment", "score"])
    df = df.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def pick_top_n(ranked, n, require_positive=False):
    """The n best rows of `ranked`. If fewer than n exist (or fewer are positive), returns what there is."""
    if n < 1:
        raise ValueError("N must be at least 1")
    pool = ranked[ranked["score"] > 0] if require_positive else ranked
    return pool.head(n).reset_index(drop=True)


@dataclass
class Allocation:
    shares: Dict[str, int]                 # whole shares per picked symbol (only symbols that got >= 1 share)
    spent_cents: int
    cash_cents: int
    per_pick_cents: int                    # the equal dollar budget given to each pick
    notes: List[str] = field(default_factory=list)

    @property
    def leftover_cents(self):
        return self.cash_cents - self.spent_cents

    def cost(self, prices):
        return sum(q * _cents(prices[s]) for s, q in self.shares.items())


def allocate_equal(picks, prices, cash):
    """Split `cash` equally across the picks and buy whole shares of each at `prices`.
    picks: DataFrame from pick_top_n (or a list of symbols). prices: {symbol: price}. cash: dollars."""
    symbols = list(picks["symbol"]) if hasattr(picks, "columns") else list(picks)
    if not symbols:
        raise ValueError("nothing to allocate: no stocks were picked")
    cash_c = _cents(cash)
    if cash_c < 0:
        raise ValueError("cash cannot be negative")
    notes: List[str] = []
    per = cash_c // len(symbols)
    shares: Dict[str, int] = {}
    spent = 0
    for sym in symbols:
        if sym not in prices:
            raise ValueError(f"no price for {sym}")
        px = _cents(prices[sym])
        if px <= 0:
            raise ValueError(f"{sym}: price must be positive")
        q = per // px
        if q == 0:
            notes.append(f"{sym}: its share of ${per / 100:,.2f} cannot buy one share at ${px / 100:,.2f}, so it is skipped.")
            continue
        shares[sym] = q
        spent += q * px
    if not shares:
        notes.append("No pick was affordable with its equal share of the cash.")
    assert spent <= cash_c                  # the one rule that must never break
    return Allocation(shares, spent, cash_c, per, notes)


def choose(predicted, sentiment, prices, cash, n, weight=DEFAULT_WEIGHT, require_positive=False):
    """Convenience: score -> top N -> allocate. Returns (ranked, picks, allocation)."""
    ranked = score_stocks(predicted, sentiment, weight)
    picks = pick_top_n(ranked, n, require_positive)
    return ranked, picks, allocate_equal(picks, prices, cash)


def format_ranking(ranked, picks=None, alloc: Optional[Allocation] = None, prices=None):
    chosen = set(picks["symbol"]) if picks is not None else set()
    lines = [f"{'rank':>4}  {'symbol':<7}{'predicted':>10}{'sentiment':>10}{'score':>9}  {'shares':>7}"]
    for r in ranked.itertuples():
        q = alloc.shares.get(r.symbol, 0) if alloc else 0
        mark = "<- picked" if r.symbol in chosen else ""
        lines.append(f"{r.rank:>4}  {r.symbol:<7}{r.predicted:>+10.3f}{r.sentiment:>+10.2f}{r.score:>+9.3f}  {q if q else '':>7} {mark}")
    if alloc:
        lines.append(f"\nspent ${alloc.spent_cents / 100:,.2f} of ${alloc.cash_cents / 100:,.2f} "
                     f"(equal budget ${alloc.per_pick_cents / 100:,.2f} per pick, ${alloc.leftover_cents / 100:,.2f} left over)")
        lines += alloc.notes
    return "\n".join(lines)


if __name__ == "__main__":
    pred = {"AAA": 0.031, "BBB": 0.012, "CCC": -0.004, "DDD": 0.020, "EEE": 0.018}
    sent = {"AAA": -0.9, "BBB": 0.8, "DDD": 0.1}
    px = {"AAA": 120.0, "BBB": 45.5, "CCC": 10.0, "DDD": 310.0, "EEE": 75.25}
    ranked, picks, alloc = choose(pred, sent, px, cash=10000, n=3)
    print(format_ranking(ranked, picks, alloc, px))
