"""Step 3, first half: score, rank, pick the top N, and allocate cash. No Tkinter, no model code.

    final score = predicted return + weight * sentiment
    top N       = the N highest scores (ties broken by symbol so the result is repeatable)
    allocation  = equal dollars per pick, whole shares only, never more than the cash

Money is handled in integer cents so "total <= cash" is exact (no float drift).
"""
import math
import time
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


# =====================================================================================
# Step 3, second half: goal map -> Assignment 1 planner -> running cash balance
# =====================================================================================
from planner import BUY, SELL, Goal, PlanResult, PortfolioState, Stock, TradeAction, bfs  # noqa: E402

DEFAULT_LOTS = (1, 10, 100)          # the Assignment 1 GUI default
DEFAULT_NODE_BUDGET = 50000


class PlanError(ValueError):
    """The goal cannot be reached (lot sizes, cash, or sell-only stocks)."""


def build_goal(holdings, allocation):
    """Goal map: every picked symbol -> its whole-share target, and every held-but-unpicked symbol -> 0.
    The planner only checks symbols that are in the goal, so the zeros are what make it SELL them."""
    goal = {s: 0 for s, q in holdings.items() if q > 0}
    goal.update(allocation.shares)
    return goal


def min_lot_counts(amount, lots):
    """Fewest lots (from `lots`, reusable) that add up to `amount`, as a list, or None if impossible."""
    best = [None] * (amount + 1)
    best[0] = []
    for a in range(1, amount + 1):
        for lot in lots:
            if lot <= a and best[a - lot] is not None and (best[a] is None or len(best[a - lot]) + 1 < len(best[a])):
                best[a] = best[a - lot] + [lot]
    return None if best[amount] is None else sorted(best[amount], reverse=True)


def direct_plan(start, goal, stock_map, fee, lots):
    """Fewest-trade plan without a search: per symbol, the fewest lots that cover the change; all sells first
    (they only add cash), then the buys. Returns a PlanResult, or None if it is impossible or goes below zero cash.
    With a flat fee this has the same trade count as the BFS whenever the BFS finishes."""
    t0 = time.time()
    sells, buys = [], []
    for sym in sorted(goal):
        delta = goal[sym] - start.get_shares(sym)
        if delta == 0:
            continue
        counts = min_lot_counts(abs(delta), lots)
        if counts is None:
            return None
        (buys if delta > 0 else sells).extend(TradeAction(BUY if delta > 0 else SELL, sym, c) for c in counts)
    state, actions = start, sells + buys
    for a in actions:
        state = state.apply(a, stock_map, fee)
        if state is None:
            return None
    return PlanResult(actions, 0, int((time.time() - t0) * 1000), state)


@dataclass
class TradeRow:
    step: int
    action: str            # "BUY" / "SELL"
    symbol: str
    lot: int
    price: float
    gross: float           # lot * price
    fee: float
    cash_after: float      # running balance after this trade


@dataclass
class TradePlan:
    goal: Dict[str, int]
    allocation: Allocation
    rows: List[TradeRow]
    start_cash: float
    end_cash: float
    total_fee: float
    method: str            # "bfs" (Assignment 1 planner) or "direct"
    nodes_expanded: int
    budget: float          # dollars given to the allocator
    notes: List[str] = field(default_factory=list)

    @property
    def num_trades(self):
        return len(self.rows)


def plan_trades(holdings, cash, picks, prices, fee, lots=DEFAULT_LOTS, node_budget=DEFAULT_NODE_BUDGET):
    """Allocate, build the goal map, call the Assignment 1 BFS planner, and return a TradePlan with a running
    cash balance for every trade.

    Budget = cash + market value of the current holdings (everything could be sold). Because each trade costs a
    fee, the budget is lowered by the fees of the plan until the plan is affordable."""
    holdings = {s: int(q) for s, q in holdings.items() if q}
    symbols = list(picks["symbol"]) if hasattr(picks, "columns") else list(picks)
    for s in set(holdings) | set(symbols):
        if s not in prices:
            raise PlanError(f"no price for {s}")
    stock_map = {s: Stock(s, float(prices[s])) for s in set(holdings) | set(symbols)}
    start = PortfolioState(holdings, cash)
    total_c = start.cash_cents + sum(q * _cents(prices[s]) for s, q in holdings.items())
    fee_c = _cents(fee)
    lots = sorted({int(l) for l in lots})

    reserve_c, notes, alloc, goal, direct = 0, [], None, None, None
    for _ in range(6):
        alloc = allocate_equal(symbols, prices, max(total_c - reserve_c, 0) / 100)
        goal = build_goal(holdings, alloc)
        direct = direct_plan(start, goal, stock_map, fee, lots)
        if direct is not None:
            break
        counts = [min_lot_counts(abs(goal[s] - start.get_shares(s)), lots) for s in goal]
        if any(c is None for c in counts):
            raise PlanError(f"lot sizes {lots} cannot make the changes needed for {goal}")
        reserve_c += fee_c * max(1, sum(len(c) for c in counts))
    else:
        raise PlanError("no legal plan: cash is not enough to cover the trades and their fees")
    if reserve_c:
        notes.append(f"${reserve_c / 100:,.2f} of the budget was held back to pay trading fees.")
    notes += alloc.notes

    result, method = bfs(start, Goal(goal), stock_map, fee, lots, node_budget), "bfs"
    if result is None or result.num_trades > direct.num_trades:
        result, method = direct, "direct"
        notes.append(f"The BFS search ran out of its {node_budget:,}-node budget, so the plan was built directly "
                     f"(same fewest-trades rule).")

    rows, state = [], start
    for i, a in enumerate(result.actions, 1):
        state = state.apply(a, stock_map, fee)
        assert state is not None and state.cash_cents >= 0          # cash never negative
        px = float(prices[a.symbol])
        rows.append(TradeRow(i, a.type, a.symbol, a.lot_size, px, a.lot_size * px, float(fee), state.cash))
    assert Goal(goal).is_satisfied_by(state)
    return TradePlan(goal, alloc, rows, start.cash, state.cash, len(rows) * float(fee), method,
                     result.nodes_expanded, max(total_c - reserve_c, 0) / 100, notes)


def format_trades(plan):
    lines = [f"{'#':>2}  {'trade':<16}{'fee':>8}{'cash after':>14}"]
    for r in plan.rows:
        lines.append(f"{r.step:>2}  {r.action + ' ' + str(r.lot) + ' ' + r.symbol:<16}{r.fee:>8.2f}{r.cash_after:>14,.2f}")
    lines.append(f"{plan.num_trades} trades, total fees ${plan.total_fee:,.2f}   "
                 f"(cash ${plan.start_cash:,.2f} -> ${plan.end_cash:,.2f})")
    return "\n".join(lines)
