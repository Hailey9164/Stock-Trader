"""Trade planner: Python port of the Assignment 1 Java code (BFS over portfolio states).

Ported from Stock.java, TradeAction.java, PortfolioState.java, Goal.java, TradePlanner.java.

One deliberate difference from the Java version: cash is stored as integer CENTS.
Java compares doubles exactly in equals()/hashCode(); in Python, float drift (e.g. 0.1 + 0.2)
would make two identical states look different and break the visited set. Integer cents
make state equality exact. Prices and fees are rounded to the nearest cent when a trade is
applied. No pure-Python float is ever used as part of a state's identity.
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def to_cents(dollars: float) -> int:
    return int(round(float(dollars) * 100))


# --------------------------------------------------------------------------- Stock
@dataclass(frozen=True)
class Stock:
    symbol: str
    price: float
    sell_only: bool = False


# ---------------------------------------------------------------------- TradeAction
BUY = "BUY"
SELL = "SELL"


@dataclass(frozen=True)
class TradeAction:
    type: str          # BUY or SELL
    symbol: str
    lot_size: int

    def __str__(self) -> str:
        return f"{self.type} {self.lot_size} {self.symbol}"


# -------------------------------------------------------------------- PortfolioState
class PortfolioState:
    """Shares per symbol plus cash. Immutable; apply() returns a new state or None."""

    __slots__ = ("_shares", "_cash_cents", "_key")

    def __init__(self, shares: Dict[str, int], cash: float = 0.0, *, cash_cents: Optional[int] = None):
        # Zero-share entries are dropped so {"KO": 0} and {} are the same state.
        self._shares = {s: int(n) for s, n in shares.items() if int(n) != 0}
        self._cash_cents = int(cash_cents) if cash_cents is not None else to_cents(cash)
        self._key = (frozenset(self._shares.items()), self._cash_cents)

    @property
    def shares(self) -> Dict[str, int]:
        return dict(self._shares)

    @property
    def cash(self) -> float:
        return self._cash_cents / 100.0

    @property
    def cash_cents(self) -> int:
        return self._cash_cents

    def get_shares(self, symbol: str) -> int:
        return self._shares.get(symbol, 0)

    def apply(self, action: TradeAction, stock_map: Dict[str, Stock],
              fee_per_trade: float) -> Optional["PortfolioState"]:
        stock = stock_map.get(action.symbol)
        if stock is None:
            return None

        current = self.get_shares(action.symbol)
        lot = action.lot_size
        fee_cents = to_cents(fee_per_trade)
        gross_cents = to_cents(lot * stock.price)

        new_shares = dict(self._shares)
        new_cash = self._cash_cents

        if action.type == BUY:
            if stock.sell_only:
                return None
            cost = gross_cents + fee_cents
            if new_cash < cost:          # cash may never go negative
                return None
            new_cash -= cost
            new_shares[action.symbol] = current + lot
        else:
            if current < lot:
                return None
            new_cash += gross_cents - fee_cents
            new_shares[action.symbol] = current - lot

        return PortfolioState(new_shares, cash_cents=new_cash)

    def __eq__(self, other) -> bool:
        return isinstance(other, PortfolioState) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        return f"PortfolioState(shares={self._shares}, cash={self.cash:.2f})"


# ------------------------------------------------------------------------------ Goal
class Goal:
    """Target shares per symbol with an optional tolerance percentage.

    Only symbols in target_shares are checked, exactly as in the Java version. To make the
    planner SELL a position you no longer want, include it with a target of 0."""

    def __init__(self, target_shares: Dict[str, int], tolerance_percent: float = 0.0):
        self.target_shares = dict(target_shares)
        self.tolerance_percent = float(tolerance_percent)

    def is_satisfied_by(self, state: PortfolioState) -> bool:
        for symbol, target in self.target_shares.items():
            actual = state.get_shares(symbol)
            if self.tolerance_percent == 0:
                if actual != target:
                    return False
            else:
                lower = target * (1.0 - self.tolerance_percent / 100.0)
                upper = target * (1.0 + self.tolerance_percent / 100.0)
                if actual < lower or actual > upper:
                    return False
        return True


# ------------------------------------------------------------------------ TradePlanner
@dataclass
class PlanResult:
    actions: List[TradeAction]
    nodes_expanded: int
    elapsed_millis: int
    final_state: PortfolioState

    @property
    def num_trades(self) -> int:
        return len(self.actions)


def bfs(start: PortfolioState, goal: Goal, stock_map: Dict[str, Stock],
        fee_per_trade: float, lot_sizes: Sequence[int],
        node_budget: int = 200000) -> Optional[PlanResult]:
    """Breadth-first search for the shortest trade sequence from start to a goal state.

    With a flat fee per trade, the fewest trades is also the lowest total fee.
    Returns None if no plan is found within node_budget expansions (or none exists)."""
    t0 = time.time()
    frontier = deque([start])
    parent: Dict[PortfolioState, Tuple[PortfolioState, TradeAction]] = {}
    visited = {start}
    nodes_expanded = 0

    while frontier and nodes_expanded < node_budget:
        current = frontier.popleft()
        nodes_expanded += 1

        if goal.is_satisfied_by(current):
            elapsed = int((time.time() - t0) * 1000)
            return PlanResult(_reconstruct(current, parent), nodes_expanded, elapsed, current)

        for stock in stock_map.values():
            for lot in lot_sizes:
                for kind in (BUY, SELL):
                    action = TradeAction(kind, stock.symbol, lot)
                    nxt = current.apply(action, stock_map, fee_per_trade)
                    if nxt is not None and nxt not in visited:
                        visited.add(nxt)
                        parent[nxt] = (current, action)
                        frontier.append(nxt)
    return None


def _reconstruct(goal_state: PortfolioState,
                 parent: Dict[PortfolioState, Tuple[PortfolioState, TradeAction]]) -> List[TradeAction]:
    actions: List[TradeAction] = []
    cur = goal_state
    while cur in parent:
        prev, action = parent[cur]
        actions.append(action)
        cur = prev
    actions.reverse()
    return actions


# ---------------------------------------------------------------------------- helpers
def trade_log(start: PortfolioState, actions: Sequence[TradeAction],
              stock_map: Dict[str, Stock], fee_per_trade: float):
    """Replay a plan and return [(action, fee, cash_after), ...] so the GUI can show the
    running cash balance next to each trade. Raises if a trade is illegal."""
    log, state = [], start
    for a in actions:
        state = state.apply(a, stock_map, fee_per_trade)
        if state is None:
            raise ValueError(f"Illegal trade in plan: {a}")
        log.append((a, float(fee_per_trade), state.cash))
    return log


def total_fees(actions: Sequence[TradeAction], fee_per_trade: float) -> float:
    """Flat fee per trade, as in the Java GUI: number of trades * fee."""
    return len(actions) * float(fee_per_trade)
