"""Plain-English explanation of the finished plan. No Tkinter.

Every number the explainer may use is computed by our code and formatted ONCE, in build_facts(). Both the LLM
prompt and the offline template are written from those same strings, so the two can never disagree.

    explain(facts, generate_fn) -> (text, prompt, source)
        source is "llm" or "template". The raw prompt is always returned so the GUI can show it next to the text.

The LLM's reply is checked: any number in it that is not in the prompt means the model invented it, and the
template is used instead. Any LLM failure or an empty reply also gives the template.
"""
import re

HORIZON_DAYS = 20


def _pct(x):
    return f"{x * 100:+.1f}%"


def _usd(x):
    return f"${x:,.2f}"


def build_facts(ranked, picks, plan, holdings, reasons=None, top_n=None):
    """Collect every number the explanation may mention, already formatted as text.
    ranked/picks: portfolio.score_stocks / pick_top_n output. plan: portfolio.TradePlan.
    holdings: {symbol: shares} before the plan. reasons: {symbol: one-line sentiment reason}."""
    reasons = reasons or {}
    by_sym = ranked.set_index("symbol")
    buys, kept = [], []
    for sym in picks["symbol"]:
        r = by_sym.loc[sym]
        item = dict(symbol=sym, rank=int(r["rank"]), predicted=_pct(r["predicted"]), sentiment=f"{r['sentiment']:+.2f}",
                    reason=reasons.get(sym, ""), shares=plan.allocation.shares.get(sym, 0))
        (buys if item["shares"] else kept).append(item)
    sold = []
    for sym, q in sorted(holdings.items()):
        if q > 0 and plan.goal.get(sym, 0) == 0:
            r = by_sym.loc[sym] if sym in by_sym.index else None
            sold.append(dict(symbol=sym, held=int(q), rank=int(r["rank"]) if r is not None else None,
                             predicted=_pct(r["predicted"]) if r is not None else None,
                             sentiment=f"{r['sentiment']:+.2f}" if r is not None else None,
                             reason=reasons.get(sym, "")))
    return dict(
        n_picked=len(picks), n_ranked=len(ranked), top_n=top_n or len(picks), horizon=HORIZON_DAYS,
        buys=buys, unaffordable=kept, sold=sold,
        n_trades=plan.num_trades, total_fee=_usd(plan.total_fee),
        start_cash=_usd(plan.start_cash), end_cash=_usd(plan.end_cash), per_pick=_usd(plan.allocation.per_pick_cents / 100),
        trades=[f"{r.action} {r.lot} {r.symbol}" for r in plan.rows],
    )


def _stock_line(b):
    s = f"{b['symbol']} (rank {b['rank']}): predicted {b['predicted']}, sentiment {b['sentiment']}, {b['shares']} shares"
    return s + (f", news: {b['reason']}" if b["reason"] and b["reason"] != "offline" else "")


def build_prompt(f):
    """The prompt sent to the LLM: only computed values, and an instruction to add none."""
    lines = [
        "You write a short plain-English summary of a stock trading plan for a student project.",
        "Use ONLY the facts below. Do not add any number, price, percentage or date that is not listed.",
        "Write one paragraph of at most 90 words. Do not give advice or promise results.",
        "",
        f"Stocks chosen (top {f['n_picked']} of {f['n_ranked']} ranked by predicted {f['horizon']}-day return plus news sentiment):",
    ]
    lines += [f"- {_stock_line(b)}" for b in f["buys"]] or ["- none"]
    if f["sold"]:
        lines.append("Positions being sold completely:")
        for s in f["sold"]:
            why = (f"rank {s['rank']}, predicted {s['predicted']}, sentiment {s['sentiment']}" if s["rank"] else "not ranked")
            lines.append(f"- {s['symbol']} ({s['held']} shares held; {why})")
    lines += [
        f"Number of trades: {f['n_trades']}",
        f"Total fees: {f['total_fee']}",
        f"Cash before: {f['start_cash']}; cash after: {f['end_cash']}",
        f"Equal dollar budget per stock: {f['per_pick']}",
    ]
    return "\n".join(lines)


def template_explanation(f):
    """Offline fallback: the same facts, in fixed wording. Always works, needs no model."""
    parts = []
    if f["buys"]:
        items = [f"{b['shares']} shares of {b['symbol']}" for b in f["buys"]]
        names = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
        parts.append(f"This plan buys {names}. These are the {f['n_picked']} stocks with the highest combined score "
                     f"(predicted {f['horizon']}-day return plus news sentiment) out of {f['n_ranked']} ranked, "
                     f"splitting the money equally ({f['per_pick']} each).")
        notes = [f"{b['symbol']}: predicted {b['predicted']}, sentiment {b['sentiment']}"
                 + (f" ({b['reason']})" if b["reason"] and b["reason"] != "offline" else "") for b in f["buys"]]
        parts.append("Scores: " + "; ".join(notes) + ".")
    else:
        parts.append("No stock could be bought with its equal share of the cash.")
    if f["sold"]:
        sells = [f"{s['symbol']} ({s['held']} shares)" for s in f["sold"]]
        parts.append("It sells out of " + (sells[0] if len(sells) == 1 else ", ".join(sells[:-1]) + " and " + sells[-1])
                     + ", which did not make the top picks.")
    if f["unaffordable"]:
        parts.append("Skipped because one share costs more than its equal share: "
                     + ", ".join(u["symbol"] for u in f["unaffordable"]) + ".")
    parts.append(f"In total that is {f['n_trades']} trades with {f['total_fee']} in fees; cash goes from "
                 f"{f['start_cash']} to {f['end_cash']}. These are predictions, not promises.")
    return " ".join(parts)


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def numbers_in(text):
    """Every number in the text, as a normalised string (commas dropped, trailing zeros trimmed)."""
    out = set()
    for m in _NUM.findall(text):
        m = m.replace(",", "").rstrip(".")
        if "." in m:
            m = m.rstrip("0").rstrip(".")
        out.add(m or "0")
    return out


def invented_numbers(prompt, reply):
    """Numbers in the reply that do not appear in the prompt: the model made these up."""
    return sorted(numbers_in(reply) - numbers_in(prompt), key=float)


def explain(facts, generate_fn=None):
    """Returns (text, prompt, source). source: "llm", or "template (reason)" when the LLM was off or rejected."""
    prompt = build_prompt(facts)
    if generate_fn is None:
        return template_explanation(facts), prompt, "template (LLM off)"
    try:
        reply = (generate_fn(prompt) or "").strip()
    except Exception:  # noqa: BLE001  any LLM failure -> offline wording
        return template_explanation(facts), prompt, "template (LLM failed)"
    if not reply:
        return template_explanation(facts), prompt, "template (empty reply)"
    bad = invented_numbers(prompt, reply)
    if bad:
        return template_explanation(facts), prompt, "template (reply had numbers not in the prompt: " + ", ".join(bad) + ")"
    return reply, prompt, "llm"
