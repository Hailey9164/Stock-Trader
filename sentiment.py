"""LLM sentiment scorer: headlines in, one score in [-1, 1] and a one-line reason out.

The LLM handles words, the code handles numbers. The model only ever sees headlines (no prices,
no dates to calculate with), and its answer is parsed defensively.

    ANY failure -> (0.0, "offline") and the app keeps running:
    model not installed, download failed, generation error, malformed JSON, missing key,
    non-numeric or non-finite score.

Not a failure: a stock with no headlines in the window scores (0.0, "No headlines in window").

One thing is deliberately NOT swallowed: if a headline dated on or after the decision date reaches
this module, AssertionError is raised. A look-ahead bug must be loud, never a quiet 0.0.

The model loading code follows the lab notebook (LLM_Chat_Example_2, Steps 2-4 and 9). It is
imported lazily, so the rest of the app runs on a machine with no torch/transformers at all.
"""
import hashlib
import json
import math
import os
import re
import warnings

import pandas as pd

from data import headlines_before

MODEL_ID = "Qwen/Qwen3.5-0.8B"          # same model as the lab
OFFLINE = (0.0, "offline")
NO_NEWS = (0.0, "No headlines in window")


# ------------------------------------------------------------------------- prompt
def build_prompt(symbol, headlines, decision_date):
    """The exact prompt from the Project Walkthrough. Nothing but the symbol and headlines."""
    day = pd.Timestamp(decision_date).strftime("%Y-%m-%d")
    lines = "\n".join(f' {i}. "{text.replace(chr(34), chr(39))}"' for i, (_, text) in enumerate(headlines, 1))
    return (
        "You score financial news sentiment.\n\n"
        f"Stock: {symbol}\n"
        f"Headlines (all published before {day}):\n"
        f"{lines}\n\n"
        "Return JSON only:\n"
        '{"score": <-1.0 to 1.0>, "reason": "<12 words max>"}'
    )


def assert_before(headlines, decision_date):
    """The look-ahead guard. Raises AssertionError if any headline is on/after the decision date."""
    cutoff = pd.Timestamp(decision_date).normalize()
    late = [(ts, t) for ts, t in headlines if pd.Timestamp(ts) >= cutoff]
    if late:
        raise AssertionError(
            f"LEAK: {len(late)} headline(s) dated on/after the decision date {cutoff.date()} "
            f"reached the sentiment scorer, e.g. {late[0][0]}: {late[0][1]!r}")


# ------------------------------------------------------------------------- parse
def parse_response(text):
    """Pull {"score", "reason"} out of the model's reply. Returns (score clamped to [-1, 1], reason).
    Raises ValueError on anything malformed. Tolerates prose or a ```json fence around the object."""
    decoder = json.JSONDecoder()
    obj = None
    for m in re.finditer(r"\{", text or ""):
        try:
            cand, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and "score" in cand:
            obj = cand
            break
    if obj is None:
        raise ValueError("no JSON object with a 'score' key in the reply")
    raw = obj["score"]
    if isinstance(raw, bool):
        raise ValueError("score is a boolean")
    try:
        score = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"score is not a number: {raw!r}") from None
    if not math.isfinite(score):
        raise ValueError("score is not finite")
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("missing 'reason'")
    return max(-1.0, min(1.0, score)), " ".join(reason.split())[:120]


# ------------------------------------------------------------------------- model
def load_qwen_generator(model_id=MODEL_ID, max_new_tokens=80):
    """Load Qwen the way the lab notebook does and return generate(prompt) -> reply text.

    Greedy decoding (no sampling), so the same headlines always give the same score. Raises
    ImportError / OSError if torch, transformers or the weights are not available; callers treat
    that as "offline"."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    if torch.cuda.is_available():                                   # lab Step 2
        device = "cuda"
        try:
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:                                           # older PyTorch
            native_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if native_bf16 else torch.float32
    else:
        device, dtype = "cpu", torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)             # lab Step 4
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()

    def generate(prompt):                                           # lab Steps 8-9
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return generate


# ------------------------------------------------------------------------- score
def score_sentiment(symbol, headlines, decision_date, generate_fn):
    """Score one stock. headlines = [(Timestamp, text), ...]. generate_fn = prompt -> reply text,
    or None for offline mode. Returns (score, reason); never raises except on a look-ahead leak."""
    assert_before(headlines, decision_date)          # OUTSIDE the try on purpose: leaks stay loud
    if generate_fn is None:
        return OFFLINE
    if not headlines:
        return NO_NEWS
    try:
        return parse_response(generate_fn(build_prompt(symbol, headlines, decision_date)))
    except Exception:                                # noqa: BLE001 - any failure means "offline"
        return OFFLINE


def score_all(symbols, news, decision_date, use_llm=True, generate_fn=None,
              window_days=30, max_headlines=12, progress=None):
    """Score every symbol. Returns (DataFrame[Symbol, sentiment, reason, n_headlines], status_text).

    use_llm=False                     -> every score 0.0, reason "offline" (the LLM-off toggle)
    use_llm=True, generate_fn given   -> use it (tests, or a model you already loaded)
    use_llm=True, generate_fn None    -> try to load Qwen; if that fails, everything is offline
    news=None                         -> no news file loaded: no headlines for anyone
    progress(i, n, symbol) is called after each symbol (for the GUI progress bar)."""
    if news is None:
        news = pd.DataFrame({"Timestamp": pd.Series(dtype="datetime64[ns]"),
                             "Symbol": pd.Series(dtype=str), "Headline": pd.Series(dtype=str)})
    if not use_llm:
        generate_fn, status = None, "LLM off: every sentiment score is 0.0 (offline)"
    elif generate_fn is not None:
        status = "LLM on: custom generator"
    else:
        try:
            generate_fn, status = load_qwen_generator(), f"LLM on: {MODEL_ID}"
        except Exception as e:                       # noqa: BLE001
            generate_fn = None
            status = (f"LLM unavailable ({type(e).__name__}: {str(e)[:80]}). "
                      "Every sentiment score is 0.0 (offline)")

    rows = []
    for i, sym in enumerate(symbols, 1):
        hl = headlines_before(news, sym, decision_date, window_days, max_headlines)
        score, reason = score_sentiment(sym, hl, decision_date, generate_fn)
        rows.append({"Symbol": sym, "sentiment": score, "reason": reason, "n_headlines": len(hl)})
        if progress:
            progress(i, len(symbols), sym)
    out = pd.DataFrame(rows, columns=["Symbol", "sentiment", "reason", "n_headlines"])
    if generate_fn is not None:
        n_failed = int(((out["reason"] == "offline") & (out["n_headlines"] > 0)).sum())
        if n_failed:
            status += f" ({n_failed} of {len(out)} calls failed and fell back to offline)"
    return out, status


# ------------------------------------------------------------ sentiment as a feature
def _load_cache(path):
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass                                        # a damaged cache is just an empty one
    return {}


def _save_cache(path, cache):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)                               # never leave a half-written cache behind


def sentiment_for_rows(feats, news, generate_fn=None, window_days=30, max_headlines=12,
                       every_n_days=5, model_id=MODEL_ID, cache_path=None, progress=None):
    """A sentiment score for EVERY row of `feats` (the 7th network input). Returns (Series, stats).

    Scoring one prompt per row would mean thousands of LLM calls, so this cuts the work three ways:
      * every_n_days: a stock is scored on every n-th trading day and that score is carried forward
        until the next scoring day. The carried score only uses OLDER headlines, so nothing leaks.
      * the same stock with the same headlines in its window is scored once, whatever the date.
      * `cache_path`: scores are saved to a JSON file as they arrive, so an interrupted run resumes
        where it stopped and a re-run costs no LLM calls. Failures are not cached.

    Look-ahead is still impossible: each scored row only sees headlines strictly before its own date
    (headlines_before) and score-time asserts it (assert_before).

    generate_fn=None (LLM off) or no news -> every score is 0.0, with no calls. A stock with no headlines
    in the window scores 0.0. A failed call scores 0.0 and is counted in stats['failed'].
    progress(done, total, symbol) is called every 25 scored rows, with total counting scoring days."""
    if int(every_n_days) < 1:
        raise ValueError("every_n_days must be at least 1.")
    step = int(every_n_days)
    out = pd.Series(0.0, index=feats.index, name="sentiment")
    stats = {"rows": int(len(feats)), "scoring_rows": 0, "no_news": 0, "llm_calls": 0, "cache_hits": 0, "failed": 0}
    if generate_fn is None or news is None or len(news) == 0:
        return out, stats

    cache = _load_cache(cache_path)
    by_symbol = {s: g for s, g in news.groupby("Symbol")}
    counts = feats.groupby("Symbol").size()
    total = int(sum(-(-int(c) // step) for c in counts))        # ceil(c / step)
    fresh = 0

    for sym, rows in feats.groupby("Symbol"):
        rows = rows.sort_values("Date")
        sym_news = by_symbol.get(sym)
        score = 0.0
        for k, (idx, date) in enumerate(zip(rows.index, rows["Date"])):
            if k % step == 0:
                stats["scoring_rows"] += 1
                hl = headlines_before(sym_news, sym, date, window_days, max_headlines) if sym_news is not None else []
                assert_before(hl, date)                          # the look-ahead guard, per scored row
                if not hl:
                    stats["no_news"] += 1
                    score = 0.0
                else:
                    # The key is the stock + its headlines, NOT the prompt text: the prompt also carries the
                    # decision date, which differs on every row, so keying on it would defeat the dedupe.
                    key = hashlib.sha256("\n".join([model_id, sym, *(t for _, t in hl)]).encode("utf-8")).hexdigest()
                    if key in cache:
                        stats["cache_hits"] += 1
                        score = float(cache[key][0])
                    else:
                        try:
                            score, reason = parse_response(generate_fn(build_prompt(sym, hl, date)))
                            cache[key] = [score, reason]
                            stats["llm_calls"] += 1
                            fresh += 1
                            if fresh % 50 == 0:
                                _save_cache(cache_path, cache)
                        except Exception:                       # noqa: BLE001 - failed call: 0.0, not cached
                            stats["failed"] += 1
                            score = 0.0
                if progress and (stats["scoring_rows"] % 25 == 0 or stats["scoring_rows"] == total):
                    progress(stats["scoring_rows"], total, sym)
            out.at[idx] = score
    _save_cache(cache_path, cache)
    return out, stats


if __name__ == "__main__":
    import sys
    from data import load_news, load_prices

    prices_csv = sys.argv[1] if len(sys.argv) > 1 else "sample_market_A.csv"
    news_csv = sys.argv[2] if len(sys.argv) > 2 else "sample_news.csv"
    use_llm = "--llm" in sys.argv
    syms = sorted(load_prices(prices_csv)["Symbol"].unique())
    day = load_prices(prices_csv)["Date"].max()
    table, status = score_all(syms, load_news(news_csv), day, use_llm=use_llm)
    print(status, f"(decision date {day.date()})\n")
    print(table.to_string(index=False))
