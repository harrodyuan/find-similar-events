"""Live price-refresh tier for the matched pairs.

The matcher (``pipeline_all.py``) decides *which* Kalshi and Polymarket markets
are the same event. That set changes slowly, so the matcher only needs to run a
few times a day. Prices, however, move constantly, so re-running the whole
collection pipeline just to refresh prices is wasteful.

This module keeps a small in-memory cache of LIVE top-of-book prices for the
highest-similarity matched pairs and refreshes them on a short interval
(default 30 min). For each pair it computes the real two-leg, lock-$1
arbitrage edge so the dashboard can surface genuine opportunities.

Arbitrage logic (a market pays $1 if it resolves YES, $0 if NO):
  * Buy YES on one venue and NO on the other. If the two asks sum to < $1 the
    payoff ($1) beats the cost regardless of outcome -> locked edge = 1 - cost.
  * We check both directions and report the better one, plus the executable
    size (the smaller of the two legs' ask sizes).

All reads use public, read-only endpoints (see ``live_orderbook.py``). No order
placement, no credentials.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import requests

import data as data_module
import live_orderbook as lo

logger = logging.getLogger("live_prices")

# --- tunables (env-overridable so a small host can dial it down) ---
TOP_N = int(os.getenv("LIVE_TOP_N", "120"))          # how many pairs to price
MIN_SIM = float(os.getenv("LIVE_MIN_SIM", "0.9"))    # only price confident matches
INTERVAL = int(os.getenv("LIVE_REFRESH_SECONDS", "1800"))  # 30 min default
WORKERS = int(os.getenv("LIVE_WORKERS", "6"))        # parallel fetchers

_prices: dict[str, dict] = {}
_lock = threading.Lock()
_state = {
    "running": False,
    "last_started": None,
    "last_finished": None,
    "count": 0,
    "errors": 0,
    "interval_seconds": INTERVAL,
    "top_n": TOP_N,
    "min_similarity": MIN_SIM,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None:
        return round((bid + ask) / 2, 4)
    return bid if bid is not None else ask


def _min_size(a, b):
    sizes = [s for s in (a, b) if s is not None]
    return min(sizes) if sizes else None


def _clean_token(value) -> str:
    """CSV token ids can arrive as floats ('1.23e+20') or 'nan'; normalize."""
    s = str(value or "").strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    return s.split(".")[0]


def _selected_pairs() -> list[dict]:
    """Highest-similarity matched pairs that carry the ids we need to price."""
    df = data_module.load_dataframe()
    if df.empty:
        return []
    if "similarity_score" in df.columns:
        df = df[df["similarity_score"].astype(float) >= MIN_SIM]
        df = df.sort_values("similarity_score", ascending=False)

    pairs: list[dict] = []
    seen: set[str] = set()
    for _, r in df.head(5000).iterrows():
        ticker = str(r.get("kalshi_ticker", "") or "").strip()
        yes_tok = _clean_token(r.get("poly_token_yes_id"))
        no_tok = _clean_token(r.get("poly_token_no_id"))
        if not ticker or not yes_tok:
            continue
        key = f"{ticker}|{yes_tok}"
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "key": key,
            "kalshi_ticker": ticker,
            "poly_token_yes_id": yes_tok,
            "poly_token_no_id": no_tok,
            "kalshi_title": str(r.get("kalshi_title_raw", "") or ""),
            "poly_title": str(r.get("poly_title_raw", "") or ""),
            "similarity": round(float(r.get("similarity_score", 0) or 0), 4),
        })
        if len(pairs) >= TOP_N:
            break
    return pairs


def _fetch_pair(pair: dict, session: requests.Session) -> dict:
    k = lo.kalshi_orderbook_top(pair["kalshi_ticker"], timeout=8, session=session)
    p_yes = lo.poly_book(pair["poly_token_yes_id"], timeout=8, session=session)
    p_no = (lo.poly_book(pair["poly_token_no_id"], timeout=8, session=session)
            if pair["poly_token_no_id"] else {})

    k_yes_ask, k_no_ask = k.get("yes_ask"), k.get("no_ask")
    p_yes_ask, p_no_ask = p_yes.get("ask"), p_no.get("ask")
    k_mid = _mid(k.get("yes_bid"), k.get("yes_ask"))
    p_mid = p_yes.get("mid")

    arbs = []
    # Direction 1: buy YES on Kalshi + buy NO on Polymarket.
    if k_yes_ask is not None and p_no_ask is not None:
        arbs.append(("buy YES @Kalshi + NO @Poly", round(1 - (k_yes_ask + p_no_ask), 4),
                     _min_size(k.get("yes_ask_size"), p_no.get("ask_size"))))
    # Direction 2: buy YES on Polymarket + buy NO on Kalshi.
    if p_yes_ask is not None and k_no_ask is not None:
        arbs.append(("buy YES @Poly + NO @Kalshi", round(1 - (p_yes_ask + k_no_ask), 4),
                     _min_size(p_yes.get("ask_size"), k.get("no_ask_size"))))
    best = max(arbs, key=lambda a: a[1]) if arbs else None

    return {
        "key": pair["key"],
        "kalshi_ticker": pair["kalshi_ticker"],
        "poly_token_yes_id": pair["poly_token_yes_id"],
        "kalshi_title": pair["kalshi_title"],
        "poly_title": pair["poly_title"],
        "similarity": pair["similarity"],
        "kalshi_yes_bid": k.get("yes_bid"), "kalshi_yes_ask": k_yes_ask,
        "kalshi_no_ask": k_no_ask, "kalshi_mid": k_mid,
        "poly_yes_bid": p_yes.get("bid"), "poly_yes_ask": p_yes_ask,
        "poly_no_ask": p_no_ask, "poly_mid": p_mid,
        "mid_gap": (round(abs(k_mid - p_mid), 4)
                    if (k_mid is not None and p_mid is not None) else None),
        "arb_edge": best[1] if best else None,
        "arb_direction": best[0] if best else None,
        "arb_size": best[2] if best else None,
        "ok": bool(k or p_yes),
    }


def refresh_once() -> int:
    """Re-price the selected pairs. Returns the number of pairs priced."""
    if _state["running"]:
        return _state["count"]
    _state.update(running=True, last_started=_now())
    errors = 0
    try:
        pairs = _selected_pairs()
        results: dict[str, dict] = {}
        session = requests.Session()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = [ex.submit(_fetch_pair, p, session) for p in pairs]
            for fut in futures:
                try:
                    rec = fut.result()
                except Exception:  # noqa: BLE001 - one bad pair must not kill the cycle
                    errors += 1
                    continue
                if rec.get("ok"):
                    results[rec["key"]] = rec
                else:
                    errors += 1
        with _lock:
            _prices.clear()
            _prices.update(results)
        _state["count"] = len(results)
        _state["errors"] = errors
        logger.info("live refresh: priced %d pairs (%d errors)", len(results), errors)
        return len(results)
    finally:
        _state.update(running=False, last_finished=_now())


def get_live(min_edge: Optional[float] = None, q: str = "",
             sort: str = "edge", limit: int = 200) -> list[dict]:
    with _lock:
        recs = list(_prices.values())

    if min_edge is not None:
        recs = [r for r in recs if (r.get("arb_edge") is not None and r["arb_edge"] >= min_edge)]
    if q:
        ql = q.lower()
        recs = [r for r in recs
                if ql in r["kalshi_title"].lower() or ql in r["poly_title"].lower()]

    if sort == "gap":
        recs.sort(key=lambda r: (r.get("mid_gap") is None, -(r.get("mid_gap") or 0)))
    else:  # edge
        recs.sort(key=lambda r: (r.get("arb_edge") is None, -(r.get("arb_edge") or -1)))

    return recs[: int(limit)]


def status() -> dict:
    with _lock:
        live_count = len(_prices)
    return {**_state, "cached": live_count}


def start_background(interval: Optional[int] = None) -> None:
    """Kick off the periodic refresher in a daemon thread."""
    iv = interval or INTERVAL

    def loop():
        while True:
            try:
                refresh_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("live refresh cycle failed: %s", exc)
            time.sleep(iv)

    threading.Thread(target=loop, daemon=True).start()
