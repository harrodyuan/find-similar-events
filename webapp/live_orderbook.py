"""Live top-of-book helpers for Kalshi + Polymarket (READ-ONLY, public endpoints).

Given a matched pair we want the *current* best bid/ask and size on each leg so
the dashboard can show a live arbitrage edge, not the snapshot captured when the
matcher last ran.

Both endpoints are public (no auth / no private key needed for reading books):
  - Kalshi : GET /trade-api/v2/markets/{ticker}/orderbook
  - Poly   : GET https://clob.polymarket.com/book?token_id=...

Kalshi orderbook semantics:
  The book lists resting *bids*. `yes_dollars` = bids to buy YES, `no_dollars`
  = bids to buy NO, each level [price, size]. To BUY one side you must match the
  opposite side's bids, so the best ASK on a side = 1 - (best bid on the other
  side), and the size available at that ask = the other side's best-bid size.

Note on hosting: Polymarket geoblocks order *placement* from some regions, but
these read endpoints work from the EU host this dashboard is meant to run on.
"""

from typing import Dict, Optional

import requests

KALSHI_BASE = "https://api.elections.kalshi.com"
POLY_CLOB_BASE = "https://clob.polymarket.com"


def _to_dec(value) -> Optional[float]:
    """Parse a price level into a 0-1 decimal. Kalshi cent books give 0-100."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f / 100.0, 4) if f > 1 else round(f, 4)


def _best_bid(levels):
    """Return (price_dec, size) for the highest-priced resting bid."""
    if not levels:
        return (None, None)
    lvl = max(levels, key=lambda x: float(x[0]))
    return (_to_dec(lvl[0]), round(float(lvl[1])))


def kalshi_orderbook_top(ticker: str, timeout: int = 8,
                         session: Optional[requests.Session] = None) -> Dict:
    """Best Yes/No bid+ask prices and sizes for a Kalshi market.

    Returns a dict with keys yes_bid/yes_ask/no_bid/no_ask (0-1 decimals) and
    matching *_size (contracts). Missing sides are None. Network/parse errors
    return {} so callers can degrade gracefully.
    """
    sess = session or requests
    try:
        resp = sess.get(f"{KALSHI_BASE}/trade-api/v2/markets/{ticker}/orderbook",
                        timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()
    except (requests.RequestException, ValueError):
        return {}

    ob = raw.get("orderbook_fp") or raw.get("orderbook") or {}
    yes = ob.get("yes_dollars") or ob.get("yes") or []
    no = ob.get("no_dollars") or ob.get("no") or []

    yb, ybs = _best_bid(yes)
    nb, nbs = _best_bid(no)
    return {
        "yes_bid": yb, "yes_bid_size": ybs,
        "no_bid": nb, "no_bid_size": nbs,
        # ask on one side = complement of the best bid on the other side
        "yes_ask": round(1 - nb, 4) if nb is not None else None,
        "yes_ask_size": nbs,
        "no_ask": round(1 - yb, 4) if yb is not None else None,
        "no_ask_size": ybs,
    }


def poly_book(token_id: str, timeout: int = 8,
              session: Optional[requests.Session] = None) -> Dict:
    """Best bid/ask + sizes for a Polymarket CLOB token (0-1 decimals, shares).

    Errors return {} so callers can degrade gracefully.
    """
    if not token_id:
        return {}
    sess = session or requests
    try:
        resp = sess.get(f"{POLY_CLOB_BASE}/book", params={"token_id": token_id},
                        timeout=timeout)
        resp.raise_for_status()
        d = resp.json()
    except (requests.RequestException, ValueError):
        return {}

    bids = d.get("bids") or []
    asks = d.get("asks") or []
    bb = max(bids, key=lambda x: float(x["price"])) if bids else None
    ba = min(asks, key=lambda x: float(x["price"])) if asks else None
    bid = round(float(bb["price"]), 4) if bb else None
    ask = round(float(ba["price"]), 4) if ba else None
    return {
        "bid": bid, "ask": ask,
        "bid_size": round(float(bb["size"])) if bb else None,
        "ask_size": round(float(ba["size"])) if ba else None,
        "mid": round((bid + ask) / 2, 4) if (bid is not None and ask is not None) else None,
    }
