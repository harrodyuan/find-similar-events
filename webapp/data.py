"""Loads the latest matched-events CSV and turns it into clean JSON records
for the dashboard. Falls back to a small bundled sample if no run exists yet."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
SIMILAR_DIR = REPO_ROOT / "historical_data" / "similar_events"
SAMPLE_CSV = WEBAPP_DIR / "sample_matches.csv"

# Simple in-process cache keyed on (path, mtime) so we only re-read on change.
_cache: dict = {"path": None, "mtime": None, "df": None}


def latest_matches_path() -> Path:
    """Newest similar_events_*.csv (SBERT or TF-IDF), or the bundled sample."""
    files = sorted(
        glob.glob(str(SIMILAR_DIR / "similar_events_*.csv")),
        key=lambda p: os.path.getmtime(p),
    )
    if files:
        return Path(files[-1])
    return SAMPLE_CSV


def _to_prob(value, scale: float = 1.0):
    """Coerce a price/probability to a 0..1 float, or None if not usable."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(x):
        return None
    x = x / scale
    if x < 0 or x > 1:
        return None
    return round(x, 4)


def load_dataframe() -> pd.DataFrame:
    path = latest_matches_path()
    if not path.exists():
        return pd.DataFrame()
    mtime = path.stat().st_mtime
    if _cache["path"] == str(path) and _cache["mtime"] == mtime:
        return _cache["df"]
    df = pd.read_csv(path, low_memory=False)
    if "similarity_score" in df.columns:
        df = df.sort_values("similarity_score", ascending=False)
    _cache.update(path=str(path), mtime=mtime, df=df)
    return df


def _record(row: pd.Series) -> dict:
    k_title = str(row.get("kalshi_title_raw", "") or "")
    p_title = str(row.get("poly_title_raw", "") or "")

    # Kalshi mid price is in cents (0-100); fall back to implied probability.
    k_prob = _to_prob(row.get("kalshi_mid_price"), scale=100.0)
    if k_prob is None:
        k_prob = _to_prob(row.get("kalshi_implied_prob"))
    p_prob = _to_prob(row.get("poly_yes_price"))

    gap = None
    if k_prob is not None and p_prob is not None:
        gap = round(abs(k_prob - p_prob), 4)

    def num(v):
        try:
            f = float(v)
            return None if pd.isna(f) else f
        except (TypeError, ValueError):
            return None

    return {
        "kalshi_title": k_title,
        "poly_title": p_title,
        "similarity": round(float(row.get("similarity_score", 0) or 0), 4),
        "kalshi_prob": k_prob,
        "poly_prob": p_prob,
        "gap": gap,
        "kalshi_ticker": str(row.get("kalshi_ticker", "") or ""),
        "poly_condition_id": str(row.get("poly_condition_id", "") or ""),
        "kalshi_volume": num(row.get("kalshi_volume")),
        "kalshi_days_to_exp": num(row.get("kalshi_days_to_exp")),
        # Site-scoped search links always resolve to the right market page.
        "kalshi_url": f"https://www.google.com/search?q={quote_plus('site:kalshi.com ' + k_title)}",
        "poly_url": f"https://www.google.com/search?q={quote_plus('site:polymarket.com ' + p_title)}",
    }


def get_matches(min_similarity: float = 0.9, q: str = "", limit: int = 200,
                sort: str = "similarity") -> list[dict]:
    df = load_dataframe()
    if df.empty:
        return []

    df = df[df["similarity_score"].astype(float) >= float(min_similarity)]

    if q:
        ql = q.lower()
        mask = (
            df.get("kalshi_title_raw", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(ql, na=False)
            | df.get("poly_title_raw", pd.Series("", index=df.index)).astype(str).str.lower().str.contains(ql, na=False)
        )
        df = df[mask]

    records = [_record(r) for _, r in df.head(5000).iterrows()]

    if sort == "gap":
        records.sort(key=lambda r: (r["gap"] is None, -(r["gap"] or 0)))
    else:
        records.sort(key=lambda r: -r["similarity"])

    return records[: int(limit)]


def get_stats() -> dict:
    path = latest_matches_path()
    df = load_dataframe()
    return {
        "total_pairs": int(len(df)),
        "source_file": path.name,
        "is_sample": path == SAMPLE_CSV,
        "last_updated": (
            pd.Timestamp(path.stat().st_mtime, unit="s").isoformat() if path.exists() else None
        ),
    }
