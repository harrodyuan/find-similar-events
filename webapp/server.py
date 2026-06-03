"""FastAPI service for the Kalshi <-> Polymarket matcher dashboard.

Serves the matched pairs as JSON, ships a single-page dashboard, and can
re-run the pipeline on a schedule so a hosted instance stays fresh.

Run locally:
    pip install -r webapp/requirements.txt
    python webapp/server.py
    # open http://localhost:8000

Env vars:
    PORT                 port to bind (default 8000)
    REFRESH_HOURS        matcher auto-refresh interval in hours (default 12, 0 = off)
    REFRESH_ON_START     run the matcher pipeline once on startup (default 0)
    LIVE_REFRESH_SECONDS live price-refresh interval (default 1800; 0 = off)
    LIVE_TOP_N           how many top-similarity pairs to live-price (default 120)
    LIVE_MIN_SIM         min similarity to live-price a pair (default 0.9)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import data as data_module
import live_prices

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
STATIC_DIR = WEBAPP_DIR / "static"

app = FastAPI(title="Kalshi x Polymarket Matcher", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- pipeline refresh state ------------------------------------------------
_refresh_lock = threading.Lock()
_refresh_state = {"running": False, "mode": None, "last_started": None,
                  "last_finished": None, "last_status": "idle", "last_error": None}


def _run_pipeline(lite: bool = True) -> None:
    """Re-run the collection + matching pipeline in a background thread.

    lite=True uses the TF-IDF matcher (no PyTorch) so it works on a small host.
    lite=False uses the heavier SBERT matcher.
    """
    if not _refresh_lock.acquire(blocking=False):
        return  # a refresh is already in progress
    try:
        cmd = [sys.executable, "pipeline_all.py"] + (["--lite"] if lite else [])
        _refresh_state.update(running=True, last_status="running",
                              mode="lite" if lite else "full",
                              last_started=datetime.now(timezone.utc).isoformat(),
                              last_error=None)
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=60 * 90,
        )
        if proc.returncode == 0:
            _refresh_state["last_status"] = "success"
        else:
            _refresh_state["last_status"] = "failed"
            _refresh_state["last_error"] = (proc.stderr or proc.stdout or "")[-2000:]
    except Exception as exc:  # noqa: BLE001
        _refresh_state["last_status"] = "failed"
        _refresh_state["last_error"] = str(exc)
    finally:
        _refresh_state.update(running=False,
                              last_finished=datetime.now(timezone.utc).isoformat())
        _refresh_lock.release()


def _scheduler_loop(interval_hours: float, lite: bool = True) -> None:
    while True:
        time.sleep(interval_hours * 3600)
        _run_pipeline(lite=lite)


def _refresh_is_lite() -> bool:
    # Default to the light TF-IDF path; set REFRESH_MODE=full for SBERT.
    return os.getenv("REFRESH_MODE", "lite").lower() != "full"


@app.on_event("startup")
def _startup() -> None:
    if os.getenv("REFRESH_ON_START", "0") == "1":
        threading.Thread(target=_run_pipeline, kwargs={"lite": _refresh_is_lite()}, daemon=True).start()
    hours = float(os.getenv("REFRESH_HOURS", "12") or 0)
    if hours > 0:
        threading.Thread(target=_scheduler_loop, args=(hours, _refresh_is_lite()), daemon=True).start()

    # Live price tier: re-prices the top matched pairs on a short interval so the
    # dashboard shows a fresh arbitrage edge without re-running the matcher.
    if int(os.getenv("LIVE_REFRESH_SECONDS", "1800") or 0) > 0:
        # Prime the cache once immediately, then refresh on the interval.
        threading.Thread(target=live_prices.refresh_once, daemon=True).start()
        live_prices.start_background()


# ---- API -------------------------------------------------------------------
@app.get("/api/matches")
def api_matches(
    min: float = Query(0.9, ge=0.0, le=1.0),
    q: str = Query(""),
    limit: int = Query(200, ge=1, le=2000),
    sort: str = Query("similarity"),
):
    return data_module.get_matches(min_similarity=min, q=q, limit=limit, sort=sort)


@app.get("/api/stats")
def api_stats():
    stats = data_module.get_stats()
    stats["refresh"] = _refresh_state
    return stats


@app.post("/api/refresh")
def api_refresh(mode: str = Query("lite")):
    if _refresh_state["running"]:
        return JSONResponse({"ok": False, "message": "A refresh is already running."}, status_code=409)
    lite = mode.lower() != "full"
    threading.Thread(target=_run_pipeline, kwargs={"lite": lite}, daemon=True).start()
    return {"ok": True, "message": f"Pipeline refresh started ({'lite/TF-IDF' if lite else 'full/SBERT'})."}


# ---- live prices (top matched pairs, refreshed on a short interval) ---------
@app.get("/api/live")
def api_live(
    min_edge: float = Query(None),
    q: str = Query(""),
    sort: str = Query("edge"),
    limit: int = Query(200, ge=1, le=2000),
):
    return {"status": live_prices.status(),
            "items": live_prices.get_live(min_edge=min_edge, q=q, sort=sort, limit=limit)}


@app.get("/api/live/status")
def api_live_status():
    return live_prices.status()


@app.post("/api/live/refresh")
def api_live_refresh():
    if live_prices.status()["running"]:
        return JSONResponse({"ok": False, "message": "A live refresh is already running."}, status_code=409)
    threading.Thread(target=live_prices.refresh_once, daemon=True).start()
    return {"ok": True, "message": "Live price refresh started."}


# ---- static dashboard ------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")),
                reload=False)
