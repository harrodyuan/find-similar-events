"""FastAPI service for the Kalshi <-> Polymarket matcher dashboard.

Serves the matched pairs as JSON, ships a single-page dashboard, and can
re-run the pipeline on a schedule so a hosted instance stays fresh.

Run locally:
    pip install -r webapp/requirements.txt
    python webapp/server.py
    # open http://localhost:8000

Env vars:
    PORT                 port to bind (default 8000)
    REFRESH_HOURS        auto-refresh interval in hours (default 12, 0 = off)
    REFRESH_ON_START     run the pipeline once on startup (default 0)
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
_refresh_state = {"running": False, "last_started": None, "last_finished": None,
                  "last_status": "idle", "last_error": None}


def _run_pipeline() -> None:
    """Re-run the full collection + matching pipeline in a background thread."""
    if not _refresh_lock.acquire(blocking=False):
        return  # a refresh is already in progress
    try:
        _refresh_state.update(running=True, last_status="running",
                              last_started=datetime.now(timezone.utc).isoformat(),
                              last_error=None)
        proc = subprocess.run(
            [sys.executable, "pipeline_all.py"],
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


def _scheduler_loop(interval_hours: float) -> None:
    while True:
        time.sleep(interval_hours * 3600)
        _run_pipeline()


@app.on_event("startup")
def _startup() -> None:
    if os.getenv("REFRESH_ON_START", "0") == "1":
        threading.Thread(target=_run_pipeline, daemon=True).start()
    hours = float(os.getenv("REFRESH_HOURS", "12") or 0)
    if hours > 0:
        threading.Thread(target=_scheduler_loop, args=(hours,), daemon=True).start()


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
def api_refresh():
    if _refresh_state["running"]:
        return JSONResponse({"ok": False, "message": "A refresh is already running."}, status_code=409)
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return {"ok": True, "message": "Pipeline refresh started."}


# ---- static dashboard ------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")),
                reload=False)
