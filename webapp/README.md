# Live dashboard

A small web app that shows the matched Kalshi ↔ Polymarket events in a browser.
It's one FastAPI service: it serves the API *and* the dashboard, and it can
re-run the pipeline on a schedule so a hosted copy stays fresh.

It works out of the box with bundled sample data, then automatically switches to
your real results once you've run the pipeline (`python pipeline_all.py`), which
writes to `historical_data/similar_events/`.

## Run it locally

```bash
pip install -r webapp/requirements.txt
python webapp/server.py
# open http://localhost:8000
```

That's enough to browse the data. The **Refresh** button (and the scheduler)
re-run the collection + matching pipeline. There are two modes:

- **lite** (default) — collection + a **TF-IDF** matcher. No PyTorch, so it runs
  on a small/free host. This is what the Refresh button triggers.
- **full** — collection + the **SBERT** matcher (heavier, slightly better on
  paraphrases). Trigger with `POST /api/refresh?mode=full`.

Either way, a refresh needs the data-collection deps + your Kalshi credentials
(and `full` additionally needs torch / sentence-transformers):

```bash
pip install -r requirements.txt          # full set incl. torch (only needed for SBERT)
cp kalshi_pipeline/.env.example kalshi_pipeline/.env   # then add your key
```

## API

| Endpoint | What it returns |
|----------|-----------------|
| `GET /api/matches?min=0.9&q=fed&sort=gap&limit=300` | matched pairs as JSON |
| `GET /api/stats` | totals, source file, last-updated, refresh status |
| `POST /api/refresh?mode=lite` | kicks off a TF-IDF (no-PyTorch) re-run in the background |
| `POST /api/refresh?mode=full` | same, but uses the SBERT matcher |

## Keep it online 24/7

The easiest path is [Render](https://render.com) (free tier works):

1. Make sure this repo is on GitHub (it is: `find-similar-events`).
2. Render → **New + → Blueprint** → pick the repo. It reads `webapp/render.yaml`.
3. Deploy. You'll get a public URL serving the dashboard.

Any host works the same way — the start command is:

```bash
uvicorn server:app --app-dir webapp --host 0.0.0.0 --port $PORT
```

### Auto-refresh on the host
The scheduler is off by default. To enable it once collection deps + Kalshi creds
are in place, set:

- `REFRESH_HOURS=12` — re-run every 12 hours
- `REFRESH_ON_START=1` — also run once at boot
- `REFRESH_MODE=lite` (default) or `full` — which matcher the scheduler uses

The scheduled job defaults to **lite** so it stays light enough for free tiers.
If a refresh can't run (e.g. no credentials), the dashboard just keeps serving
the last good data and shows a small notice.
