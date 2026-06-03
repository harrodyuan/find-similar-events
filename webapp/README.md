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
re-run the full collection + matching pipeline, which additionally needs the
root deps and your Kalshi credentials:

```bash
pip install -r requirements.txt          # torch / sentence-transformers / etc.
cp kalshi_pipeline/.env.example kalshi_pipeline/.env   # then add your key
```

## API

| Endpoint | What it returns |
|----------|-----------------|
| `GET /api/matches?min=0.9&q=fed&sort=gap&limit=300` | matched pairs as JSON |
| `GET /api/stats` | totals, source file, last-updated, refresh status |
| `POST /api/refresh` | kicks off a pipeline re-run in the background |

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
Re-running the pipeline needs the heavy ML deps and Kalshi creds, so it's off by
default on a light host. To enable it once those are in place, set:

- `REFRESH_HOURS=12` — re-run every 12 hours
- `REFRESH_ON_START=1` — also run once at boot

If a refresh can't run (e.g. no credentials), the dashboard just keeps serving
the last good data and shows a small notice.
