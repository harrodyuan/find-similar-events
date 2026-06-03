# Find Similar Events — Kalshi ↔ Polymarket Market Matcher

[![CI](https://github.com/harrodyuan/find-similar-events/actions/workflows/ci.yml/badge.svg)](https://github.com/harrodyuan/find-similar-events/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Built with Sentence-BERT](https://img.shields.io/badge/NLP-Sentence--BERT-orange.svg)](https://www.sbert.net/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

Prediction markets are funny: the exact same question — same election, same game, same coin flip — often shows up on both [Kalshi](https://kalshi.com) and [Polymarket](https://polymarket.com) at different prices. I built this to find those pairs automatically.

It pulls every open market from both platforms and uses a sentence-embedding model (Sentence-BERT) to match the ones that are actually the *same* event. Think of it as the groundwork for cross-platform analysis / arbitrage research — what you do with the matches is up to you.

One honest note before you get excited: this is for research and learning, not financial advice. A lot of those price gaps quietly disappear once you account for fees, slippage, and how each platform settles — and these markets may be restricted where you live. Trade at your own risk.

## How it works

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  kalshi_pipeline │     │  polymarket_pipeline │     │  similarity_analysis  │
│  collect open    │     │  collect open        │     │  SBERT embeddings +   │
│  markets (auth)  │     │  markets (public)    │     │  cosine similarity    │
└────────┬─────────┘     └──────────┬───────────┘     └───────────┬──────────┘
         │                          │                              │
         └──────────► historical_data/processed_markets ◄──────────┘
                                    │
                                    ▼
                  historical_data/similar_events/*.csv  (matched pairs)
```

1. **Collect Kalshi markets** — paginates all open events (with nested markets) via the authenticated Kalshi API.
2. **Collect Polymarket markets** — pulls all markets from the public Polymarket CLOB API and filters to open ones.
3. **Standardize** — cleans, lemmatizes, and aligns titles/questions into a common format.
4. **Match** — encodes every title with the `all-MiniLM-L6-v2` Sentence-BERT model and computes cosine similarity (in memory-safe chunks) to find pairs above a threshold, with a year-matching guard to avoid cross-year false positives.
5. **Save** — writes a timestamped CSV of matched pairs (with tickers, titles, prices, and similarity scores).

---

## Repository structure

```
find_similar_events/
├── pipeline_all.py            # Orchestrator: runs the entire pipeline end-to-end
├── kalshi_pipeline/           # Kalshi data collection (authenticated)
├── polymarket_pipeline/       # Polymarket data collection (public API)
├── similarity_analysis/       # SBERT cross-platform matching
├── historical_data/           # Generated output (gitignored)
└── requirements.txt
```

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/harrodyuan/find-similar-events.git
cd find-similar-events
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

> The first run downloads the SBERT model (~90 MB) and a couple of small NLTK datasets automatically.

### 2. Configure credentials

**Kalshi (required — the API needs authentication):**

1. In your Kalshi account, go to **Profile → API Keys** and create a key.
2. Download the private key file and save it as `kalshi_pipeline/private_key.pem`.
3. Create your env file:
   ```bash
   cp kalshi_pipeline/.env.example kalshi_pipeline/.env
   # then edit kalshi_pipeline/.env and set KALSHI_KEY_ID
   ```

**Polymarket (optional):** market data is read from the public API. A wallet key is only needed for trading. If you want to set one:
```bash
cp polymarket_pipeline/.env.example polymarket_pipeline/.env
# then edit polymarket_pipeline/.env and set PK
```

Your `.env` and `*.pem` files are gitignored, so they won't get committed by accident — only the `.env.example` templates are. Keep it that way.

### 3. Run

```bash
python pipeline_all.py
```

You can also run any stage independently:
```bash
python kalshi_pipeline/pipeline_all_kalshi.py        # Kalshi only
python polymarket_pipeline/pipeline_all_poly.py      # Polymarket only
cd similarity_analysis && python find_similar_events.py
```

---

## Output

| File | Description |
|------|-------------|
| `historical_data/processed_markets/kalshi_markets_<YYYYMMDD>.csv` | Standardized Kalshi open markets |
| `historical_data/processed_markets/polymarket_markets_<YYYYMMDD>.csv` | Standardized Polymarket open markets |
| `historical_data/similar_events/similar_events_sbert_<YYYYMMDD>.csv` | **Matched pairs** with similarity scores, tickers, titles, and prices |

Key columns in the matches file include `kalshi_ticker`, `poly_condition_id`, `kalshi_title_raw`, `poly_title_raw`, `similarity_score`, and price fields for both platforms.

---

## Examples

Sample collected/processed data is included so you can see the output format before running anything:

- [`kalshi_pipeline/historical_data_example/processed_markets/`](kalshi_pipeline/historical_data_example/processed_markets) — example standardized Kalshi market CSVs
- [`kalshi_pipeline/historical_data_example/processed_events/`](kalshi_pipeline/historical_data_example/processed_events) — example processed Kalshi events
- [`kalshi_pipeline/historical_data_example/open_events/`](kalshi_pipeline/historical_data_example/open_events) — raw open-events snapshots

Here's roughly what a full run looks like:

```text
$ python pipeline_all.py
Running Kalshi pipeline...      collected 6,755 open events -> 56,114 markets
Running Polymarket pipeline...  1.3M markets pulled -> 38,260 open
Encoding texts...
Calculating similarities (chunked)...
Similar events saved to historical_data/similar_events/similar_events_sbert_20260603.csv

Top matches (similarity = 1.000):
  Will Fernando Haddad win the 2026 Brazilian presidential election?   (Kalshi == Polymarket)
  Will François Ruffin win the 2027 French presidential election?      (Kalshi == Polymarket)
  Will Laurence Louie win Top Chef Season 23?                          (Kalshi == Polymarket)
```

The matcher is title-driven, so identical questions land at ~1.0 and close-but-not-identical wordings still rank high — that's where the interesting cross-platform cases tend to hide.

---

## Notes & tuning

- **Similarity threshold:** default is `0.70` in `similarity_analysis/find_similar_events.py`. Raise it (e.g. `0.90`) for stricter, higher-precision matches.
- **Rate limits:** the Kalshi client retries with exponential backoff on HTTP 429, so collection is resilient but can take a few minutes at scale (Kalshi has thousands of open events).
- **Memory:** similarity is computed in chunks, so it runs comfortably even when matching tens of thousands of markets on both sides.
- **Prices:** illiquid markets may have empty bid/ask — that's expected. Titles are always present, which is what matching relies on.

---

## Requirements

- Python 3.9+
- See `requirements.txt` (pandas, numpy, scikit-learn, sentence-transformers, nltk, cryptography, py-clob-client, …)

## Contributing

Happy to take PRs. A few things I'd love help with if you're interested:

- More platforms (PredictIt, Manifold, …)
- Better matching — entity/date extraction, candidate blocking, or trying embeddings beyond MiniLM
- A fee- and slippage-aware "edge" calculator that sits on top of the matched pairs

Open an issue first if it's a big change, otherwise just fork and send it over.

## License

[MIT](LICENSE) — provided as-is, without warranty.

## Author

[@harrodyuan](https://github.com/harrodyuan)
