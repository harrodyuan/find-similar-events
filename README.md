# Find Similar Events — Kalshi ↔ Polymarket Market Matcher

Find the **same real-world event listed on both [Kalshi](https://kalshi.com) and [Polymarket](https://polymarket.com)**. Prediction markets often list identical questions (elections, sports, economics, crypto) on different platforms at *different prices*. This tool automatically collects every open market from both platforms and uses semantic similarity (Sentence-BERT) to match them up — the starting point for cross-platform analysis and arbitrage research.

> **Example output:** a single run matched markets like *"Will Fernando Haddad win the 2026 Brazilian presidential election?"* on Kalshi to the identical Polymarket question with a 1.000 similarity score.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not** financial advice. Prediction markets carry risk, may be restricted in your jurisdiction, and price differences are not guaranteed to be exploitable after fees, slippage, and settlement rules. Use at your own risk. You are responsible for complying with each platform's Terms of Service.

---

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

> 🔒 **Security:** `.env` files and `*.pem` keys are gitignored and must **never** be committed. Only the `.env.example` templates are tracked.

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

A real run produces near-perfect matches, for example:

| similarity | Kalshi | Polymarket |
|-----------|--------|------------|
| 1.000 | Will Fernando Haddad win the 2026 Brazilian presidential election? | Will Fernando Haddad win the 2026 Brazilian presidential election? |
| 1.000 | Will François Ruffin win the 2027 French presidential election? | Will François Ruffin win the 2027 French presidential election? |
| 1.000 | Will Laurence Louie win Top Chef Season 23? | Will Laurence Louie win Top Chef Season 23? |

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

## License

[MIT](LICENSE) — provided as-is, without warranty.

## Author

[@harrodyuan](https://github.com/harrodyuan)
