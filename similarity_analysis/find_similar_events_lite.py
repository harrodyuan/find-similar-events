"""Lightweight matcher: same inputs/outputs as find_similar_events.py, but uses
TF-IDF (scikit-learn) instead of Sentence-BERT. No PyTorch, so it runs on a
small/free host. Identical titles still score ~1.0; paraphrases score a bit
lower than the SBERT version.

Writes: ../historical_data/similar_events/similar_events_tfidf_<YYYYMMDD>.csv
(same columns as the SBERT output, so the dashboard reads it transparently)."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple
from datetime import datetime
import re
import nltk
from nltk.stem import WordNetLemmatizer
import logging

logging.basicConfig(format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('wordnet')
nltk.download('stopwords')


# --- Text preprocessing (identical to the SBERT pipeline) ---

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    return text.lower().strip()


def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(w) for w in text.split())


def standardize_time(text: str, current_year: int) -> str:
    text = re.sub(r'\bthis year\b', f'in {current_year}', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnext year\b', f'in {current_year + 1}', text, flags=re.IGNORECASE)
    return text


def extract_years(text: str) -> set:
    return set(re.findall(r'\b20\d{2}\b', text))


def load_data(date_suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current_year = datetime.now().year

    kalshi_df = pd.read_csv(f'../historical_data/processed_markets/kalshi_markets_{date_suffix}.csv')
    kalshi_df = kalshi_df.dropna(subset=['title'])

    poly_df = pd.read_csv(f'../historical_data/processed_markets/polymarket_markets_{date_suffix}.csv')
    poly_df = poly_df.dropna(subset=['question'])

    kalshi_df['combined_title'] = kalshi_df['title'].fillna('')
    kalshi_df['combined_title'] = kalshi_df['combined_title'].apply(
        lambda x: standardize_time(x, current_year)).apply(clean_text).apply(lemmatize_text)

    poly_df['combined_title'] = poly_df['question'].fillna('')
    poly_df['combined_title'] = poly_df['combined_title'].apply(
        lambda x: standardize_time(x, current_year)).apply(clean_text).apply(lemmatize_text)

    kalshi_df['years'] = kalshi_df['combined_title'].apply(extract_years)
    poly_df['years'] = poly_df['combined_title'].apply(extract_years)

    return kalshi_df, poly_df


def find_similar_events(kalshi_df: pd.DataFrame, poly_df: pd.DataFrame,
                        similarity_threshold: float = 0.7) -> pd.DataFrame:
    similar_events = []

    logger.info("Vectorizing titles (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english')
    kalshi_texts = kalshi_df['combined_title'].fillna('').astype(str).tolist()
    poly_texts = poly_df['combined_title'].fillna('').astype(str).tolist()
    vectorizer.fit(kalshi_texts + poly_texts)
    kalshi_vectors = vectorizer.transform(kalshi_texts)
    poly_vectors = vectorizer.transform(poly_texts)

    logger.info("Calculating similarities (chunked)...")
    chunk_size = 512
    n_kalshi = kalshi_vectors.shape[0]
    for start in range(0, n_kalshi, chunk_size):
        end = min(start + chunk_size, n_kalshi)
        sims_chunk = cosine_similarity(kalshi_vectors[start:end], poly_vectors)
        local_i, local_j = np.where(sims_chunk > similarity_threshold)
        for li, j in zip(local_i, local_j):
            i = start + li
            kalshi_row = kalshi_df.iloc[i]
            poly_row = poly_df.iloc[j]
            kalshi_years = kalshi_row['years']
            poly_years = poly_row['years']
            if not (kalshi_years & poly_years or (not kalshi_years and not poly_years)):
                continue
            similar_events.append({
                'kalshi_ticker': kalshi_row.get('ticker', ''),
                'kalshi_event_ticker': kalshi_row.get('event_ticker', ''),
                'poly_condition_id': poly_row.get('condition_id', ''),
                'poly_token_yes_id': poly_row.get('token_0_id', ''),
                'poly_token_no_id': poly_row.get('token_1_id', ''),
                'kalshi_title_raw': kalshi_row.get('title', ''),
                'poly_title_raw': poly_row.get('question', ''),
                'kalshi_question': kalshi_row['combined_title'],
                'poly_title': poly_row['combined_title'],
                'similarity_score': sims_chunk[li, j],
                'kalshi_years': ', '.join(kalshi_years),
                'poly_years': ', '.join(poly_years),
                'kalshi_yes_bid': kalshi_row.get('yes_bid', ''),
                'kalshi_yes_ask': kalshi_row.get('yes_ask', ''),
                'kalshi_mid_price': kalshi_row.get('mid_price', ''),
                'kalshi_implied_prob': kalshi_row.get('market_implied_probability', ''),
                'kalshi_volume': kalshi_row.get('volume', ''),
                'kalshi_days_to_exp': kalshi_row.get('days_to_expiration', ''),
                'poly_yes_price': poly_row.get('token_0_price', ''),
                'poly_no_price': poly_row.get('token_1_price', ''),
                'poly_end_date': poly_row.get('end_date_iso', ''),
            })

    return pd.DataFrame(similar_events)


def main():
    today_date = datetime.now().strftime("%Y%m%d")

    logger.info("Loading data...")
    kalshi_df, poly_df = load_data(date_suffix=today_date)

    similar_events = find_similar_events(kalshi_df, poly_df, similarity_threshold=0.7)
    if not similar_events.empty:
        similar_events = similar_events.sort_values('similarity_score', ascending=False)

    output_filename = f'../historical_data/similar_events/similar_events_tfidf_{today_date}.csv'
    similar_events.to_csv(output_filename, index=False)
    logger.info(f"Similar events saved to {output_filename}")
    logger.info(f"Found {len(similar_events)} matched pairs.")


if __name__ == "__main__":
    main()
