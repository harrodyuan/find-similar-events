import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from datetime import datetime
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import logging

# --- Logging Configuration ---
logging.basicConfig(
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Download required NLTK data (if not already downloaded)
nltk.download('wordnet')
nltk.download('stopwords')

# --- Text Preprocessing Functions ---

def clean_text(text: str) -> str:
    """Cleans the input text by removing special characters, HTML tags, and extra whitespace."""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)  # Remove special characters except numbers
    text = text.lower()
    text = text.strip()
    return text

def lemmatize_text(text: str) -> str:
    """Lemmatizes the input text to standardize word forms."""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def standardize_time(text: str, current_year: int) -> str:
    """Standardizes relative time expressions like 'this year' to 'in {current_year}'."""
    text = re.sub(r'\bthis year\b', f'in {current_year}', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnext year\b', f'in {current_year + 1}', text, flags=re.IGNORECASE)
    return text

def extract_years(text: str) -> set:
    """Extracts four-digit years (e.g., 2025) from the text."""
    years = re.findall(r'\b20\d{2}\b', text)
    return set(years)

# --- Data Loading Function ---

def load_data(date_suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and preprocesses Kalshi and Polymarket data from CSV files."""
    current_year = datetime.now().year

    # Load Kalshi data
    kalshi_df = pd.read_csv(f'../historical_data/processed_markets/kalshi_markets_{date_suffix}.csv')
    kalshi_df = kalshi_df.dropna(subset=['title'])

    # Load Polymarket data
    poly_df = pd.read_csv(f'../historical_data/processed_markets/polymarket_markets_{date_suffix}.csv')
    poly_df = poly_df.dropna(subset=['question'])

    # Process Kalshi titles - just use title field
    kalshi_df['combined_title'] = kalshi_df['title'].fillna('')
    kalshi_df['combined_title'] = kalshi_df['combined_title'].apply(
        lambda x: standardize_time(x, current_year)).apply(clean_text).apply(lemmatize_text)

    # Process Polymarket questions
    poly_df['combined_title'] = poly_df['question'].fillna('')
    poly_df['combined_title'] = poly_df['combined_title'].apply(
        lambda x: standardize_time(x, current_year)).apply(clean_text).apply(lemmatize_text)

    # Extract years from preprocessed text
    kalshi_df['years'] = kalshi_df['combined_title'].apply(extract_years)
    poly_df['years'] = poly_df['combined_title'].apply(extract_years)

    return kalshi_df, poly_df

# --- Similarity Detection Function ---

def find_similar_events(kalshi_df: pd.DataFrame, poly_df: pd.DataFrame, similarity_threshold: float = 0.7) -> pd.DataFrame:
    """Finds similar events between Kalshi and Polymarket using Sentence-BERT embeddings and year matching."""
    similar_events = []

    # Load Sentence-BERT model
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode texts into embeddings
    print("Encoding texts...")
    kalshi_embeddings = model.encode(kalshi_df['combined_title'].values)
    poly_embeddings = model.encode(poly_df['combined_title'].values)

    # Calculate cosine similarities in chunks to avoid building a giant
    # (n_kalshi x n_poly) matrix in memory (which can be many GB and OOM).
    print("Calculating similarities (chunked)...")
    chunk_size = 1000
    n_kalshi = kalshi_embeddings.shape[0]
    for start in range(0, n_kalshi, chunk_size):
        end = min(start + chunk_size, n_kalshi)
        sims_chunk = cosine_similarity(kalshi_embeddings[start:end], poly_embeddings)
        local_i, local_j = np.where(sims_chunk > similarity_threshold)
        for li, j in zip(local_i, local_j):
            i = start + li
            kalshi_row = kalshi_df.iloc[i]
            poly_row = poly_df.iloc[j]
            kalshi_years = kalshi_row['years']
            poly_years = poly_row['years']
            # Match if years overlap or both have no years specified
            if not (kalshi_years & poly_years or (not kalshi_years and not poly_years)):
                continue
            similarities_value = sims_chunk[li, j]
            similar_events.append({
                # Identifiers
                'kalshi_ticker': kalshi_row.get('ticker', ''),
                'kalshi_event_ticker': kalshi_row.get('event_ticker', ''),
                'poly_condition_id': poly_row.get('condition_id', ''),
                'poly_token_yes_id': poly_row.get('token_0_id', ''),
                'poly_token_no_id': poly_row.get('token_1_id', ''),
                # Original titles (uncleaned)
                'kalshi_title_raw': kalshi_row.get('title', ''),
                'poly_title_raw': poly_row.get('question', ''),
                # Cleaned titles (for debugging)
                'kalshi_question': kalshi_row['combined_title'],
                'poly_title': poly_row['combined_title'],
                # Similarity
                'similarity_score': similarities_value,
                'kalshi_years': ', '.join(kalshi_years),
                'poly_years': ', '.join(poly_years),
                # Kalshi prices
                'kalshi_yes_bid': kalshi_row.get('yes_bid', ''),
                'kalshi_yes_ask': kalshi_row.get('yes_ask', ''),
                'kalshi_mid_price': kalshi_row.get('mid_price', ''),
                'kalshi_implied_prob': kalshi_row.get('market_implied_probability', ''),
                'kalshi_volume': kalshi_row.get('volume', ''),
                'kalshi_days_to_exp': kalshi_row.get('days_to_expiration', ''),
                # Polymarket prices
                'poly_yes_price': poly_row.get('token_0_price', ''),
                'poly_no_price': poly_row.get('token_1_price', ''),
                'poly_end_date': poly_row.get('end_date_iso', ''),
            })

    return pd.DataFrame(similar_events)

# --- Main Execution Function ---

def main():
    """Main function to execute the similarity analysis."""
    today_date = datetime.now().strftime("%Y%m%d")
    
    print("Loading data...")
    kalshi_df, poly_df = load_data(date_suffix=today_date)
    
    similar_events = find_similar_events(kalshi_df, poly_df, similarity_threshold=0.7)
    similar_events = similar_events.sort_values('similarity_score', ascending=False)
    
    # Updated output path
    output_filename = f'../historical_data/similar_events/similar_events_sbert_{today_date}.csv'
    similar_events.to_csv(output_filename, index=False)
    logger.info(f"Similar events saved to {output_filename}")
    
    logger.info("\nTop 10 most similar events:")
    logger.info(similar_events.head(10))


if __name__ == "__main__":
    main()