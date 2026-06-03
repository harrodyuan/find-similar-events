import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from datetime import datetime
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

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
    kalshi_df = pd.read_csv(f'./kalshi_data_collector/historical_data/processed_events/processed_events_{date_suffix}.csv')
    kalshi_df = kalshi_df.dropna(subset=['title'])  # Require title

    # Load Polymarket data
    poly_df = pd.read_csv(f'./polymarket_data_collector/open_markets_{date_suffix}.csv')
    poly_df = poly_df.dropna(subset=['question'])  # Require question

    # Combine titles and preprocess for Kalshi
    kalshi_df['combined_title'] = (kalshi_df['sub_title'].fillna('') + ' ' + kalshi_df['title'].fillna('')).str.strip()
    kalshi_df['combined_title'] = kalshi_df['combined_title'].apply(
        lambda x: standardize_time(x, current_year)).apply(clean_text).apply(lemmatize_text)

    # Combine question and description and preprocess for Polymarket
    poly_df['combined_title'] = (poly_df['question'].fillna('') + ' ' + poly_df['description'].fillna('')).str.strip()
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

    # Calculate cosine similarities
    print("Calculating similarities...")
    similarities = cosine_similarity(kalshi_embeddings, poly_embeddings)

    # Find similar pairs with matching years
    print("Finding similar pairs...")
    indices = np.where(similarities > similarity_threshold)
    for i, j in zip(indices[0], indices[1]):
        kalshi_years = kalshi_df.iloc[i]['years']
        poly_years = poly_df.iloc[j]['years']
        # Match if years overlap or both have no years specified
        if kalshi_years & poly_years or (not kalshi_years and not poly_years):
            similar_events.append({
                'kalshi_question': kalshi_df.iloc[i]['combined_title'],
                'poly_title': poly_df.iloc[j]['combined_title'],
                'similarity_score': similarities[i, j],
                'kalshi_years': ', '.join(kalshi_years),
                'poly_years': ', '.join(poly_years)
            })

    return pd.DataFrame(similar_events)

# --- Main Execution Function ---

def main():
    """Main function to execute the similarity analysis."""
    # Generate today's date string
    today_date = datetime.now().strftime("%Y%m%d")

    # Load data
    print("Loading data...")
    kalshi_df, poly_df = load_data(date_suffix=today_date)

    # Find similar events with a threshold of 0.7
    similar_events = find_similar_events(kalshi_df, poly_df, similarity_threshold=0.7)

    # Sort by similarity score in descending order
    similar_events = similar_events.sort_values('similarity_score', ascending=False)

    # Save results to CSV
    output_filename = f'similar_events_sbert_{today_date}.csv'
    similar_events.to_csv(output_filename, index=False)
    print(f"Similar events have been saved to {output_filename}")

    # Print top 10 most similar events
    print("\nTop 10 most similar events:")
    print(similar_events.head(10))

if __name__ == "__main__":
    main()