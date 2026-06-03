import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from datetime import datetime
import re  # For cleaning
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data (if not already downloaded)
nltk.download('wordnet')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    """Cleans the input text by removing special characters, HTML tags, and extra whitespace."""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove special characters
    text = text.lower()
    text = text.strip()
    return text

def lemmatize_text(text: str) -> str:
    """Lemmatizes the input text."""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def load_data(date_suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Read the CSV files
    kalshi_df = pd.read_csv(f'processed_markets_{date_suffix}.csv')
    kalshi_df = kalshi_df.dropna(subset=['title'])  # Require title

    poly_df = pd.read_csv(f'open_markets_{date_suffix}.csv')
    poly_df = poly_df.dropna(subset=['question', 'description'])  # Require both question and description

    # Combine title and description, cleaning the text
    kalshi_df['combined_title'] = kalshi_df['title'].str.strip() + ' ' + kalshi_df['subtitle'].str.strip() + ' ' + kalshi_df['yes_sub_title'].str.strip()
    kalshi_df['combined_title'] = kalshi_df['combined_title'].fillna('').astype(str).apply(clean_text).apply(lemmatize_text)

    poly_df['combined_title'] = poly_df['question'].str.strip() + ' ' + poly_df['description'].str.strip()
    poly_df['combined_title'] = poly_df['combined_title'].fillna('').astype(str).apply(clean_text).apply(lemmatize_text)
    
    return kalshi_df, poly_df

def find_similar_events(kalshi_df: pd.DataFrame, poly_df: pd.DataFrame, similarity_threshold: float = 0.3) -> pd.DataFrame:
    similar_events = []
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english')

    kalshi_texts = kalshi_df['combined_title'].fillna('').astype(str).tolist()
    poly_texts = poly_df['combined_title'].fillna('').astype(str).tolist()
    all_texts = kalshi_texts + poly_texts

    vectorizer.fit(all_texts)
    kalshi_vectors = vectorizer.transform(kalshi_texts)
    poly_vectors = vectorizer.transform(poly_texts)
    
    print("Calculating similarities...")
    similarities = cosine_similarity(kalshi_vectors, poly_vectors)
    
    print("Finding similar pairs...")
    for i, j in zip(*np.where(similarities > similarity_threshold)):
        similar_events.append({
            'kalshi_question': kalshi_df.iloc[i]['combined_title'],
            'poly_title': poly_df.iloc[j]['combined_title'],
            'similarity_score': similarities[i, j]
        })
    
    return pd.DataFrame(similar_events)

def main():
    # Load environment variables (not needed anymore)
    # load_dotenv()
    
    # Generate today's date string
    today_date = datetime.now().strftime("%Y%m%d")
    
    # Load data
    print("Loading data...")
    kalshi_df, poly_df = load_data(date_suffix=today_date)
    
    # Find similar events
    similar_events = find_similar_events(kalshi_df, poly_df)
    
    # Sort by similarity score in descending order
    similar_events = similar_events.sort_values('similarity_score', ascending=False)
    
    # Save results
    output_filename = f'similar_events_tfidf_{today_date}.csv'
    similar_events.to_csv(output_filename, index=False)
    print(f"Similar events have been saved to {output_filename}")
    
    # Print top 10 most similar events
    print("\nTop 10 most similar events:")
    print(similar_events.head(10))

if __name__ == "__main__":
    main()
