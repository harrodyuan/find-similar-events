import pandas as pd
from datetime import datetime
import re

def clean_text_exact(text: str) -> str:
    """Cleans the input text for exact matching."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters, keep numbers
    text = text.lower()
    text = text.strip()
    return text

def load_data(date_suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Read the CSV files
    kalshi_df = pd.read_csv(f'processed_markets_{date_suffix}.csv')
    kalshi_df = kalshi_df.dropna(subset=['title'])  # Require title

    poly_df = pd.read_csv(f'open_markets_{date_suffix}.csv')
    poly_df = poly_df.dropna(subset=['question', 'description'])  # Require both question and description

    # Combine title and description, cleaning the text
    kalshi_df['combined_title'] = kalshi_df['title'].str.strip() + ' ' + kalshi_df['subtitle'].str.strip() + ' ' + kalshi_df['yes_sub_title'].str.strip()
    kalshi_df['combined_title'] = kalshi_df['combined_title'].fillna('').astype(str).apply(clean_text_exact)

    poly_df['combined_title'] = poly_df['question'].str.strip() + ' ' + poly_df['description'].str.strip()
    poly_df['combined_title'] = poly_df['combined_title'].fillna('').astype(str).apply(clean_text_exact)
    
    return kalshi_df, poly_df

def find_identical_events(kalshi_df: pd.DataFrame, poly_df: pd.DataFrame) -> pd.DataFrame:
    identical_events = []

    for i in range(len(kalshi_df)):
        for j in range(len(poly_df)):
            if kalshi_df['combined_title'].iloc[i] == poly_df['combined_title'].iloc[j]:
                identical_events.append({
                    'kalshi_question': kalshi_df['combined_title'].iloc[i],
                    'poly_title': poly_df['combined_title'].iloc[j]
                })

    return pd.DataFrame(identical_events)

def main():
    # Generate today's date string
    today_date = datetime.now().strftime("%Y%m%d")
    
    # Load data
    print("Loading data...")
    kalshi_df, poly_df = load_data(date_suffix=today_date)
    
    # Find identical events
    identical_events = find_identical_events(kalshi_df, poly_df)
    
    # Save results
    output_filename = f'identical_events_{today_date}.csv'
    identical_events.to_csv(output_filename, index=False)
    print(f"Identical events have been saved to {output_filename}")
    
    # Print top 10 most similar events
    print("\nIdentical events:")
    print(identical_events)

if __name__ == "__main__":
    main()
