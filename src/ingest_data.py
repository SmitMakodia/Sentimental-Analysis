import pandas as pd
from datasets import load_dataset
import os

RAW_DATA_PATH = os.path.join("..", "data", "raw")
os.makedirs(RAW_DATA_PATH, exist_ok=True)

def ingest_data():
    print("Starting Data Ingestion Pipeline (Fixed Version)...")
    
    print("\n[1/3] Downloading IMDb Dataset...")
    try:
        imdb = load_dataset("imdb")
        df_imdb = pd.DataFrame(imdb['train'])
        df_imdb = df_imdb[['text', 'label']]
        df_imdb['domain'] = 'imdb'
        print(f"   - IMDb loaded: {len(df_imdb)} rows")
    except Exception as e:
        print(f"   ! Error loading IMDb: {e}")

    print("\n[2/3] Downloading Yelp Polarity Dataset...")
    try:
        yelp = load_dataset("yelp_polarity")
        df_yelp = pd.DataFrame(yelp['train'])
        df_yelp = df_yelp[['text', 'label']]
        df_yelp['domain'] = 'yelp'
        print(f"   - Yelp loaded: {len(df_yelp)} rows")
    except Exception as e:
        print(f"   ! Error loading Yelp: {e}")

    print("\n[3/3] Downloading TweetEval (Sentiment) Dataset...")
    try:
        twitter = load_dataset("tweet_eval", "sentiment")
        df_twitter = pd.DataFrame(twitter['train'])
        df_twitter = df_twitter[df_twitter['label'] != 1].copy()
        df_twitter['label'] = df_twitter['label'].replace(2, 1)
        df_twitter['domain'] = 'twitter'
        print(f"   - Twitter loaded: {len(df_twitter)} rows (Neutrals removed)")
    except Exception as e:
        print(f"   ! Error loading Twitter: {e}")

    print("\nSaving raw CSV files to local disk...")
    df_imdb.to_csv(os.path.join(RAW_DATA_PATH, "imdb_raw.csv"), index=False)
    df_yelp.to_csv(os.path.join(RAW_DATA_PATH, "yelp_raw.csv"), index=False)
    df_twitter.to_csv(os.path.join(RAW_DATA_PATH, "twitter_raw.csv"), index=False)
    
    print(f"Success! Data saved to {os.path.abspath(RAW_DATA_PATH)}")

if __name__ == "__main__":
    ingest_data()
