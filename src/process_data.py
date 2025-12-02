import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join("..", "data", "raw")
PROCESSED_PATH = os.path.join("..", "data", "processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

def clean_text(text, domain):
    text = str(text).lower()
    
    if domain == 'imdb':
        text = text.replace('<br />', ' ')
    
    if domain == 'twitter':
        text = re.sub(r'@\w+', '@user', text)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_data():
    print("Starting Data Preprocessing & Balancing...")

    print("   [1/4] Loading raw CSVs...")
    df_imdb = pd.read_csv(os.path.join(RAW_PATH, "imdb_raw.csv"))
    df_yelp = pd.read_csv(os.path.join(RAW_PATH, "yelp_raw.csv"))
    df_twitter = pd.read_csv(os.path.join(RAW_PATH, "twitter_raw.csv"))

    print("   [2/4] Balancing Data...")
    print(f"       - Original Yelp size: {len(df_yelp)}")
    
    df_yelp_balanced = df_yelp.sample(n=50000, random_state=42)
    
    print(f"       - Balanced Yelp size: {len(df_yelp_balanced)}")
    print(f"       - IMDb size: {len(df_imdb)}")
    print(f"       - Twitter size: {len(df_twitter)}")

    df_final = pd.concat([df_imdb, df_yelp_balanced, df_twitter], axis=0)
    
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("   [3/4] Cleaning text (removing HTML, standardizing)...")
    df_final['text_cleaned'] = df_final.apply(lambda x: clean_text(x['text'], x['domain']), axis=1)

    print("   [4/4] Splitting into Train/Test (80/20)...")
    X = df_final[['text_cleaned', 'domain']]
    y = df_final['label']

    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['label'])

    train_df.to_parquet(os.path.join(PROCESSED_PATH, "train.parquet"))
    test_df.to_parquet(os.path.join(PROCESSED_PATH, "test.parquet"))

    print("Success! Data Processed.")
    print(f"   - Training Set: {len(train_df)} rows")
    print(f"   - Test Set: {len(test_df)} rows")
    print(f"   - Saved to: {os.path.abspath(PROCESSED_PATH)}")

if __name__ == "__main__":
    process_data()
