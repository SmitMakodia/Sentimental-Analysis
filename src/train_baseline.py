import pandas as pd 
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

PROCESSED_PATH = os.path.join("..", "data", "processed")
MODEL_PATH = os.path.join("..", "models", "baseline")
os.makedirs(MODEL_PATH, exist_ok=True)

def train_baseline():
    print("Starting Baseline Training (TF-IDF + LogReg)...")

    print("   Loading data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "test.parquet"))

    X_train = train_df['text_cleaned']
    y_train = train_df['label']
    X_test = test_df['text_cleaned']
    y_test = test_df['label']

    print("   Building Pipeline (Vectorization -> Model)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    print("   Training model (this might take 30-60 seconds)...")
    pipeline.fit(X_train, y_train)

    print("   Evaluating...")
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nBaseline Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nDomain-Specific Performance:")
    test_df['prediction'] = y_pred
    for domain in test_df['domain'].unique():
        domain_subset = test_df[test_df['domain'] == domain]
        domain_acc = accuracy_score(domain_subset['label'], domain_subset['prediction'])
        print(f"   - {domain.upper()}: {domain_acc * 100:.2f}%")

    model_file = os.path.join(MODEL_PATH, "baseline_model.pkl")
    joblib.dump(pipeline, model_file)
    print(f"\nModel saved to: {model_file}")

if __name__ == "__main__":
    train_baseline()
