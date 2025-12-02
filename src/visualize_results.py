import pandas as pd
import numpy as np
import os
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
from tqdm import tqdm

PROCESSED_PATH = os.path.join("..", "data", "processed")
MODEL_PATH_TRANSFORMER = os.path.join("..", "models", "transformer")
MODEL_PATH_BASELINE = os.path.join("..", "models", "baseline", "baseline_model.pkl")
OUTPUT_DIR = os.path.join("..", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_data():
    print("Loading Test Data...")
    return pd.read_parquet(os.path.join(PROCESSED_PATH, "test.parquet"))

def get_baseline_preds(model, df):
    print("Generating Baseline Predictions...")
    return model.predict_proba(df['text_cleaned'])

def get_transformer_preds(model_path, df):
    print("Generating Transformer Predictions...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    predictions = []
    batch_size = 32
    texts = df['text_cleaned'].tolist()
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions.extend(probs.cpu().numpy())
            
    return np.array(predictions)

def plot_confusion_matrices(y_true, y_pred_base, y_pred_trans):
    print("Plotting Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_base = confusion_matrix(y_true, y_pred_base)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title("Baseline (Logistic Regression)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticklabels(['Negative', 'Positive'])
    axes[0].set_yticklabels(['Negative', 'Positive'])

    cm_trans = confusion_matrix(y_true, y_pred_trans)
    sns.heatmap(cm_trans, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
    axes[1].set_title(f"Transformer (DistilRoBERTa)\nAccuracy: {accuracy_score(y_true, y_pred_trans):.4f}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticklabels(['Negative', 'Positive'])
    axes[1].set_yticklabels(['Negative', 'Positive'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_confusion_matrices.png"), dpi=300)
    plt.close()

def plot_roc_curves(y_true, probs_base, probs_trans):
    print("Plotting ROC Curves...")
    fpr_base, tpr_base, _ = roc_curve(y_true, probs_base[:, 1])
    roc_auc_base = auc(fpr_base, tpr_base)
    
    fpr_trans, tpr_trans, _ = roc_curve(y_true, probs_trans[:, 1])
    roc_auc_trans = auc(fpr_trans, tpr_trans)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_trans, tpr_trans, color='darkorange', lw=2, label=f'Transformer (AUC = {roc_auc_trans:.3f})')
    plt.plot(fpr_base, tpr_base, color='navy', lw=2, linestyle='--', label=f'Baseline (AUC = {roc_auc_base:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "2_roc_curves.png"), dpi=300)
    plt.close()

def plot_domain_performance(df, y_pred_base, y_pred_trans):
    print("Plotting Domain Specific Performance...")
    domains = df['domain'].unique()
    results = []
    
    for domain in domains:
        idx = df['domain'] == domain
        y_subset = df[idx]['label']
        acc_base = accuracy_score(y_subset, y_pred_base[idx])
        acc_trans = accuracy_score(y_subset, y_pred_trans[idx])
        results.append({'Domain': domain.upper(), 'Model': 'Baseline', 'Accuracy': acc_base})
        results.append({'Domain': domain.upper(), 'Model': 'Transformer', 'Accuracy': acc_trans})
        
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x='Domain', y='Accuracy', hue='Model', palette=['#3498db', '#2ecc71'])
    plt.ylim(0.7, 1.0)
    plt.title("Model Robustness Across Domains")
    plt.ylabel("Accuracy Score")
    
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.3f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', 
                           xytext=(0, 9), textcoords='offset points')
                           
    plt.savefig(os.path.join(OUTPUT_DIR, "3_domain_performance.png"), dpi=300)
    plt.close()

def plot_wordclouds(df):
    print("Generating Word Clouds...")
    pos_text = " ".join(df[df['label'] == 1]['text_cleaned'])
    neg_text = " ".join(df[df['label'] == 0]['text_cleaned'])
    
    wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
    wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(neg_text)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(wc_neg, interpolation='bilinear')
    axes[0].set_title("Negative Words", fontsize=16, color='darkred')
    axes[0].axis('off')
    
    axes[1].imshow(wc_pos, interpolation='bilinear')
    axes[1].set_title("Positive Words", fontsize=16, color='darkgreen')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_wordclouds.png"), dpi=300)
    plt.close()

def main():
    test_df = load_data()
    y_true = test_df['label'].values
    
    baseline = joblib.load(MODEL_PATH_BASELINE)
    base_probs = get_baseline_preds(baseline, test_df)
    base_preds = base_probs.argmax(axis=1)
    
    trans_probs = get_transformer_preds(MODEL_PATH_TRANSFORMER, test_df)
    trans_preds = trans_probs.argmax(axis=1)
    
    plot_confusion_matrices(y_true, base_preds, trans_preds)
    plot_roc_curves(y_true, base_probs, trans_probs)
    plot_domain_performance(test_df, base_preds, trans_preds)
    plot_wordclouds(test_df)
    
    print(f"All plots saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
