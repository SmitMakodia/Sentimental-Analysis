import pandas as pd
import numpy as np
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_PATH = os.path.join("..", "data", "processed")
MODEL_OUTPUT_PATH = os.path.join("..", "models", "transformer")
MODEL_CHECKPOINT = "distilroberta-base"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train_transformer():
    print(f"Starting Transformer Training (GPU: {torch.cuda.get_device_name(0)})...")
    
    print("   Loading Parquet data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_PATH, "test.parquet"))
    
    train_dataset = Dataset.from_pandas(train_df[['text_cleaned', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text_cleaned', 'label']])
    
    train_dataset = train_dataset.rename_column("text_cleaned", "text")
    test_dataset = test_dataset.rename_column("text_cleaned", "text")

    print(f"   Tokenizing data using {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    print("   Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )

    print("   Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_OUTPUT_PATH, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("\nTraining Started... (This will take 15-25 mins)")
    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
    
    print(f"Training Complete! Model saved to {os.path.abspath(MODEL_OUTPUT_PATH)}")

    print("\nFinal Evaluation on Test Set:")
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. Training will be extremely slow on CPU.")
    else:
        train_transformer()
