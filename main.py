import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load and prepare dataset
df = pd.read_csv("Balanced_Bengali_hate_Speech(10000).csv")
df = df.rename(columns={"sentence": "text", "hate": "label"})
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Use a small transformer model (tiny but functional)
model_name = "prajjwal1/bert-tiny"  # Lightweight alternative to DistilBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=32)

encoded_dataset = dataset.map(preprocess, batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Training configuration (no logging, no tracking, safe for CPU)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to="none",
    logging_strategy="no",
    push_to_hub=False,
    disable_tqdm=True,
    seed=42
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print("✅ Evaluation:", metrics)

# Predict function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Hate" if prediction == 1 else "Not Hate"

# Try prediction
sample_text = "তুমি একটা বাজে লোক"
print("Sample Prediction:", predict(sample_text))

# Save model and tokenizer
model.save_pretrained("bert_tiny_bangla_hate_model")
tokenizer.save_pretrained("bert_tiny_bangla_hate_model")
