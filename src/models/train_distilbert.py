# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
import mlflow
import mlflow.pytorch
import numpy as np
import os

# Label encoding
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['sentence'].tolist()
        self.labels = df['sentiment_label'].map(LABEL2ID).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_distilbert():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("stock-sentiment-classifier")

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    EPOCHS = 3

    with mlflow.start_run(run_name="distilbert_finetune"):
        mlflow.log_param("model_type", "DistilBERT")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("max_length", 128)

        for epoch in range(EPOCHS):
            # --- Training ---
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # --- Validation ---
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            report = classification_report(
                all_labels, all_preds,
                target_names=["negative", "neutral", "positive"],
                output_dict=True
            )

            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f}")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("macro_f1", macro_f1, step=epoch)
            mlflow.log_metric("negative_f1", report['negative']['f1-score'], step=epoch)
            mlflow.log_metric("neutral_f1", report['neutral']['f1-score'], step=epoch)
            mlflow.log_metric("positive_f1", report['positive']['f1-score'], step=epoch)

        # Final report
        print(classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"]))

        # Save model
        os.makedirs("models/distilbert", exist_ok=True)
        model.save_pretrained("models/distilbert")
        tokenizer.save_pretrained("models/distilbert")
        print("Model saved to models/distilbert")

if __name__ == "__main__":
    train_distilbert()