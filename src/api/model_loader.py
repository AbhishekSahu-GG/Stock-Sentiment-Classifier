import torch
import os
# os.add_dll_directory("D:\\Projects\\Stock Sentiment Analysis (FastAPI + Docker)\\stock-sentiment-classifier\\venv311-stock\\Lib\\site-packages\\torch\\lib")

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_PATH = "models/distilbert"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def predict(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return {
        "sentiment": LABEL_MAP[pred.item()],
        "confidence": round(confidence.item(), 4)
    }