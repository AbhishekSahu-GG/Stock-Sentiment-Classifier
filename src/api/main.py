from fastapi import FastAPI
from src.api.schemas import NewsInput, BatchInput, SentimentOutput, BatchOutput, HealthResponse
from src.api.model_loader import load_model, predict

app = FastAPI(
    title="Stock Sentiment Classifier",
    description="Financial news sentiment analysis using fine-tuned DistilBERT",
    version="1.0.0"
)

# Load model once at startup
tokenizer, model = load_model()

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model": "DistilBERT",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=SentimentOutput)
def predict_sentiment(input: NewsInput):
    result = predict(input.headline, tokenizer, model)
    return {
        "headline": input.headline,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"]
    }

@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(input: BatchInput):
    results = []
    for headline in input.headlines:
        result = predict(headline, tokenizer, model)
        results.append({
            "headline": headline,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        })
    return {"results": results}