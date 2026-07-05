from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "version" in data

def test_predict_single():
    response = client.post("/predict", json={
        "headline": "Company reports record profits"
    })
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_batch():
    response = client.post("/predict/batch", json={
        "headlines": [
            "Company reports record profits",
            "Firm files for bankruptcy",
            "Annual meeting scheduled"
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3

def test_predict_invalid_input():
    response = client.post("/predict", json={})
    assert response.status_code == 422