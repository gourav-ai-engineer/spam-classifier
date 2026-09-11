"""FastAPI service for spam classification."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import SpamClassifier

app = FastAPI(title="Spam Classifier API", version="1.0.0")
_model = None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


def get_model() -> SpamClassifier:
    global _model
    if _model is None:
        _model = SpamClassifier(Path("artifacts/spam_classifier.joblib"))
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_available": Path("artifacts/spam_classifier.joblib").exists()}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return get_model().predict(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
