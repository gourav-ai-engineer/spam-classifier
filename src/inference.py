"""Inference API for the trained classifier."""
from pathlib import Path

import joblib


class SpamClassifier:
    def __init__(self, model_path: str | Path = "artifacts/spam_classifier.joblib"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
        self.model = joblib.load(path)

    def predict(self, text: str) -> dict:
        label = int(self.model.predict([text])[0])
        probability = float(self.model.predict_proba([text])[0][label])
        return {"label": "spam" if label else "ham", "spam_probability": probability}
