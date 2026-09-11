"""Train and persist a strong classical spam classifier."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.text import normalize_text


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_text,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=100_000,
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_model(texts, labels, output_path: str | Path = "artifacts/spam_classifier.joblib") -> Pipeline:
    model = build_pipeline()
    model.fit(texts, labels)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model
