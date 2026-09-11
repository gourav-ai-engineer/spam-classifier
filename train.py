"""Train the spam classifier from a CSV dataset."""
import argparse

from src.data.loader import load_dataset
from src.models.train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spam.csv")
    parser.add_argument("--output", default="artifacts/spam_classifier.joblib")
    args = parser.parse_args()
    texts, labels = load_dataset(args.data)
    model = train_model(texts, labels, args.output)
    print(f"Trained on {len(texts)} samples; saved to {args.output}")
