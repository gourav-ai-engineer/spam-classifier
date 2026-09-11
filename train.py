"""Train, evaluate and persist the spam classifier."""
import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset
from src.evaluation import evaluate_model
from src.models.train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spam.csv")
    parser.add_argument("--output", default="artifacts/spam_classifier.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    texts, labels = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    model = train_model(X_train, y_train, args.output)
    metrics = evaluate_model(model, X_test, y_test)

    metrics_path = Path(args.output).with_name("metrics.json")
    metrics_path.write_text(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2), encoding="utf-8")

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(metrics["classification_report"])
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {metrics_path}")
