"""Predict whether a message is spam."""
import argparse

from src.inference import SpamClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", default="artifacts/spam_classifier.joblib")
    args = parser.parse_args()
    print(SpamClassifier(args.model).predict(args.text))
