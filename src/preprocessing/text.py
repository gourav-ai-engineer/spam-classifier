"""Lightweight deterministic text normalization."""
import re


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    return re.sub(r"\s+", " ", text).strip()
