"""Dataset loading and validation."""
from pathlib import Path

import pandas as pd


LABEL_ALIASES = {"label", "target", "category", "class", "v1"}
TEXT_ALIASES = {"text", "message", "sms", "body", "v2"}


def load_dataset(path: str | Path) -> tuple[pd.Series, pd.Series]:
    """Load a CSV containing one text column and one binary label column."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    normalized = {str(c).strip().lower(): c for c in df.columns}
    text_col = next((normalized[x] for x in TEXT_ALIASES if x in normalized), None)
    label_col = next((normalized[x] for x in LABEL_ALIASES if x in normalized), None)
    if text_col is None or label_col is None:
        if len(df.columns) >= 2:
            text_col, label_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("CSV must contain text and label columns")

    text = df[text_col].fillna("").astype(str)
    labels = df[label_col].astype(str).str.strip().str.lower()
    mapping = {"ham": 0, "not spam": 0, "0": 0, "spam": 1, "1": 1}
    labels = labels.map(mapping)
    if labels.isna().any():
        raise ValueError("Labels must be ham/spam or 0/1")
    return text, labels.astype(int)
