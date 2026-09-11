import pytest

from src.inference import SpamClassifier


def test_missing_model_is_clear():
    with pytest.raises(FileNotFoundError):
        SpamClassifier("artifacts/does-not-exist.joblib")
