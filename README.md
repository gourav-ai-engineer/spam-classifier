# Spam Classifier

A production-oriented SMS/email spam classification project built with Python, scikit-learn, TF-IDF and Logistic Regression, with a FastAPI inference service and Docker support.

## Architecture

```text
CSV dataset → validation → text normalization → TF-IDF → Logistic Regression → saved model
                                                                    ↓
                                                            FastAPI /predict
```

## Project structure

```text
spam-classifier/
├── app/main.py                 # FastAPI service
├── data/                       # Local datasets (not committed)
├── notebooks/                  # Experiments / EDA
├── artifacts/                  # Trained model (ignored by git)
├── src/
│   ├── data/loader.py          # Dataset loading and validation
│   ├── preprocessing/text.py   # Text normalization
│   ├── models/train.py         # TF-IDF + Logistic Regression
│   └── inference.py            # Prediction service
├── tests/
├── train.py                    # Training CLI
├── predict.py                  # Prediction CLI
├── Dockerfile
└── requirements.txt
```

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add data

Place your CSV at `data/spam.csv`. It should contain a text column and a label column. Common formats include `v1,v2` from the SMS Spam Collection or `label,text`. Labels can be `ham`/`spam` or `0`/`1`.

### 3. Train

```bash
python train.py --data data/spam.csv
```

The trained pipeline is saved to `artifacts/spam_classifier.joblib`.

### 4. Predict

```bash
python predict.py "Congratulations! You won a free prize."
```

### 5. Run API

```bash
uvicorn app.main:app --reload
```

Then open `/docs` for Swagger UI.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"You won a free prize\"}"
```

### 6. Test

```bash
pytest -q
```

### 7. Docker

Build and run after training a model:

```bash
docker build -t spam-classifier .
docker run -p 8000:8000 spam-classifier
```

## Current scope

This repository provides a clean baseline that can be extended with cross-validation, precision/recall/F1 reporting, confusion matrices, experiment tracking, model versioning, authentication, rate limiting and CI/CD.
