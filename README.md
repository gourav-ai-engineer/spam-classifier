# Spam Classifier

A production-oriented SMS spam classification project built with Python, scikit-learn, TF-IDF and Logistic Regression, with a FastAPI inference service, automated dataset setup, tests and Docker support.

## Architecture

```text
UCI SMS Spam Collection
        ↓
scripts/download_dataset.py
        ↓
CSV validation
        ↓
Text normalization
        ↓
TF-IDF (1–2 grams)
        ↓
Logistic Regression
        ↓
Saved model + evaluation metrics
        ↓
CLI / FastAPI
```

## Project structure

```text
spam-classifier/
├── app/main.py                 # FastAPI service
├── data/                       # Local dataset (ignored by git)
├── notebooks/                  # Experiments / EDA
├── artifacts/                  # Model + metrics (model files ignored)
├── scripts/download_dataset.py # Downloads official UCI dataset
├── src/
│   ├── data/loader.py          # Dataset loading and validation
│   ├── preprocessing/text.py   # Text normalization
│   ├── models/train.py         # TF-IDF + Logistic Regression
│   ├── evaluation.py           # Accuracy / precision / recall / F1
│   └── inference.py            # Prediction service
├── tests/
├── train.py                    # Train + evaluate CLI
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

### 2. Get the dataset automatically

```bash
python scripts/download_dataset.py
```

This downloads the **UCI SMS Spam Collection** (5,574 labeled messages) and creates `data/spam.csv`. The UCI dataset is published under CC BY 4.0. [UCI dataset](https://archive.ics.uci.edu/dataset/228/sms)

### 3. Train and evaluate

```bash
python train.py
```

The command creates:

```text
artifacts/spam_classifier.joblib
artifacts/metrics.json
```

It prints accuracy, precision, recall, F1 and the classification report for a stratified 80/20 test split.

### 4. Predict

```bash
python predict.py "Congratulations! You won a free prize!"
python predict.py "Hey, are you coming to class today?"
```

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

Health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"You won a free prize\"}"
```

### 6. Run tests

```bash
pytest -q
```

### 7. Docker

After training a model:

```bash
docker build -t spam-classifier .
docker run -p 8000:8000 spam-classifier
```

## Data and reproducibility

The dataset is not committed to Git because it is downloaded reproducibly by `scripts/download_dataset.py`. The source is the official UCI Machine Learning Repository. Cite Tiago Almeida and Jos Hidalgo when using the dataset.

## Next upgrades

Planned improvements can include Naive Bayes and SVM baselines, cross-validation, hyperparameter search, experiment tracking, model versioning, threshold tuning, monitoring, authentication, rate limiting and CI/CD.
