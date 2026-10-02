"""
Email Spam Classifier - Model Training Script
=============================================
This script loads the labeled email/SMS spam dataset, performs text
preprocessing and TF-IDF feature extraction, trains a Multinomial Naive
Bayes classifier, evaluates performance (Accuracy and Precision), and
saves the trained model artifacts for real-time inference.
"""

import os
import re
import json
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import joblib

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "spam.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Fallback dataset URL if local dataset is missing
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"


def clean_text(text: str) -> str:
    """
    Cleans raw email/message text for NLP feature extraction.
    1. Converts text to lowercase.
    2. Removes URLs and web links.
    3. Removes HTML tags if present.
    4. Removes special punctuation/symbols while keeping alphanumeric words.
    5. Normalizes whitespace.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_dataset_exists() -> str:
    """
    Ensures that dataset/spam.csv exists. If not found, downloads the
    authentic UCI SMS/Email Spam Collection dataset automatically.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at: {DATASET_PATH}")
        print("Downloading UCI SMS/Email Spam Collection dataset...")
        try:
            req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as response:
                df = pd.read_csv(response, sep="\t", header=None, names=["label", "message"])
            df.to_csv(DATASET_PATH, index=False)
            print(f"Successfully downloaded and saved dataset ({len(df)} records) to {DATASET_PATH}\n")
        except Exception as e:
            raise RuntimeError(
                f"Failed to automatically download dataset: {e}\n"
                f"Please manually place 'spam.csv' inside the 'dataset/' folder with 'label' and 'message' columns."
            )
    return DATASET_PATH


def train_and_evaluate(quiet: bool = False):
    """
    Trains the Multinomial Naive Bayes model on the spam dataset,
    evaluates Accuracy and Precision dynamically on the test set,
    and serializes the model, vectorizer, and metrics to disk.
    """
    csv_file = ensure_dataset_exists()

    if not quiet:
        print("[1/5] Loading dataset...")
    df = pd.read_csv(csv_file)

    # Validate dataset structure
    if "label" not in df.columns or "message" not in df.columns:
        # Check if tab or semicolon separated fallback
        if len(df.columns) == 1:
            df = pd.read_csv(csv_file, sep="\t")
        if "label" not in df.columns or "message" not in df.columns:
            raise ValueError("Dataset must contain 'label' and 'message' columns.")

    # Handle missing values
    df.dropna(subset=["label", "message"], inplace=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]

    if not quiet:
        print(f"      Total samples: {len(df)} (Ham: {(df['label'] == 'ham').sum()}, Spam: {(df['label'] == 'spam').sum()})")
        print("[2/5] Cleaning text and preprocessing...")

    # Text cleaning
    df["cleaned_message"] = df["message"].apply(clean_text)
    # Remove any messages that became empty after cleaning
    df = df[df["cleaned_message"].str.len() > 0]

    # Map labels: ham -> 0, spam -> 1
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # Train / test split (80% train, 20% test with stratified sampling)
    if not quiet:
        print("[3/5] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_message"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    # Feature extraction using TF-IDF
    if not quiet:
        print("[4/5] Extracting TF-IDF features and training Naive Bayes model...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=5000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Naive Bayes Classifier
    model = MultinomialNB(alpha=0.2)
    model.fit(X_train_vec, y_train)

    # Model Evaluation on the held-out test dataset
    if not quiet:
        print("[5/5] Evaluating model performance on test set...")
    y_pred = model.predict(X_test_vec)

    # Dynamic metrics calculation
    accuracy = float(accuracy_score(y_test, y_pred) * 100)
    precision = float(precision_score(y_test, y_pred, pos_label=1) * 100)

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    if not quiet:
        print("\n" + "=" * 40)
        print("         TRAINING COMPLETED")
        print("=" * 40)
        print(f"Model Accuracy : {accuracy:.2f}%")
        print(f"Precision      : {precision:.2f}%")
        print("=" * 40)
        print(f"Model saved to      : {MODEL_PATH}")
        print(f"Vectorizer saved to : {VECTORIZER_PATH}")
        print(f"Metrics saved to    : {METRICS_PATH}")
        print("=" * 40 + "\n")

    return model, vectorizer, accuracy, precision


if __name__ == "__main__":
    train_and_evaluate()
