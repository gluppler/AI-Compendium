#!/usr/bin/env python3
"""
Skills Assessment - IMDB Sentiment Analysis.

Trains a classifier on the IMDB dataset (50,000 movie reviews)
for binary sentiment analysis (positive=1, negative=0).

Uses TfidfVectorizer + SGDClassifier (log loss) to achieve 90%+ accuracy.
"""

import os
import sys
import json
import logging
import time
from typing import Any, Tuple, List, Dict

import requests
import zipfile
import io
from tqdm import tqdm

import dask.dataframe as dd
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


# Configuration
TARGET_IP: str = "localhost"
TARGET_URL: str = f"http://{TARGET_IP}:5000"
API_ENDPOINT: str = f"{TARGET_URL}/api/upload"
MODEL_FILE: str = "skills_assessment.joblib"
DATASET_URL: str = "https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip"

# NLTK path (project-local)
NLTK_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.insert(0, NLTK_PATH)

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s — %(message)s",
)
LOGGER = logging.getLogger(__name__)


def setup_nltk() -> None:
    """Download required NLTK data to project-local directory."""
    LOGGER.debug("Setting up NLTK data in %s", NLTK_PATH)

    packages: List[str] = ["punkt", "punkt_tab", "stopwords"]
    for pkg in tqdm(packages, desc="NLTK", unit="pkg"):
        if pkg == "stopwords":
            check_path: str = os.path.join(NLTK_PATH, "corpora", "stopwords")
        else:
            check_path = os.path.join(NLTK_PATH, "tokenizers", pkg)

        if os.path.exists(check_path):
            LOGGER.debug("Package %s already at %s", pkg, check_path)
            continue

        # Download from NLTK GitHub repo
        base_url: str = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"
        url: str = f"{base_url}/{'corpora' if pkg == 'stopwords' else 'tokenizers'}/{pkg}.zip"

        LOGGER.info("Downloading %s from %s...", pkg, url)
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size: int = int(response.headers.get("content-length", 0))

            chunks: List[bytes] = []
            with tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                ncols=60,
                desc=f"  {pkg}",
                leave=False,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                        pbar.update(len(chunk))

            zip_data = io.BytesIO(b"".join(chunks))
            with zipfile.ZipFile(zip_data) as z:
                pkg_type: str = "corpora" if pkg == "stopwords" else "tokenizers"
                z.extractall(os.path.join(NLTK_PATH, pkg_type))
                LOGGER.debug("Extracted %s to %s", pkg, check_path)
        except Exception as e:
            LOGGER.warning("Failed to download %s: %s", pkg, e)

    LOGGER.debug("NLTK setup complete")


def download_dataset() -> None:
    """Download IMDB dataset from HTB Academy."""
    if os.path.exists("train.json") and os.path.exists("test.json"):
        LOGGER.debug("Dataset already present, skipping download")
        return

    LOGGER.info("Downloading IMDB dataset from HTB...")
    try:
        response = requests.get(DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()
        total_size: int = int(response.headers.get("content-length", 0))

        chunks: List[bytes] = []
        with tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            ncols=60,
            desc="IMDB dataset"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    pbar.update(len(chunk))

        zip_data = io.BytesIO(b"".join(chunks))
        with zipfile.ZipFile(zip_data) as z:
            z.extractall(".")
            LOGGER.info("Extracted: %s", z.namelist())
    except Exception as e:
        LOGGER.error("Failed to download dataset: %s", e)
        sys.exit(1)


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load IMDB train and test datasets using dask."""
    LOGGER.info("Loading IMDB dataset with dask")

    train_ddf: DaskDataFrame = dd.read_json("train.json", lines=False)
    test_ddf: DaskDataFrame = dd.read_json("test.json", lines=False)

    train_df: pd.DataFrame = train_ddf.compute()
    test_df: pd.DataFrame = test_ddf.compute()

    LOGGER.debug("Loaded %d train, %d test samples", len(train_df), len(test_df))

    return train_df, test_df


def preprocess_message(message: str) -> str:
    """Preprocess a single text message - minimal preprocessing for best accuracy."""
    # Just lowercase - keep all words (including stop words) for sentiment
    return message.lower()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess all messages in a DataFrame."""
    LOGGER.info("Preprocessing %d messages (minimal - just lowercase)", len(df))
    df["text"] = df["text"].apply(lambda x: x.lower())
    return df


def train_model(df: pd.DataFrame) -> Any:
    """Train TfidfVectorizer + SGDClassifier (90%+ accuracy)."""
    LOGGER.info("Training Pipeline")

    X = df["text"]
    y = df["label"].apply(lambda x: 1 if x == "positive" or x == 1 else 0)

    # TfidfVectorizer with settings that achieve 90%+ accuracy
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=100000,
        stop_words=None,  # Keep stop words (important for negation)
    )

    # SGDClassifier with log loss = online Logistic Regression
    # These params achieved 90.50% test accuracy
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,      # Best param from GridSearchCV
        eta0=0.01,       # Best param from GridSearchCV
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )

    LOGGER.info("Training model with Tfidf + SGD (log loss)...")
    pipeline.fit(X, y)

    # Calculate local accuracy
    y_pred = pipeline.predict(X)
    accuracy: float = (y_pred == y).mean()
    LOGGER.info("Training accuracy: %.4f", accuracy)

    return pipeline


def evaluate_model(model: Any, test_df: pd.DataFrame) -> None:
    """Evaluate model on test data."""
    LOGGER.info("Evaluating on test set")

    # Preprocess test data
    test_df = preprocess_dataframe(test_df)

    X_test = test_df["text"]
    y_test = test_df["label"].apply(lambda x: 1 if x == "positive" or x == 1 else 0)

    # Predict
    y_pred = model.predict(X_test)
    test_accuracy: float = (y_pred == y_test).mean()
    LOGGER.info("Test accuracy: %.4f", test_accuracy)


def save_model(model: Any) -> None:
    """Save trained model to joblib file."""
    LOGGER.info("Saving %s", MODEL_FILE)
    joblib.dump(model, MODEL_FILE)
    size_kb: float = os.path.getsize(MODEL_FILE) / 1024
    LOGGER.info("Saved %s (%.2f KB)", MODEL_FILE, size_kb)


def submit_model() -> None:
    """Submit model to HTB evaluation portal."""
    LOGGER.info("Submitting model to %s", API_ENDPOINT)

    if not os.path.exists(MODEL_FILE):
        LOGGER.error("Model file %s not found", MODEL_FILE)
        return

    try:
        with open(MODEL_FILE, "rb") as f:
            files: Dict[str, Any] = {"model": f}
            response = requests.post(API_ENDPOINT, files=files, timeout=120)

        LOGGER.info("Server response status: %d", response.status_code)
        LOGGER.info("Response: %s", response.json())

        result: dict = response.json()
        if "flag" in result:
            LOGGER.info("=== FLAG: %s ===", result["flag"])
        elif "accuracy" in result:
            LOGGER.info("Accuracy: %.4f", result["accuracy"])
            if "flag" in result:
                LOGGER.info("=== FLAG: %s ===", result["flag"])

    except Exception as e:
        LOGGER.error("Submission failed: %s", e)


def print_bayes_theory() -> None:
    """Print Naive Bayes theory (Section 9)."""
    LOGGER.info("=== Section 9: Naive Bayes Theory ===")
    LOGGER.info("Bayes: P(A|B) = P(B|A) * P(A) / P(B)")
    LOGGER.info("Spam case: P(spam|words) = P(words|spam) * P(spam) / P(words)")
    LOGGER.info("Naive bit: P(words|spam) = P(w1|spam) * P(w2|spam) * ...")
    LOGGER.info("Pick whichever class gives the bigger posterior.")


def main() -> None:
    """Main execution flow."""
    start_time: float = time.time()

    LOGGER.info("=== HTB Skills Assessment -- IMDB Sentiment Analysis ===")
    LOGGER.info("Target: %s", TARGET_IP)

    # Section 9: Theory
    print_bayes_theory()

    # Setup NLTK (skip if already present)
    if not os.path.exists(os.path.join(NLTK_PATH, "tokenizers", "punkt")):
        setup_nltk()

    # Section 10: Download dataset
    LOGGER.info("=== Section 10: Dataset Download ===")
    download_dataset()

    # Section 11: Load and preprocess
    LOGGER.info("=== Section 11: Preprocessing ===")
    train_df, test_df = load_dataset()
    train_df = preprocess_dataframe(train_df)

    # Section 12-13: Train
    LOGGER.info("=== Section 12-13: Training ===")
    model = train_model(train_df)

    # Section 14: Evaluate
    LOGGER.info("=== Section 14: Evaluation ===")
    evaluate_model(model, test_df)

    # Save model
    save_model(model)

    # Section 25: Submit
    LOGGER.info("=== Section 25: Submission ===")
    submit_model()

    elapsed: float = time.time() - start_time
    LOGGER.info("=== All done in %.2fs ===", elapsed)


if __name__ == "__main__":
    main()
