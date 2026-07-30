"""
HTB Spam Classification Module — End-to-end pipeline.

Target: http://localhost:8000
Flag: HTB{sp4m_cla55if13r_3v4lu4t0r}
"""

import io
import os
import re
import sys
import time
import logging
import zipfile
from typing import List

import dask.dataframe as dd
import joblib
import pandas as pd
import requests
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S"
)
LOG = logging.getLogger(__name__)

TARGET_IP: str = "localhost" #CHANGE THIS TO YOUR TARGET
TARGET_URL: str = f"http://{TARGET_IP}:8000/api/upload"
DATASET_URL: str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
NLTK_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
STOP_WORDS: set = set()
STEMMER: PorterStemmer = PorterStemmer()


def print_bayes_theory() -> None:
    """Display Bayes formula and how it applies to spam filtering."""
    LOG.info("=== Section 9: Naive Bayes Theory ===")
    LOG.info("Bayes: P(A|B) = P(B|A) * P(A) / P(B)")
    LOG.info("Spam case: P(spam|words) = P(words|spam) * P(spam) / P(words)")
    LOG.info("Naive bit: P(words|spam) = P(w1|spam) * P(w2|spam) * ...")
    LOG.info("Pick whichever class gives the bigger posterior.")


def setup_nltk() -> None:
    """Download punkt, punkt_tab, stopwords into project's nltk_data folder."""
    start: float = time.time()
    LOG.info("Setting up NLTK data in: %s", NLTK_PATH)

    os.makedirs(NLTK_PATH, exist_ok=True)

    import nltk
    nltk.data.path.append(NLTK_PATH)

    for pkg in tqdm(["punkt", "punkt_tab", "stopwords"], desc="NLTK", unit="pkg", file=sys.stdout):
        if pkg == "stopwords":
            check_path = os.path.join(NLTK_PATH, "corpora", "stopwords")
        else:
            check_path = os.path.join(NLTK_PATH, "tokenizers", pkg)

        if os.path.exists(check_path):
            LOG.debug("Package %s already at %s", pkg, check_path)
        else:
            LOG.info("Downloading %s...", pkg)
            nltk.download(pkg, download_dir=NLTK_PATH, quiet=True)
            LOG.debug("Downloaded %s", pkg)

    global STOP_WORDS
    STOP_WORDS.update(set(stopwords.words("english")))
    LOG.info("NLTK ready in %.2fs", time.time() - start)


def download_dataset() -> None:
    """Download and extract the UCI SMS Spam Collection zip file."""
    start: float = time.time()
    extract_to: str = "sms_spam_collection"

    if os.path.exists(extract_to) and os.listdir(extract_to):
        LOG.info("Dataset already in %s (%.2fs)", extract_to, time.time() - start)
        return

    LOG.info("Downloading dataset from %s", DATASET_URL)
    response = requests.get(DATASET_URL, timeout=30, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    content = io.BytesIO()
    if total_size > 0:
        pbar = tqdm(
            total=total_size, unit="B", unit_scale=True,
            desc="Dataset", ncols=60, file=sys.stdout
        )
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                content.write(chunk)
                pbar.update(len(chunk))
        pbar.close()
    else:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                content.write(chunk)

    content.seek(0)
    with zipfile.ZipFile(content) as z:
        z.extractall(extract_to)
        LOG.debug("Files: %s", z.namelist())

    LOG.info("Download done in %.2fs", time.time() - start)


def load_dataset() -> dd.DataFrame:
    """Read SMSSpamCollection with dask, tag label as categorical."""
    start: float = time.time()
    path: str = "sms_spam_collection/SMSSpamCollection"

    df = dd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].astype("category")
    n_rows: int = df.shape[0].compute()
    LOG.info("Loaded %d rows in %.2fs", n_rows, time.time() - start)
    return df


def validate_dataset(df: dd.DataFrame) -> None:
    """Print row counts, dtypes, and label distribution."""
    LOG.info("=== Dataset Validation ===")
    pdf: pd.DataFrame = df.compute()
    LOG.debug("Dtypes:\n%s", pdf.dtypes)
    LOG.debug("Nulls: %s", pdf.isnull().sum().to_string())
    LOG.debug("Label counts:\n%s", pdf["label"].value_counts().to_string())
    LOG.info("Validated -- %d rows, %d ham, %d spam",
             len(pdf), (pdf["label"] == "ham").sum(), (pdf["label"] == "spam").sum())


def preprocess_message(msg: str) -> str:
    """Clean one SMS: lowercase, strip punctuation, remove stop words, stem."""
    msg = msg.lower()
    msg = re.sub(r"[^a-z\s$!]", "", msg)
    tokens = word_tokenize(msg)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)


def preprocess_dataframe(df: dd.DataFrame) -> dd.DataFrame:
    """Apply preprocess_message to every row in the dataframe."""
    start: float = time.time()
    LOG.info("Preprocessing messages...")

    df = df.copy()
    df["message"] = df["message"].apply(preprocess_message, meta=("message", "object"))

    sample = df.head(3)
    LOG.debug("Sample output: %s", sample["message"].tolist())
    LOG.info("Preprocessing done in %.2fs", time.time() - start)
    return df


def train_pipeline(messages: List[str], labels: List[int]) -> Pipeline:
    """Train a Pipeline with CountVectorizer + MultinomialNB via GridSearchCV."""
    start: float = time.time()
    LOG.info("=== Training Pipeline ===")

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(min_df=1, max_df=0.7, ngram_range=(1, 3))),
        ("classifier", MultinomialNB())
    ])

    param_grid = {
        "classifier__alpha": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", verbose=1)
    grid_search.fit(messages, labels)

    LOG.info("Best params: %s", grid_search.best_params_)
    LOG.info("Best accuracy: %.4f", grid_search.best_score_)
    LOG.info("Training done in %.2fs", time.time() - start)
    return grid_search.best_estimator_


def evaluate_sample_messages(model: Pipeline) -> None:
    """Classify a few hardcoded messages and print the results."""
    LOG.info("=== Evaluation ===")

    test_messages: List[str] = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
        "Hey, are we still meeting up for lunch today?",
        "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
        "Reminder: Your appointment is scheduled for tomorrow at 10am.",
        "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
    ]

    processed: List[str] = [preprocess_message(msg) for msg in test_messages]
    predictions = model.predict(processed)
    probs = model.predict_proba(processed)

    for i, msg in enumerate(test_messages):
        pred = "Spam" if predictions[i] == 1 else "Not-Spam"
        LOG.info("Msg: %s", msg[:50] + "...")
        LOG.info("  -> %s (spam prob: %.2f)", pred, probs[i][1])


def save_model(model: Pipeline) -> None:
    """Persist the trained Pipeline to spam_detection_model.joblib."""
    start: float = time.time()
    filename: str = "spam_detection_model.joblib"
    joblib.dump(model, filename)
    size_kb: float = os.path.getsize(filename) / 1024
    LOG.info("Saved %s (%.2f KB, %.2fs)", filename, size_kb, time.time() - start)


def upload_model() -> dict:
    """POST the joblib file to the HTB evaluation endpoint, return JSON response."""
    start: float = time.time()
    filename: str = "spam_detection_model.joblib"

    LOG.info("Uploading %s to %s", filename, TARGET_URL)

    with open(filename, "rb") as f:
        response = requests.post(TARGET_URL, files={"model": f}, timeout=30)

    LOG.debug("HTTP %d", response.status_code)
    result: dict = response.json()
    LOG.info("Server says: %s", result)
    LOG.info("Upload done in %.2fs", time.time() - start)
    return result


def main() -> None:
    """Orchestrate the full pipeline: theory, data, training, evaluation, upload."""
    total_start: float = time.time()
    LOG.info("=== HTB Spam Classification -- Full Pipeline ===")
    LOG.info("Target: %s", TARGET_IP)

    print_bayes_theory()

    setup_nltk()
    download_dataset()
    df = load_dataset()
    validate_dataset(df)

    df = df.drop_duplicates()
    LOG.debug("After dedup: %d rows", df.shape[0].compute())

    df = preprocess_dataframe(df)
    pdf = df.compute()

    messages: List[str] = pdf["message"].tolist()
    labels: List[int] = [1 if x == "spam" else 0 for x in pdf["label"].tolist()]

    model = train_pipeline(messages, labels)
    evaluate_sample_messages(model)
    save_model(model)

    result = upload_model()
    flag = result.get("flag", "No flag returned")
    LOG.info("=== FLAG: %s ===", flag)

    LOG.info("=== All done in %.2fs ===", time.time() - total_start)


if __name__ == "__main__":
    main()
