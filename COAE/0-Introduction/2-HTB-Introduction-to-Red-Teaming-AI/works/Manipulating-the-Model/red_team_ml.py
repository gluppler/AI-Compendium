"""HTB Red Teaming ML -- Section 3 labs.

Demonstrates input manipulation (ML01), data poisoning (ML02), and model
theft (ML05) against a Naive Bayes spam classifier.

Target: localhost:8000
"""

import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

HERE: str = os.path.dirname(os.path.abspath(__file__))
TARGET: str = "localhost"
PORT: str = "8000"
BASE_URL: str = f"http://{TARGET}:{PORT}"
NLTK_PATH: Path = Path(HERE) / "nltk_data"
DATASET_PATH: str = os.path.join(HERE, "server_train.csv")
POISON_PATH: str = os.path.join(HERE, "poisoned_train.csv")
MODEL_PATH: str = os.path.join(HERE, "server_model.bin")

Q1_FLAG: str = "HTB{9b8de0fd17f2166743cd59f7ec876ac7}"
Q2_FLAG: str = "HTB{8ba5eff39c343c3b0170e6bb1704df02}"
Q3_FLAG: str = "8007cd6c209a40399cf3ca82dd7db02c"

_STOP_WORDS: set = set()
_STEMMER: PorterStemmer = PorterStemmer()


def setup_nltk() -> None:
    """Download NLTK data into the project-local nltk_data directory.

    Downloads punkt, punkt_tab, and stopwords corpora if not already
    present. Removes spam-indicative stop words ("free", "win", "cash",
    "urgent") from the global stop word set after download.

    Raises:
        nltk.DownloadError: If an NLTK package fails to download.
    """
    start: float = time.time()
    LOG.info("Setting up NLTK in %s", NLTK_PATH)
    NLTK_PATH.mkdir(exist_ok=True)

    import nltk
    nltk.data.path.append(str(NLTK_PATH))

    for pkg in tqdm(["punkt", "punkt_tab", "stopwords"], desc="NLTK", unit="pkg", file=sys.stdout):
        check: Path = (NLTK_PATH / "corpora" / pkg) if pkg == "stopwords" else (NLTK_PATH / "tokenizers" / pkg)
        if check.exists():
            LOG.debug("%s already present", pkg)
            continue
        LOG.info("Downloading %s...", pkg)
        nltk.download(pkg, download_dir=str(NLTK_PATH), quiet=True)

    global _STOP_WORDS
    _STOP_WORDS = set(stopwords.words("english")) - {"free", "win", "cash", "urgent"}
    LOG.info("NLTK ready in %.2fs", time.time() - start)


def preprocess_message(message: str) -> str:
    """Clean one SMS message for classification.

    Lowercases the text, strips non-alphanumeric characters (except $ and !),
    tokenizes with NLTK, removes stop words, and applies Porter stemming.

    Args:
        message: Raw SMS text to clean.

    Returns:
        Preprocessed message string with space-separated tokens.
    """
    msg: str = message.lower()
    msg = re.sub(r"[^a-z\s$!]", "", msg)
    tokens: List[str] = word_tokenize(msg)
    tokens = [_STEMMER.stem(w) for w in tokens if w not in _STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to every message and remove duplicates.

    Args:
        df: DataFrame with a "message" column containing raw SMS text.

    Returns:
        DataFrame with preprocessed messages and duplicate rows removed.
    """
    df = df.copy()
    df["message"] = df["message"].apply(preprocess_message)
    return df.drop_duplicates()


def classify_messages(
    model: Pipeline,
    msg_df: Union[str, List[str]],
    return_probabilities: bool = False,
) -> Any:
    """Classify messages using a trained model.

    Args:
        model: Trained sklearn Pipeline.
        msg_df: Single message string or list of strings.
        return_probabilities: If True, return per-class probability arrays.

    Returns:
        Predicted class (int) or probability array (ndarray).
    """
    if isinstance(msg_df, str):
        msg_preprocessed: List[str] = [preprocess_message(msg_df)]
    else:
        msg_preprocessed = [preprocess_message(msg) for msg in msg_df]

    if return_probabilities:
        return model.predict_proba(msg_preprocessed)
    return model.predict(msg_preprocessed)


def train(dataset: str) -> Pipeline:
    """Train a Naive Bayes classifier pipeline from a CSV file.

    Args:
        dataset: Path to the training CSV with label and message columns.

    Returns:
        Trained sklearn Pipeline.
    """
    start: float = time.time()
    df: pd.DataFrame = preprocess_dataframe(pd.read_csv(dataset))
    messages: List[str] = df["message"].tolist()
    labels: List[int] = [1 if x == "spam" else 0 for x in df["label"].tolist()]

    pipeline: Pipeline = Pipeline([
        ("vectorizer", CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ])
    pipeline.fit(messages, labels)
    LOG.info("Training done in %.2fs (samples=%d)", time.time() - start, len(messages))
    return pipeline


# ---------------------------------------------------------------------------
# Lab 1: Input Manipulation (ML01)
# ---------------------------------------------------------------------------


def lab1_input_manipulation() -> str:
    """Demonstrate input manipulation via overpowering.

    Trains a local model on the server dataset, finds an overpowering text
    that flips the fixed spam message to ham, and submits it to the server.

    Returns:
        Flag from the input manipulation lab.
    """
    LOG.info("=" * 60)
    LOG.info("Lab 1: Input Manipulation (ML01)")
    LOG.info("=" * 60)

    setup_nltk()

    if not os.path.exists(DATASET_PATH):
        LOG.info("Downloading dataset from server...")
        resp = requests.get(f"{BASE_URL}/data_poisoning/download", timeout=15)
        resp.raise_for_status()
        with open(DATASET_PATH, "wb") as f:
            f.write(resp.content)
        LOG.info("Saved %d rows", len(pd.read_csv(DATASET_PATH)))

    model: Pipeline = train(DATASET_PATH)

    fixed: str = "Congratulations! You've won a $1000 Walmart gift card. Go to https://bit.ly/3YCN7PF to claim now."
    overpower: str = (
        "the quick brown fox jumps over the lazy dog the five boxing wizards "
        "jump quickly how vexingly quick daft zebras jump sphinx of black quartz "
        "judge my vow pack my box with five dozen liquor jugs the jay pig fox "
        "zebra and my wolves quack blowzy red vixens fight for a quick jump cozy "
        "sphinx waves quart jug of bad milk a quick movement of the enemy will "
        "jeopardize six gunboats all questions asked by five watched experts amaze "
        "the judge jack quietly moved up front and seized the big ball of wax few "
        "black taxis drive up the major roads on soft hazy Sunday mornings heavy "
        "boxes perform full wavelength dives glib jocks quiz nymphs to vex dwarf "
        "veg big fjords vex quick waltz nymph baffled by the complexities of the "
        "human mind the quick brown fox jumps over the lazy dog crazy Fredrick "
        "bought many very exquisite opal jewels sixty zippers were quickly picked "
        "from the woven jute bag"
    )

    prob = classify_messages(model, fixed + " " + overpower, return_probabilities=True)[0]
    LOG.info("Overpowered -> Ham: %.2f%% Spam: %.2f%%", prob[0] * 100, prob[1] * 100)

    LOG.info("Submitting to %s/input_manipulation ...", BASE_URL)
    resp = requests.post(f"{BASE_URL}/input_manipulation", data={"prompt": overpower}, timeout=15)

    flag_match: re.Match[str] | None = re.search(r"HTB\{[^}]+\}", resp.text)
    if flag_match:
        flag: str = flag_match.group(0)
        LOG.info("FLAG CAPTURED: %s", flag)
    else:
        LOG.warning("Flag not found in response. Checking response...")
        if "Not Spam" in resp.text:
            LOG.info("Classification successful (Not Spam), but flag format unexpected.")
        else:
            LOG.warning("Unexpected response. Saving to debug.html")
            with open(os.path.join(HERE, "debug_q1.html"), "w") as f:
                f.write(resp.text)
        flag = ""

    return flag


# ---------------------------------------------------------------------------
# Lab 2: Data Poisoning (ML02)
# ---------------------------------------------------------------------------


def lab2_data_poisoning() -> str:
    """Demonstrate data poisoning by injecting noisy training samples.

    Downloads the original dataset, creates a poisoned version with random
    labels, and uploads it to the server so the trained classifier accuracy
    drops below 70%.

    Returns:
        Flag from the data poisoning lab.
    """
    LOG.info("=" * 60)
    LOG.info("Lab 2: Data Poisoning (ML02)")
    LOG.info("=" * 60)

    if not os.path.exists(DATASET_PATH):
        LOG.info("Downloading dataset from server...")
        resp = requests.get(f"{BASE_URL}/data_poisoning/download", timeout=15)
        resp.raise_for_status()
        with open(DATASET_PATH, "wb") as f:
            f.write(resp.content)

    df: pd.DataFrame = pd.read_csv(DATASET_PATH)
    LOG.info("Original: %d rows (%s)", len(df), df["label"].value_counts().to_dict())

    small: pd.DataFrame = pd.concat([
        df[df["label"] == "ham"].head(30),
        df[df["label"] == "spam"].head(10),
    ], ignore_index=True)

    rng = np.random.default_rng(42)
    n_noise: int = min(300, len(df))
    noise_msgs: List[str] = df["message"].sample(n=n_noise, random_state=42).tolist()
    noise: List[dict] = []
    for msg in noise_msgs:
        noise.append({"label": rng.choice(["ham", "spam"]), "message": msg})

    poisoned: pd.DataFrame = pd.concat([small, pd.DataFrame(noise)], ignore_index=True)
    poisoned.to_csv(POISON_PATH, index=False)
    LOG.info("Poisoned: %d rows (%s)", len(poisoned), poisoned["label"].value_counts().to_dict())

    LOG.info("Uploading to %s/data_poisoning ...", BASE_URL)
    with open(POISON_PATH, "rb") as f:
        resp = requests.post(f"{BASE_URL}/data_poisoning", files={"file": f}, timeout=30)

    flag_match = re.search(r"HTB\{[^}]+\}", resp.text)
    if flag_match:
        flag = flag_match.group(0)
        LOG.info("FLAG CAPTURED: %s", flag)
    else:
        LOG.warning("Flag not found. Saving response to debug_q2.html")
        with open(os.path.join(HERE, "debug_q2.html"), "w") as f:
            f.write(resp.text)
        m = re.search(r"accuracy is <b>([\d.]+)%", resp.text)
        if m:
            LOG.info("Server accuracy: %s%%", m.group(1))
        flag = ""

    return flag


# ---------------------------------------------------------------------------
# Lab 3: Model Theft (ML05)
# ---------------------------------------------------------------------------


def lab3_model_theft() -> str:
    """Demonstrate model theft by downloading the server model.

    Downloads the trained model from /model, computes its MD5 hash,
    and returns the hash as the flag.

    Returns:
        MD5 hash of the downloaded model.
    """
    LOG.info("=" * 60)
    LOG.info("Lab 3: Model Theft (ML05)")
    LOG.info("=" * 60)

    LOG.info("Downloading model from %s/model ...", BASE_URL)
    resp = requests.get(f"{BASE_URL}/model", timeout=30)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(resp.content)

    md5_hash: str = hashlib.md5(resp.content).hexdigest()
    LOG.info("Model MD5: %s", md5_hash)

    if md5_hash == Q3_FLAG:
        LOG.info("FLAG CAPTURED: %s", md5_hash)
    else:
        LOG.warning("MD5 mismatch: got %s, expected %s", md5_hash, Q3_FLAG)

    return md5_hash if md5_hash == Q3_FLAG else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all three Section 3 labs against the target server."""
    total_start: float = time.time()
    LOG.info("=" * 60)
    LOG.info("HTB Red Teaming ML -- Section 3 Labs")
    LOG.info("Target: %s:%s", TARGET, PORT)
    LOG.info("=" * 60)

    q1: str = lab1_input_manipulation()
    LOG.info("")
    q2: str = lab2_data_poisoning()
    LOG.info("")
    q3: str = lab3_model_theft()
    LOG.info("")

    LOG.info("=" * 60)
    LOG.info("All labs complete in %.2fs", time.time() - total_start)
    LOG.info("Flags:")
    LOG.info("  Q1 (Input Manipulation): %s", q1)
    LOG.info("  Q2 (Data Poisoning):     %s", q2)
    LOG.info("  Q3 (Model Theft):        %s", q3)
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()
