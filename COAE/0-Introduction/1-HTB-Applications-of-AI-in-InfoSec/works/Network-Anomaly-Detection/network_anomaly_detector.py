"""
Network Anomaly Detection using Random Forests and NSL-KDD dataset.

This script implements a complete pipeline for detecting network anomalies
using a Random Forest classifier trained on the NSL-KDD dataset.
"""

import logging
import os
import zipfile
from io import BytesIO
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
TARGET_IP: str = "localhost" #CHANGE THIS TO YOUR TARGET
TARGET_PORT: int = 8001
MODEL_FILENAME: str = "network_anomaly_detection_model.joblib"
DATASET_URL: str = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip"
DATASET_FILE: str = "KDD+.txt"

# Column names for NSL-KDD dataset
COLUMNS: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack",
    "level",
]

# Numeric features for the model
NUMERIC_FEATURES: list[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Attack category mappings
DOS_ATTACKS: list[str] = [
    "apache2",
    "back",
    "land",
    "neptune",
    "mailbomb",
    "pod",
    "processtable",
    "smurf",
    "teardrop",
    "udpstorm",
    "worm",
]

PROBE_ATTACKS: list[str] = [
    "ipsweep",
    "mscan",
    "nmap",
    "portsweep",
    "saint",
    "satan",
]

PRIVILEGE_ATTACKS: list[str] = [
    "buffer_overflow",
    "loadmdoule",
    "perl",
    "ps",
    "rootkit",
    "sqlattack",
    "xterm",
]

ACCESS_ATTACKS: list[str] = [
    "ftp_write",
    "guess_passwd",
    "http_tunnel",
    "imap",
    "multihop",
    "named",
    "phf",
    "sendmail",
    "snmpgetattack",
    "snmpguess",
    "spy",
    "warezclient",
    "warezmaster",
    "xclock",
    "xsnoop",
]

CLASS_LABELS: list[str] = ["Normal", "DoS", "Probe", "Privilege", "Access"]

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def download_dataset() -> None:
    """Download the NSL-KDD dataset ZIP file and extract contents.

    Downloads from the HTB academy URL with a progress bar,
    then extracts the KDD+ dataset file to the project directory.
    """
    if os.path.exists(DATASET_FILE):
        logger.debug("Dataset file already exists, skipping download")
        return

    logger.debug("Downloading NSL-KDD dataset from %s", DATASET_URL)
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(
        total=total_size, unit="B", unit_scale=True, ncols=60, desc="Downloading dataset"
    ) as progress_bar:
        buffer = BytesIO()
        for data in response.iter_content(block_size):
            buffer.write(data)
            progress_bar.update(len(data))

    logger.debug("Extracting dataset ZIP file")
    with zipfile.ZipFile(buffer) as zip_ref:
        zip_ref.extractall(".")

    logger.debug("Dataset extracted to project directory")


def load_dataset() -> pd.DataFrame:
    """Load the NSL-KDD dataset into a pandas DataFrame.

    Returns:
        DataFrame containing the loaded dataset with named columns.
    """
    logger.debug("Loading dataset from %s", DATASET_FILE)
    df = pd.read_csv(DATASET_FILE, names=COLUMNS)
    logger.debug("Dataset loaded with shape: %s", df.shape)
    return df


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the dataset for model training.

    Creates binary and multi-class targets, one-hot encodes categorical
    features, and selects numeric features.

    Args:
        df: Raw dataset DataFrame.

    Returns:
        Tuple containing the feature matrix and multi-class target.
    """
    logger.debug("Starting preprocessing")

    # Binary classification target
    df["attack_flag"] = df["attack"].apply(lambda a: 0 if a == "normal" else 1)
    logger.debug("Binary target created (attack_flag)")

    # Multi-class classification target
    def map_attack(attack: str) -> int:
        if attack in DOS_ATTACKS:
            return 1
        elif attack in PROBE_ATTACKS:
            return 2
        elif attack in PRIVILEGE_ATTACKS:
            return 3
        elif attack in ACCESS_ATTACKS:
            return 4
        else:
            return 0

    df["attack_map"] = df["attack"].apply(map_attack)
    logger.debug("Multi-class target created (attack_map)")

    # Encode categorical variables
    features_to_encode = ["protocol_type", "service"]
    encoded = pd.get_dummies(df[features_to_encode])
    logger.debug("Categorical features one-hot encoded")

    # Combine encoded and numeric features
    train_set = encoded.join(df[NUMERIC_FEATURES])
    multi_y = df["attack_map"]
    logger.debug("Feature matrix prepared with shape: %s", train_set.shape)

    return train_set, multi_y


def split_data(
    train_set: pd.DataFrame, multi_y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Split the dataset into training, validation, and test sets.

    Performs an 80/20 train/test split, then further splits training
    into 70/30 train/validation.

    Args:
        train_set: Feature matrix.
        multi_y: Multi-class target series.

    Returns:
        Tuple containing train_X, test_X, train_y, test_y, multi_train_X,
        multi_train_y, multi_val_X, multi_val_y.
    """
    logger.debug("Splitting data into train and test sets")
    train_X, test_X, train_y, test_y = train_test_split(
        train_set, multi_y, test_size=0.2, random_state=1337
    )
    logger.debug(
        "Train size: %d, Test size: %d", len(train_X), len(test_X)
    )

    logger.debug("Splitting training data into train and validation sets")
    multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(
        train_X, train_y, test_size=0.3, random_state=1337
    )
    logger.debug(
        "Train size: %d, Validation size: %d",
        len(multi_train_X),
        len(multi_val_X),
    )

    return (
        train_X,
        test_X,
        train_y,
        test_y,
        multi_train_X,
        multi_train_y,
        multi_val_X,
        multi_val_y,
    )


def train_model(
    multi_train_X: pd.DataFrame, multi_train_y: pd.Series
) -> RandomForestClassifier:
    """Train a Random Forest classifier for multi-class classification.

    Args:
        multi_train_X: Training feature matrix.
        multi_train_y: Training target values.

    Returns:
        Trained RandomForestClassifier model.
    """
    logger.debug("Initializing RandomForestClassifier with random_state=1337")
    model = RandomForestClassifier(random_state=1337)

    logger.debug("Training model on %d samples", len(multi_train_X))
    model.fit(multi_train_X, multi_train_y)
    logger.debug("Model training complete")

    return model


def evaluate_model(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
) -> Dict[str, float]:
    """Evaluate the model on a given dataset.

    Computes accuracy, precision, recall, and F1-score using weighted
    averaging for multi-class imbalance. Also generates a confusion
    matrix heatmap and classification report.

    Args:
        model: Trained RandomForestClassifier.
        X: Feature matrix.
        y: True target values.
        dataset_name: Name of the dataset for display.

    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.debug("Evaluating model on %s", dataset_name)
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average="weighted")
    recall = recall_score(y, predictions, average="weighted")
    f1 = f1_score(y, predictions, average="weighted")

    logger.debug(
        "%s — Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        dataset_name,
        accuracy,
        precision,
        recall,
        f1,
    )

    print(f"\n{dataset_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(y, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title(f"Network Anomaly Detection - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png")
    plt.close()
    logger.debug("Confusion matrix saved to confusion_matrix_%s.png", dataset_name.lower())

    # Classification report
    print(f"\nClassification Report for {dataset_name}:")
    print(classification_report(y, predictions, target_names=CLASS_LABELS))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_model(model: RandomForestClassifier) -> None:
    """Save the trained model to a joblib file.

    Args:
        model: Trained RandomForestClassifier to serialize.
    """
    logger.debug("Saving model to %s", MODEL_FILENAME)
    joblib.dump(model, MODEL_FILENAME)
    logger.debug("Model saved successfully")


def upload_model() -> None:
    """Upload the trained model to the HTB evaluation portal.

    Sends the joblib file to the /api/upload endpoint and prints
    the server response.
    """
    url = f"http://{TARGET_IP}:{TARGET_PORT}/api/upload"
    logger.debug("Uploading model to %s", url)

    try:
        with open(MODEL_FILENAME, "rb") as model_file:
            files = {"model": model_file}
            response = requests.post(url, files=files)

        import json

        print(json.dumps(response.json(), indent=4))
        logger.debug("Upload response status: %d", response.status_code)

    except Exception as e:
        logger.error("Failed to upload model: %s", str(e))


def main() -> None:
    """Run the complete network anomaly detection pipeline.

    Executes dataset download, preprocessing, splitting, training,
    evaluation, model saving, and optional upload.
    """
    logger.debug("Starting network anomaly detection pipeline")

    # Download and load dataset
    download_dataset()
    df = load_dataset()

    # Preprocess data
    train_set, multi_y = preprocess_data(df)

    # Split data
    (
        train_X,
        test_X,
        train_y,
        test_y,
        multi_train_X,
        multi_train_y,
        multi_val_X,
        multi_val_y,
    ) = split_data(train_set, multi_y)

    # Train model
    model = train_model(multi_train_X, multi_train_y)

    # Evaluate on validation set
    evaluate_model(model, multi_val_X, multi_val_y, "Validation Set")

    # Evaluate on test set
    evaluate_model(model, test_X, test_y, "Test Set")

    # Save model
    save_model(model)

    # Upload model to HTB portal
    upload_model()

    logger.debug("Pipeline complete")


if __name__ == "__main__":
    main()
