---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 15 - Network Anomaly Detection"]
lead: Using random forests and NSL-KDD dataset to detect anomalous network traffic.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 15 - Network Anomaly Detection

`Anomaly detection` identifies data points that deviate significantly from established norms. In cybersecurity, such deviations can signal malicious activity: network intrusions, lateral movement, or data exfiltration. `Random forests` handle the high-dimensional, mixed-type feature spaces typical of network traffic data, making them a natural fit for this task.

## Random Forests

A `Random Forest` is an ensemble of `decision trees`. In classification, each tree votes for a class and the majority vote wins. In regression, predictions are averaged across trees. Combining many trees reduces the variance of any single tree, improving generalization without sacrificing much bias.

Three mechanisms drive random forest construction:

1. `Bootstrapping`: Each tree trains on a random sample drawn with replacement from the training set. Different trees therefore see different subsets of the data.
2. `Tree Construction`: At each node split, only a random subset of features is evaluated as split candidates. This de-correlates the trees and prevents a few dominant features from appearing in every tree.
3. `Voting`: After all trees are built, the ensemble classifies by majority vote (classification) or mean prediction (regression).

## Random Forests for Anomaly Detection

For anomaly detection, the random forest learns the structure of normal traffic. When the model evaluates a new connection, points that fall outside the learned normal region or receive low-confidence predictions are flagged as potential anomalies. This makes the approach well-suited for identifying suspicious network activity without requiring labeled attack examples for every attack variant.

## NSL-KDD Dataset

The `NSL-KDD` dataset improves on the original `KDD Cup 1999` dataset by removing redundant records and correcting the severe class imbalance that made the original unreliable for benchmarking. It provides balanced, labeled instances covering both normal traffic and four categories of attacks, supporting both binary (normal vs. attack) and multi-class detection tasks. A modified version of this dataset is used here.

## Downloading the Dataset

```python
import requests, zipfile, io

# URL for the NSL-KDD dataset
url = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip"

# Download the zip file and extract its contents
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')  # Extracts to the current directory
```

## Loading the Dataset

### Importing Libraries

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
```

- `numpy` and `pandas` handle data loading and cleaning.
- `RandomForestClassifier` is the model used for anomaly detection.
- `train_test_split` and the `sklearn.metrics` functions support evaluation.
- `seaborn` and `matplotlib` produce visualizations.

### Defining Column Names and File Path

The NSL-KDD dataset has no header row. Column names must be supplied manually to identify each feature and label:

```python
# Set the file path to the dataset file
file_path = r'KDD+.txt'

# Define the column names corresponding to the NSL-KDD dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]
```

The columns span generic traffic statistics (`duration`, `src_bytes`, `dst_bytes`), categorical fields (`protocol_type`, `service`), and the target labels (`attack`, `level`).

### Reading the Dataset into a DataFrame

```python
# Read the combined NSL-KDD dataset into a DataFrame
df = pd.read_csv(file_path, names=columns)
```

The result is a DataFrame with named columns ready for inspection, preprocessing, and model training:

```python
print(df.head())
```

---

## Summary

- Anomaly detection identifies data points that deviate significantly from established norms, flagging potential security events.
- Random forests are an ensemble of decision trees where each tree votes and the majority vote determines the final classification.
- Three mechanisms drive random forests: bootstrapping (random data subsets per tree), random feature subsets at each split, and majority voting.
- Random forests generalize better than single decision trees by reducing variance through ensemble averaging.
- The NSL-KDD dataset improves on KDD Cup 1999 by removing redundant records and correcting class imbalance.
- NSL-KDD supports both binary (normal vs. attack) and multi-class (DoS, Probe, Privilege, Access) detection tasks.

---

## Best Practices

- Set `random_state` on `RandomForestClassifier` to ensure reproducible tree construction across runs and experiments.
- Supply column names explicitly when loading header-less datasets like NSL-KDD to avoid silent misalignment errors.
- Use multi-class targets in addition to binary targets — collapsing everything to "normal vs. attack" hides the nature of threats.
- Import `classification_report` from `sklearn.metrics` alongside confusion matrix for per-class precision, recall, and F1 breakdowns.
- Visualize the confusion matrix with `seaborn.heatmap` to quickly spot which attack categories the model struggles to distinguish.

---

## Quiz

**Q1:** What is bootstrapping in the context of random forests?
> Each tree in the forest trains on a different random sample of the training data drawn with replacement, so different trees see different data subsets, which de-correlates their errors and improves ensemble generalization.

**Q2:** Why does the NSL-KDD dataset improve on the original KDD Cup 1999 dataset?
> NSL-KDD removes redundant records and corrects the severe class imbalance present in the original, making it a more reliable benchmark for intrusion detection.

**Q3:** What are the four attack categories in the NSL-KDD multi-class classification target?
> DoS (Denial of Service), Probe (reconnaissance), Privilege Escalation, and Access attacks.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-16-Preprocessing-and-Splitting-the-Dataset]] — preprocessing and splitting the NSL-KDD dataset is the direct follow-on step
- see:: [[Section-9-Spam-Classification]] — parallel module structure: both sections introduce the algorithm before dataset exploration
- see:: [[Section-8-Metrics-for-Evaluating-a-Model]] — accuracy, precision, recall, and F1 used to evaluate the random forest model

**Terms**
- anomaly detection, random forest, decision tree, ensemble learning, bootstrapping, majority voting, NSL-KDD, KDD Cup 1999, intrusion detection, overfitting
