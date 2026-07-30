---
tags:
  - type/structure
  - structure/home
  - theme/machine-learning
aliases:
  - Home
  - HTB Applications of AI Home
lead: Entry point for the HTB Applications of AI in InfoSec module — practical ML for spam classification, network anomaly detection, and malware classification.
created: 2026-04-27
modified: 2026-04-27
---

# HTB | Applications of AI in InfoSec

A practical module for building AI models applied to cybersecurity domains. Uses Python, Scikit-learn, PyTorch, and JupyterLab. Covers the full ML pipeline from raw data to model evaluation across three real-world security problems.

See [Module Description](HTB-COAE-Prep/0-Introduction/1-HTB-Applications-of-AI-in-InfoSec/0-Module-Information/Module-Description.md) for prerequisites and setup.

---

## 0 — Module information

- [Module Description](HTB-COAE-Prep/0-Introduction/1-HTB-Applications-of-AI-in-InfoSec/0-Module-Information/Module-Description.md) — scope, environment requirements, key libraries
- [Conclusion](HTB-COAE-Prep/0-Introduction/1-HTB-Applications-of-AI-in-InfoSec/0-Module-Information/Conclusion.md) — closing reflection

---

## 1 — Introduction

- [Section 1 - Introduction](HTB-COAE-Prep/0-Introduction/1-HTB-Applications-of-AI-in-InfoSec/1-Introduction/Section-1-Introduction.md) — module overview and objectives
- [Section 2 - Environment Setup](1-Introduction/Section-2-Environment-Setup.md) — Miniconda, virtual environments, dependencies
- [Section 3 - JupyterLab](1-Introduction/Section-3-JupyterLab.md) — interactive development, notebook workflows
- [Section 4 - Python Libraries for AI](1-Introduction/Section-4-Python-Libraries-for-AI.md) — Scikit-learn, PyTorch, pandas, NumPy
- [Section 5 - Datasets](1-Introduction/Section-5-Datasets.md) — dataset structure, loading, inspection
- [Section 6 - Data Preprocessing](1-Introduction/Section-6-Data-Preprocessing.md) — cleaning, imputation, encoding, skew handling
- [Section 7 - Data Transformation](1-Introduction/Section-7-Data-Transformation.md) — one-hot encoding, train/test splits
- [Section 8 - Metrics for Evaluating a Model](1-Introduction/Section-8-Metrics-for-Evaluating-a-Model.md) — accuracy, precision, recall, F1, ROC

---

## 2 — Spam classification

- [Section 9 - Spam Classification](2-Spam-Classification/Section-9-Spam-Classification.md) — problem overview: text → numerical features → Naive Bayes
- [Section 10 - The Spam Dataset](2-Spam-Classification/Section-10-The-Spam-Dataset.md) — dataset exploration
- [Section 11 - Preprocessing the Spam Dataset](2-Spam-Classification/Section-11-Preprocessing-the-Spam-Dataset.md) — text cleaning and tokenisation
- [Section 12 - Feature Extraction](2-Spam-Classification/Section-12-Feature-Extraction.md) — TF-IDF, bag-of-words
- [Section 13 - Training and Evaluation (Spam Detection)](2-Spam-Classification/Section-13-Training-and-Evaluation-Spam-Detection.md) — Naive Bayes training
- [Section 14 - Model Evaluation (Spam Detection)](2-Spam-Classification/Section-14-Model-Evaluation-Spam-Detection.md) — metrics, confusion matrix

---

## 3 — Network anomaly detection

- [Section 15 - Network Anomaly Detection](3-Network-Anomaly-Detection/Section-15-Network-Anomaly-Detection.md) — problem overview: NSL-KDD dataset, random forests
- [Section 16 - Preprocessing and Splitting the Dataset](3-Network-Anomaly-Detection/Section-16-Preprocessing-and-Splitting-the-Dataset.md) — feature engineering for network data
- [Section 17 - Training and Evaluation (Network Anomaly Detection)](3-Network-Anomaly-Detection/Section-17-Training-and-Evaluation-Network-Anomaly-Detection.md) — random forest training
- [Section 18 - Model Evaluation (Network Anomaly Detection)](3-Network-Anomaly-Detection/Section-18-Model-Evaluation-Network-Anomaly-Detection.md) — metrics, feature importance

---

## 4 — Malware classification

- [Section 19 - Malware Classification](4-Malware-Classification/Section-19-Malware-Classification.md) — problem overview: binary → image → ResNet50
- [Section 20 - The Malware Dataset](4-Malware-Classification/Section-20-The-Malware-Dataset.md) — dataset structure and visualisation
- [Section 21 - Preprocessing the Malware Dataset](4-Malware-Classification/Section-21-Preprocessing-the-Malware-Dataset.md) — binary-to-image conversion
- [Section 22 - The Model](4-Malware-Classification/Section-22-The-Model.md) — ResNet50 architecture and transfer learning
- [Section 23 - Training and Evaluation (Malware Image Classification)](4-Malware-Classification/Section-23-Training-and-Evaluation-Malware-Image-Classification.md) — training loop, PyTorch
- [Section 24 - Model Evaluation (Malware Image Classification)](4-Malware-Classification/Section-24-Model-Evaluation-Malware-Image-Classification.md) — metrics, misclassification analysis

---

## 5 — Skills assessment

- [Section 25 - Skills Assessment](5-Skills-Assessment/Section-25-Skills-Assessment.md) — practical assessment

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep]]

**References**
- see:: [[HTB-Fundamentals-of-AI]] — theoretical foundations this module builds on

**Terms**
- spam classification, network anomaly detection, malware classification, Scikit-learn, PyTorch, Naive Bayes, random forest, ResNet50, JupyterLab
