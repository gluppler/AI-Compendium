---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 9 - Unsupervised Learning Algorithms"]
lead: Unsupervised learning finds hidden structure in unlabeled data through clustering, dimensionality reduction, and anomaly detection.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 9."
---

Unsupervised learning operates on data that has no labels or predefined outcomes. The algorithm's job is to find structure — clusters, compressed representations, or anomalies — from the raw input alone. This contrasts with `supervised learning`, where labeled examples guide the learning process.

Unlabeled data is abundant and cheap. Unsupervised methods become essential when labeling is expensive, impossible, or when the goal is exploratory rather than predictive.

## How Unsupervised Learning Works

Unsupervised algorithms detect patterns by quantifying similarities and differences between data points. Depending on the task, they group similar points together, compress the data into fewer dimensions, or flag points that break expected patterns.

Unsupervised learning problems fall into three broad categories:

1. `Clustering`: Groups data points by similarity — e.g., segmenting customers by purchasing behavior or grouping documents by topic.
2. Dimensionality Reduction: Compresses data to fewer features while retaining essential variance — e.g., reducing a 1000-pixel image representation to 50 components.
3. `Anomaly Detection`: Identifies data points that deviate from the norm — e.g., flagging unusual network traffic or fraudulent transactions.

## Core Concepts in Unsupervised Learning

### Unlabeled Data

Unsupervised learning works entirely from input features, with no target variable. The algorithm must discover structure purely from how the data points relate to each other.

### Similarity Measures

Most unsupervised algorithms depend on a measure of distance or similarity between data points:

- `Euclidean Distance`: Straight-line distance in multi-dimensional space. Sensitive to feature scale.
- `Cosine Similarity`: Angle between two vectors. Captures orientation rather than magnitude — useful for text data.
- `Manhattan Distance`: Sum of absolute coordinate differences. More robust to outliers than Euclidean distance in high dimensions.

The choice of measure affects which structure the algorithm finds.

### Clustering Tendency

Before clustering, it is worth verifying that the data actually contains natural groupings. If the data is uniformly distributed, any clustering result is an artifact of the algorithm rather than a real structure in the data. Tools like the Hopkins statistic assess clustering tendency.

### Cluster Validity

Two key metrics evaluate cluster quality:

- `Cohesion`: How similar points are within a cluster. High cohesion means tight, compact clusters.
- `Separation`: How different clusters are from each other. High separation means distinct, well-spaced clusters.

Composite scores like the silhouette score and the Davies-Bouldin index combine both into a single evaluation metric and help choose the right number of clusters.

### Dimensionality

Higher-dimensional data is harder to work with: distances become less meaningful, and computational cost grows. This is the "curse of dimensionality." Reducing dimensions before clustering or modeling often improves both performance and interpretability.

### Intrinsic Dimensionality

The intrinsic dimensionality is the minimum number of dimensions needed to capture the essential variation in the data, which is often much lower than the raw feature count. Dimensionality reduction methods aim to find this lower-dimensional representation.

### Anomaly

An `anomaly` is a data point that deviates significantly from expected behavior. Anomalies may indicate fraud, hardware failure, intrusion attempts, or measurement errors. Detecting them early has high practical value.

### Outlier

An `outlier` is a data point distant from the bulk of the data. The term is broader than anomaly — outliers can arise from noise, data entry errors, or genuinely unusual but valid observations.

### Feature Scaling

Distance-based algorithms are sensitive to feature scale. A feature measured in thousands will dominate over one measured in single digits. Standard preprocessing steps:

- `Min-Max Scaling`: Rescales each feature to `[0, 1]`.
- Standardization (Z-score): Centers each feature at zero with unit variance. Preferred when data is approximately normally distributed.

---

## Summary

- Unsupervised learning finds structure in data without labels, making it essential when labeling is expensive or the goal is exploratory.
- The three main problem types are clustering (group similar points), dimensionality reduction (compress features), and anomaly detection (flag deviations).
- Similarity measures — Euclidean distance, cosine similarity, Manhattan distance — drive cluster formation; the choice affects which structures are discovered.
- Cluster validity is assessed by cohesion (within-cluster tightness) and separation (between-cluster distinctness); composite metrics include silhouette score and Davies-Bouldin index.
- Distance-based algorithms are sensitive to feature scale; always standardize (min-max or z-score) before clustering or applying PCA.
- The curse of dimensionality makes high-dimensional data difficult to cluster — dimensionality reduction often precedes clustering in practice.

---

## Best Practices

- Standardize features before any distance-based unsupervised algorithm — unscaled features cause high-magnitude variables to dominate similarity calculations.
- Verify clustering tendency (e.g., Hopkins statistic) before applying a clustering algorithm; if data is uniformly distributed, clustering results are artifacts.
- Evaluate clusters with multiple metrics (silhouette score, Davies-Bouldin index) rather than relying on a single criterion.
- Apply dimensionality reduction (PCA, UMAP) before clustering high-dimensional data to reduce noise and computational cost.
- Choose the distance metric based on the data type: Euclidean for dense numerical data, cosine similarity for text/high-dimensional sparse data, Manhattan when robustness to outliers is needed.

---

## Quiz

**Q1:** What are the three main categories of unsupervised learning and give one application for each?
> Clustering (customer segmentation by purchasing behavior), dimensionality reduction (compressing image features for visualization), anomaly detection (flagging unusual network traffic patterns).

**Q2:** Why must features be scaled before running k-means or PCA?
> Both algorithms use distance or variance calculations. A feature with a range of 10,000 will dominate over one with a range of 1, making the result an artifact of scale rather than true structure. Standardization ensures all features contribute equally.

**Q3:** What is the difference between cohesion and separation in cluster evaluation?
> Cohesion measures how similar points are within a cluster (tighter is better); separation measures how distinct clusters are from each other (farther apart is better). High-quality clusters have high cohesion and high separation.

**Q4:** What is the curse of dimensionality and why does it affect distance-based algorithms?
> In high-dimensional spaces, all pairwise distances become nearly equal, making it hard to distinguish "near" from "far" neighbors. This degrades distance metrics and causes clustering algorithms to find spurious or meaningless groupings.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-10-K-Means-Clustering]] — clustering algorithm
- see:: [[Section-11-Principal-Component-Analysis]] — dimensionality reduction
- see:: [[Section-12-Anomaly-Detection]] — outlier detection

**Terms**
- clustering, dimensionality reduction, anomaly detection, similarity measures, Euclidean distance, cosine similarity, feature scaling
