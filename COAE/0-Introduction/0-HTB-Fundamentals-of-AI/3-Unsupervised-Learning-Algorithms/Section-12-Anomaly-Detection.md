---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 12 - Anomaly Detection"]
lead: Anomaly detection identifies outliers deviating from normal data patterns — critical for fraud, intrusion, and failure detection.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 12."
---

![[anomaly_detection.png]]

`Anomaly detection` identifies data points that deviate significantly from normal behavior. These anomalies — also called outliers — can indicate fraud, system failures, network intrusions, or medical emergencies. Because anomalies are rare and their characteristics are often unknown in advance, this is primarily an unsupervised task: the model learns what "normal" looks like and flags deviations.

Anomalies fall into three categories:

- `Point Anomalies`: A single data point is abnormal relative to the rest — e.g., a credit card transaction for an unusually large amount.
- `Contextual Anomalies`: A data point is normal in one context but anomalous in another — e.g., a 30°C temperature reading is expected in summer but suspicious in winter.
- `Collective Anomalies`: A group of data points collectively deviates from the norm, even though each point in isolation might look normal — e.g., a coordinated burst of login attempts from many unknown IP addresses.

Three classes of techniques address anomaly detection:

- `Statistical Methods`: Assume normal data follows a known distribution (typically Gaussian) and flag points that fall in the tail. Examples include z-score thresholding and boxplot-based IQR methods.
- Clustering-Based Methods: Group normal data into clusters; isolated points or small sparse clusters are anomalies. K-means and density-based clustering (DBSCAN) are commonly used.
- Machine Learning-Based Methods: Learn a model of normal behavior and score deviations. The three main algorithms are described below.

### One-Class SVM

![[one_class_svm.png]]

`One-Class SVM` learns a tight decision boundary enclosing the normal training data. Any new point falling outside this boundary is flagged as an outlier. Unlike standard SVMs, there is only one class of training data — the algorithm learns to detect what is not normal rather than to distinguish between two classes. Kernel functions extend it to non-linear boundaries.

### Isolation Forest

![[isolation_forest.png]]

`Isolation Forest` exploits the property that anomalies are few and different — they are easier to isolate than normal points. The algorithm builds an ensemble of random binary trees (`isolation trees`) by repeatedly selecting a random feature and a random split value. Points that get isolated in fewer splits (shorter paths) are more likely to be anomalies, because their unusual values make them stand out quickly.

The anomaly score for a data point `x` is:

$$\text{score}(x) = 2^{-\frac{E(h(x))}{c(n)}}$$

Where:

- `E(h(x))`: Mean path length of `x` across all isolation trees.
- `c(n)`: Expected path length for a dataset of `n` points in a binary search tree (normalization factor).

Scores near **1** indicate anomalies; scores near **0.5** indicate normal points.

### Local Outlier Factor (LOF)

![[local_outlier_factor.png]]

Local Outlier Factor (LOF) detects anomalies by comparing a point's local density to its neighbors' densities. A point surrounded by dense neighbors but itself in a sparse region is flagged as an outlier — it is "more isolated" relative to its neighborhood.

The LOF score for a point `p` is:

$$\text{LOF}(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}(o)}{\text{lrd}(p)}}{k}$$

Where:

- `lrd(p)`: Local reachability density of `p` — inverse of the average reachability distance to its `k` nearest neighbors.
- `lrd(o)`: Local reachability density of neighbor `o`.
- `k`: Number of nearest neighbors.

Higher LOF scores indicate lower local density relative to neighbors, signaling a likely outlier.

### Local Reachability Density

The local reachability density is:

$$\text{lrd}(p) = \frac{k}{\sum_{o \in N_k(p)} \text{reach\_dist}(p, o)}$$

Where `reach_dist(p, o)` is the reachability distance from `p` to `o`, defined as:

$$\text{reach\_dist}(p, o) = \max(d(p, o),\ k\text{-dist}(o))$$

The k-dist of `o` is the distance from `o` to its `k`-th nearest neighbor. This floor prevents instability in very dense regions where raw distances approach zero.

### Data Assumptions

- Normal Data Distribution: Statistical methods require specifying the distribution of normal data. Gaussian assumptions fail when normal data is multi-modal or heavily skewed.
- `Feature Relevance`: Poor feature selection degrades detection quality. Domain knowledge helps identify which signals are meaningful.
- Labeled Data (for some methods): Supervised variants (e.g., anomaly classifiers) need labeled examples of both normal and anomalous data, which is often unavailable.

---

## Summary

- Anomaly detection identifies data points that deviate significantly from normal behavior — used for fraud, network intrusion, hardware failures, and medical anomaly detection.
- Anomalies fall into three categories: point (a single unusual point), contextual (normal in another context), and collective (a group of points deviating together).
- Statistical methods assume a known distribution and flag tail points; clustering methods flag isolated or sparse-cluster points; ML-based methods learn a model of normal.
- One-Class SVM learns a tight boundary around normal training data and flags points outside it — extended to non-linear boundaries via kernel functions.
- Isolation Forest isolates anomalies in fewer random tree splits because their unusual values make them stand out quickly; shorter path length → higher anomaly score.
- Local Outlier Factor (LOF) compares a point's local density to its neighbors' densities — points in locally sparse regions receive high LOF scores.

---

## Best Practices

- Validate that the distribution assumed by statistical methods matches the actual data; Gaussian assumptions fail for multimodal or skewed normals.
- Use Isolation Forest as a scalable default for large datasets — it handles high dimensionality well and requires no distribution assumption.
- Apply LOF when anomalies are expected to be locally dense pockets in otherwise sparse regions rather than globally extreme outliers.
- Use domain knowledge to distinguish true anomalies from valid but rare events (e.g., large legitimate transactions) before labeling them for model evaluation.
- When labeled anomaly examples are available, use supervised or semi-supervised variants rather than unsupervised methods — labeled data dramatically improves precision.
- Tune the contamination parameter carefully — it sets the expected fraction of anomalies and directly controls the threshold for anomaly/normal classification.

---

## Quiz

**Q1:** What are the three categories of anomalies? Give a real-world example of each.
> Point anomaly: a single unusually large credit card transaction. Contextual anomaly: a 30°C temperature reading in winter (normal in summer). Collective anomaly: a coordinated burst of login attempts from many IPs that are each individually plausible.

**Q2:** How does Isolation Forest score anomalies, and why does it work?
> It builds random trees by repeatedly splitting on random feature-threshold pairs. Anomalies have unusual values that place them in low-density regions, so they get isolated in fewer splits (shorter path lengths). The anomaly score is `2^(-E(h(x))/c(n))` — scores near 1 indicate anomalies.

**Q3:** What does LOF measure and what does a high LOF score indicate?
> LOF compares a point's local density to its k nearest neighbors' densities. A high LOF score indicates the point sits in a much sparser region than its neighbors — it is more isolated relative to its local neighborhood, making it a likely outlier.

**Q4:** What is the key limitation of statistical anomaly detection methods?
> They require specifying the distribution of normal data. If normal data is multi-modal, skewed, or follows a complex distribution, Gaussian or other parametric assumptions produce inaccurate probability estimates and unreliable anomaly thresholds.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-10-K-Means-Clustering]] — density-based clusters can highlight anomalous points
- see:: [[Section-8-Support-Vector-Machines]] — one-class SVM is a key anomaly detection method

**Terms**
- isolation forest, one-class SVM, local outlier factor, outlier, inlier, contamination, anomaly score
