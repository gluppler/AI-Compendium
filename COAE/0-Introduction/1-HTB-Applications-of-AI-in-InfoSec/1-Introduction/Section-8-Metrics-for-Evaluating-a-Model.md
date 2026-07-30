---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 8 - Metrics for Evaluating a Model"]
lead: Common evaluation metrics for classification models — accuracy, precision, recall, F1, and ROC.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 8 - Metrics for Evaluating a Model

Evaluating a trained classifier means examining numerical metrics that quantify the relationship between predictions and known ground-truth labels. The [Fundamentals of AI](https://academy.hackthebox.com/module/details/290) module introduced `accuracy`, `precision`, `recall`, and `F1-score`. Each captures a different aspect of model behavior, and they are most useful when read together.

## Accuracy

`Accuracy` is the proportion of correct predictions out of all predictions made. A model with `accuracy: 0.9950` classifies instances correctly 99.50% of the time.

Key points:

- Measures overall correctness.
- Computed as `(true positives + true negatives) / (all instances)`.
- Can be misleading when class distributions are imbalanced.

Consider a spam classifier where only 1% of emails are spam and 99% are legitimate. A model that always predicts "not spam" achieves `accuracy: 0.99` but catches no spam at all. Accuracy alone hides this failure, which is why complementary metrics are necessary for imbalanced datasets.

## Precision

`Precision` measures how often the model's positive predictions are correct. A `precision: 0.9949` means that when the model labels an instance as positive, it is right 99.49% of the time.

Key points:

- Reflects quality of positive predictions.
- Computed as `true positives / (true positives + false positives)`.
- High precision reduces wasted effort caused by false alarms.

In the spam context: if the model flags 100 emails as spam and 99 actually are, precision is high. Legitimate emails rarely land in the spam folder. However, if the model is overly conservative and only flags emails it is very confident about, it may miss many spam messages. High precision alone does not guarantee good coverage.

## Recall

`Recall` measures how well the model identifies all positive instances. A `recall: 0.9950` means the model detects 99.50% of all actual positives.

Key points:

- Reflects completeness of positive detection.
- Computed as `true positives / (true positives + false negatives)`.
- High recall reduces the risk of missing critical cases.

A spam classifier with high recall catches most spam. But if precision is low, it also misclassifies many legitimate emails as spam. The inbox is protected but the spam folder becomes cluttered with false positives.

## F1-Score

`F1-score` is the harmonic mean of `precision` and `recall`. A `F1-score: 0.9949` indicates near-perfect balance between the two.

Key points:

- Balances `precision` and `recall`.
- Computed as `2 * (precision * recall) / (precision + recall)`.
- Useful when class imbalance is present.

The `F1-score` penalizes models that sacrifice one metric to boost the other. A model that catches nearly all spam without flooding legitimate mail with false positives will score well on both, producing a high `F1-score`. This makes it a better single-number summary than accuracy or either component metric alone.

## Additional Considerations

Beyond these four core metrics, others provide further insight:

- `Specificity`: How effectively the model identifies true negatives.
- `AUC`: Area Under the ROC Curve, measures discriminative capability across decision thresholds.
- `Matthews Correlation Coefficient`: A single balanced metric well-suited to highly imbalanced datasets.
- `Confusion Matrix`: A table summarizing predictions versus true labels across all classes, giving a complete view of where errors occur.

These metrics and visualizations confirm whether high performance reflects genuine model quality or merely favorable dataset conditions.

## Contextualizing the Metrics

When reviewing metrics such as `accuracy: 0.9750`, `precision: 0.9300`, `recall: 0.9100`, `F1-score: 0.9200`, consider:

- Are the metrics consistent across different data segments or only on the aggregate?
- Does the dataset reflect real-world class distributions, including any imbalances?
- Are the costs of false positives and false negatives appropriately weighted for the application?

High accuracy can be trivially achieved when one class dominates. Verifying that both `precision` and `recall` remain strong confirms that the model performs well on both classes, not just the majority.

Different settings impose different trade-offs:

- In threat detection, `recall` is often prioritized because missing a real threat is more costly than investigating a false alarm.
- In resource-constrained environments, `precision` reduces the burden of following up on false positives.

Reading `precision` and `recall` together, summarized by the `F1-score`, gives a balanced view of whether a model's decisions are both reliable and complete.

---

## Summary

- Accuracy measures overall correctness but is misleading when class distributions are imbalanced.
- Precision measures the quality of positive predictions (how many flagged positives are truly positive).
- Recall measures completeness of positive detection (how many actual positives are caught).
- F1-score is the harmonic mean of precision and recall, penalizing models that sacrifice one to boost the other.
- Additional metrics include specificity, AUC-ROC, Matthews Correlation Coefficient, and the confusion matrix.
- In threat detection, recall is typically prioritized; in resource-constrained environments, precision reduces false-positive follow-up burden.

---

## Best Practices

- Always report precision and recall together alongside accuracy — accuracy alone can hide complete failures on minority classes.
- Use the F1-score as a single summary metric when class imbalance is present, since it balances precision and recall.
- Plot the confusion matrix to see per-class errors — a high overall score can mask systematic misclassification of specific classes.
- Consider the cost asymmetry of your domain: in security, a false negative (missed threat) is typically costlier than a false positive.
- Use AUC-ROC when you need to evaluate model performance across different decision thresholds, not just a fixed one.
- Verify that strong aggregate metrics hold across data segments and not just on the majority class.

---

## Quiz

**Q1:** Why can a spam classifier achieve 99% accuracy while catching zero spam messages?
> If 99% of emails are legitimate and the model always predicts "not spam," it is correct 99% of the time despite never detecting any spam — accuracy rewards predicting the majority class.

**Q2:** What is the F1-score and when is it most useful?
> F1-score is the harmonic mean of precision and recall; it is most useful when class imbalance is present and a single balanced metric is needed that penalizes models sacrificing one component for the other.

**Q3:** In security threat detection, why is recall often prioritized over precision?
> Missing a real threat (false negative) is typically more costly than investigating a false alarm (false positive), so maximizing recall — catching all actual threats — takes precedence.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-13-Training-and-Evaluation-Spam-Detection]] — first module section that applies these metrics to a real model
- see:: [[Section-14-Model-Evaluation-Spam-Detection]] — dedicated evaluation section using precision, recall, and F1 on spam data
- see:: [[Section-18-Model-Evaluation-Network-Anomaly-Detection]] — uses the same metric framework for the anomaly detection model

**Terms**
- accuracy, precision, recall, F1-score, true positives, false positives, false negatives, true negatives, class imbalance, confusion matrix, AUC, ROC curve, Matthews Correlation Coefficient, specificity
