---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 17 - Training and Evaluation Network Anomaly Detection"]
lead: Training a random forest classifier for network anomaly detection and evaluating results.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 17 - Training and Evaluation (Network Anomaly Detection)

## Training the Model

A `RandomForestClassifier` is initialized with `random_state=1337` for reproducibility and fitted on the training subset produced in [[Section-16-Preprocessing-and-Splitting-the-Dataset]]:

```python
# Train RandomForest model for multi-class classification
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)
```

`fit` builds the ensemble by constructing each decision tree on a bootstrapped sample of `multi_train_X`, evaluating random feature subsets at every split.

## Evaluating the Model on the Validation Set

Predict on the validation set and compute weighted accuracy, precision, recall, and F1-score. `average='weighted'` accounts for class imbalance by weighting each class's metric by its support:

```python
# Predict and evaluate the model on the validation set
multi_predictions = rf_model_multi.predict(multi_val_X)
accuracy = accuracy_score(multi_val_y, multi_predictions)
precision = precision_score(multi_val_y, multi_predictions, average='weighted')
recall = recall_score(multi_val_y, multi_predictions, average='weighted')
f1 = f1_score(multi_val_y, multi_predictions, average='weighted')

print(f"Validation Set Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix for Validation Set
conf_matrix = confusion_matrix(multi_val_y, multi_predictions)
class_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Network Anomaly Detection - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report for Validation Set
print("Classification Report for Validation Set:")
print(classification_report(multi_val_y, multi_predictions, target_names=class_labels))
```

The confusion matrix heatmap shows per-class true positives, false positives, and false negatives at a glance. The classification report provides per-class precision, recall, F1-score, and support, useful for identifying which attack categories the model struggles with.

## Testing the Model on the Test Set

![[anomaly_test.png]]

Final evaluation runs on the held-out test set, which the model has never seen during training or validation:

```python
# Final evaluation on the test set
test_multi_predictions = rf_model_multi.predict(test_X)
test_accuracy = accuracy_score(test_y, test_multi_predictions)
test_precision = precision_score(test_y, test_multi_predictions, average='weighted')
test_recall = recall_score(test_y, test_multi_predictions, average='weighted')
test_f1 = f1_score(test_y, test_multi_predictions, average='weighted')

print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

# Confusion Matrix for Test Set
test_conf_matrix = confusion_matrix(test_y, test_multi_predictions)
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Network Anomaly Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report for Test Set
print("Classification Report for Test Set:")
print(classification_report(test_y, test_multi_predictions, target_names=class_labels))
```

Comparing test metrics against validation metrics reveals whether the model generalizes cleanly or whether validation-time tuning introduced any optimism bias.

## Saving Model

Serialize the trained model with `joblib` for submission and future use:

```python
import joblib

# Save the trained model to a file
model_filename = 'network_anomaly_detection_model.joblib'
joblib.dump(rf_model_multi, model_filename)
print(f"Model saved to {model_filename}")
```

---

## Summary

- A `RandomForestClassifier` is initialized with `random_state=1337` and fitted on `multi_train_X` for reproducible multi-class anomaly detection.
- Validation evaluation uses `average='weighted'` for precision, recall, and F1-score to account for class imbalance across attack categories.
- A confusion matrix heatmap reveals per-class true positives, false positives, and false negatives visually at a glance.
- The classification report provides per-class precision, recall, F1-score, and support — identifying which attack types are harder to classify.
- Final test evaluation on the held-out set confirms whether the model generalizes cleanly or whether validation tuning introduced optimism bias.
- The trained model is serialized with `joblib.dump` to a `.joblib` file for submission and future use.

---

## Best Practices

- Always evaluate on the validation set before the test set — comparing validation and test metrics reveals whether any overfitting to the validation set occurred.
- Use `average='weighted'` for multi-class metrics to weight each class by its sample count, avoiding misleading averages for imbalanced classes.
- Print `classification_report` to get per-class breakdowns — a strong overall F1 can hide poor performance on rare attack categories.
- Visualize the confusion matrix with class labels (`Normal`, `DoS`, `Probe`, `Privilege`, `Access`) rather than raw integers for interpretability.
- Save the model immediately after training with `joblib.dump` before any further experimentation that could overwrite the trained state.

---

## Quiz

**Q1:** Why is `average='weighted'` used when computing precision and recall for the network anomaly model?
> Weighted averaging accounts for class imbalance by weighting each class's metric contribution by its number of samples (support), preventing rare classes from disproportionately skewing the aggregate metric.

**Q2:** What does the confusion matrix heatmap show for a multi-class classifier?
> It shows per-class counts of true positives (diagonal), false positives (column off-diagonal), and false negatives (row off-diagonal) so misclassification patterns between specific class pairs are visible.

**Q3:** What does comparing validation metrics against test metrics reveal about a trained model?
> It reveals whether the model generalizes cleanly to completely unseen data or whether the hyperparameter tuning performed on the validation set introduced optimism bias that inflates validation scores relative to true generalization.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-18-Model-Evaluation-Network-Anomaly-Detection]] — the model saved here is uploaded for final evaluation in Section 18
- see:: [[Section-16-Preprocessing-and-Splitting-the-Dataset]] — the multi_train_X and test_X splits used here are produced in Section 16
- see:: [[Section-8-Metrics-for-Evaluating-a-Model]] — accuracy, precision, recall, and F1 metrics computed here are defined in Section 8

**Terms**
- RandomForestClassifier, multi-class classification, confusion matrix, classification report, validation set, test set, weighted averaging, seaborn heatmap, joblib model saving, reproducibility
