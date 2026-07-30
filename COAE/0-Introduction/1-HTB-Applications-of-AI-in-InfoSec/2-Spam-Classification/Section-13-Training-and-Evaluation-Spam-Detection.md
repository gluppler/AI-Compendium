---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 13 - Training and Evaluation Spam Detection"]
lead: Training a Naive Bayes spam classifier and evaluating its performance on held-out data.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 13 - Training and Evaluation (Spam Detection)

## Training

`MultinomialNB` is well suited for text classification: it handles large sparse feature matrices efficiently and its probabilistic model maps naturally onto the Naive Bayes framework introduced in [[Section-9-Spam-Classification]]. Wrapping vectorization and classification in a `Pipeline` ensures the same `CountVectorizer` transformation is applied consistently to both training and inference data:

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Build the pipeline by combining vectorization and classification
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])
```

`GridSearchCV` then searches over the `alpha` smoothing parameter of `MultinomialNB`. Laplace smoothing via `alpha` prevents zero probabilities for words absent from the training data; tuning it controls the bias-variance trade-off. Five-fold cross-validation with F1-score as the scoring metric guides the search:

```python
# Define the parameter grid for hyperparameter tuning
param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
}

# Perform the grid search with 5-fold cross-validation and the F1-score as metric
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

# Fit the grid search on the full dataset
grid_search.fit(df["message"], y)

# Extract the best model identified by the grid search
best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_)
```

## Evaluation

![[spam_eval.png]]

Assessing the trained model on unseen messages verifies that it generalizes beyond the training corpus. New messages must pass through the same preprocessing steps used during training before being fed to the classifier.

### Setting Up the Evaluation Messages

```python
# Example SMS messages for evaluation
new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]
```

### Preprocessing New Messages

The `preprocess_message` function mirrors the training pipeline: lowercase, strip non-alphabetic characters (keeping `$` and `!`), tokenize, remove stop words, and stem:

```python
import numpy as np
import re

# Preprocess function that mirrors the training-time preprocessing
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)
```

Apply it to each new message:

```python
# Preprocess and vectorize messages
processed_messages = [preprocess_message(msg) for msg in new_messages]
```

### Vectorizing the Processed Messages

Use the `CountVectorizer` stored inside the pipeline to transform the preprocessed strings into the same feature space the classifier was trained on:

```python
# Transform preprocessed messages into feature vectors
X_new = best_model.named_steps["vectorizer"].transform(processed_messages)
```

### Making Predictions

The `MultinomialNB` classifier returns both predicted labels and class probabilities:

```python
# Predict with the trained classifier
predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)
```

### Displaying Predictions and Probabilities

For each message, print the original text, the predicted label, the spam probability, and the ham probability:

```python
# Display predictions and probabilities for each evaluated message
for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
    spam_probability = prediction_probabilities[i][1]  # Probability of being spam
    ham_probability = prediction_probabilities[i][0]   # Probability of being not spam

    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Not-Spam Probability: {ham_probability:.2f}")
    print("-" * 50)
```

Representative output:

```bash
Message: Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.
Prediction: Spam
Spam Probability: 1.00
Not-Spam Probability: 0.00
--------------------------------------------------
Message: Hey, are we still meeting up for lunch today?
Prediction: Not-Spam
Spam Probability: 0.00
Not-Spam Probability: 1.00
--------------------------------------------------
Message: Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify
Prediction: Spam
Spam Probability: 0.94
Not-Spam Probability: 0.06
--------------------------------------------------
Message: Reminder: Your appointment is scheduled for tomorrow at 10am.
Prediction: Not-Spam
Spam Probability: 0.00
Not-Spam Probability: 1.00
--------------------------------------------------
Message: FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!
Prediction: Spam
Spam Probability: 1.00
Not-Spam Probability: 0.00
--------------------------------------------------
```

The model correctly separates the benign messages from a range of spam types, and the probability scores reflect its confidence in each prediction.

### Using joblib for Saving Models

`joblib` serializes Python objects (including scikit-learn pipelines) to a binary file format optimized for objects containing large NumPy arrays. Saving the trained model avoids re-training on every restart, which is critical in production:

```python
import joblib

# Save the trained model to a file for future use
model_filename = 'spam_detection_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved to {model_filename}")
```

The file captures the full pipeline state: vocabulary, vectorizer settings, and all learned classifier parameters. To reload and predict, run the same preprocessing before calling `predict`:

```python
# Load the saved model
loaded_model = joblib.load(model_filename)

# Preprocess new messages before prediction
new_data_processed = [preprocess_message(msg) for msg in new_messages]

# Make predictions on the preprocessed data
predictions = loaded_model.predict(new_data_processed)
```

---

## Summary

- `MultinomialNB` is well suited for sparse text feature matrices and maps naturally to the Naive Bayes probabilistic framework.
- Wrapping vectorization and classification in a `Pipeline` ensures consistent transformation is applied at both training and inference time.
- `GridSearchCV` with 5-fold cross-validation searches over `alpha` smoothing values to find the optimal bias-variance trade-off.
- F1-score is used as the `GridSearchCV` scoring metric to balance precision and recall during hyperparameter selection.
- Inference requires preprocessing new messages with the exact same function used during training before passing them to the classifier.
- `joblib.dump` serializes the full pipeline (vocabulary, vectorizer settings, and classifier parameters) for persistence and production deployment.

---

## Best Practices

- Use `Pipeline` to bundle vectorizer and classifier together so `transform` is never accidentally omitted at inference time.
- Tune `MultinomialNB`'s `alpha` with `GridSearchCV` — the default value of 1.0 is often not optimal for a given corpus.
- Score `GridSearchCV` with `scoring="f1"` rather than accuracy when dealing with class-imbalanced datasets like spam collections.
- Apply `preprocess_message` to every new message before calling `predict` — skipping preprocessing at inference is the most common deployment mistake.
- Use `best_model.named_steps["vectorizer"].transform(...)` to apply the fitted vectorizer from inside the pipeline without re-fitting.
- Save models with `joblib.dump` rather than `pickle` for better performance on objects containing large NumPy arrays.

---

## Quiz

**Q1:** What role does `alpha` smoothing play in `MultinomialNB`?
> `alpha` (Laplace smoothing) prevents zero probabilities for words that appear in the test data but were absent from the training corpus; without it, any unseen word would make the entire posterior probability zero.

**Q2:** Why is the preprocessing function applied to new messages before passing them to the trained classifier?
> The classifier was trained on vectorized representations of preprocessed text; applying the same preprocessing ensures new messages are transformed into the same feature space as the training data.

**Q3:** What does `joblib.dump` save when called on a trained `Pipeline` object?
> It saves the complete pipeline state: the fitted `CountVectorizer` vocabulary, all vectorizer settings, and all learned `MultinomialNB` parameters — everything needed to make predictions without retraining.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-14-Model-Evaluation-Spam-Detection]] — final model evaluation via the HTB portal follows directly from training here
- see:: [[Section-12-Feature-Extraction]] — the CountVectorizer pipeline step used here was built in Section 12
- see:: [[Section-8-Metrics-for-Evaluating-a-Model]] — F1-score used as GridSearchCV scoring metric is defined in Section 8

**Terms**
- MultinomialNB, Pipeline, GridSearchCV, alpha smoothing, cross-validation, F1-score, hyperparameter tuning, joblib serialization, predict_proba, spam probability
