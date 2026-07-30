---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 7 - Data Transformation"]
lead: Applying one-hot encoding and train/test splitting to prepare data for model training.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 7 - Data Transformation

Data transformations improve how features are represented and distributed, making them more useful for machine learning models. Converting categorical variables into numeric form and addressing skewed distributions both improve model stability, interpretability, and predictive performance.

## Encoding Categorical Features

Encoding converts categorical values into numeric form so machine learning algorithms can process them. The right method depends on the data:

- `OneHotEncoder`: Creates binary indicator columns, one per category. Avoids implying any order between categories.
- `LabelEncoder`: Assigns a unique integer to each category. Can unintentionally imply an ordering.
- `HashingEncoder` or frequency-based methods: Handle high-cardinality features while controlling feature space size.

After encoding, verify that the transformed features are meaningful and that no artificial ordering has been introduced.

### One-Hot Encoding

One-hot encoding replaces a categorical column with a set of binary columns, one for each possible category value. Each row receives a `1` in the column corresponding to its category and `0` in all others.

For example, a `color` feature with values `red`, `green`, and `blue` becomes three binary columns: `color_red`, `color_green`, and `color_blue`. A row with `color = red` gets `color_red = 1`, `color_green = 0`, `color_blue = 0`.

![[data_encoding.png]]

This prevents models from interpreting category labels as numeric magnitudes. The trade-off is that the feature count grows with the number of unique values.

The `protocol` column is encoded using `OneHotEncoder`:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df[['protocol']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['protocol']))
df = pd.concat([df.drop('protocol', axis=1), encoded_df], axis=1)
```

The original `protocol` column is replaced with distinct binary columns, and the model treats each protocol independently.

## Handling Skewed Data

A skewed feature has values clustered at one end of the distribution, with a few extreme outliers stretching the other tail. This can degrade model performance, particularly for algorithms sensitive to scale or that assume roughly uniform distributions.

Applying a log transform compresses large values more aggressively than small ones, producing a more balanced distribution. The `log1p` function (which computes $\log(1 + x)$) handles values at or near zero without producing undefined results:

```python
import numpy as np

# Apply logarithmic transformation to reduce skewness
df["bytes_transferred"] = np.log1p(df["bytes_transferred"])  # Add 1 to avoid log(0)
```

Before the transform, a few very large `bytes_transferred` values dominate the distribution and overshadow the majority of observations. After the transform, the distribution is more even, reducing the risk of the model overfitting to extreme values.

![[log_histogram.png]]

The histogram comparison confirms that the original skew is substantially reduced. No data is lost; the transform changes the scale, not the information content.

## Data Splitting

Data splitting divides a dataset into training, validation, and test subsets to support reliable model evaluation. Keeping these subsets separate prevents the model from being evaluated on data it was trained on:

- `Training Set`: Used to fit the model. Typically 60–80% of the full dataset.
- `Validation Set`: Used to tune hyperparameters and select between models. Typically 10–20%.
- `Test Set`: Used only after all training and tuning is complete. Typically 10–20%.

The code below uses `train_test_split` from `scikit-learn`. The first split allocates 80% for training and 20% for testing. The second split divides the 80% training portion into 60% for final training and 20% for validation.

Note: `test_size=0.25` in the second split refers to 25% of the 80% training subset, that is, $0.8 \times 0.25 = 0.2$, or 20% of the full dataset.

```python
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop("threat_level", axis=1)
y = df["threat_level"]

# Initial split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# Second split: from the 80% training portion, allocate 60% for final training and 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1337)
```

The resulting workflow:

- Train on `X_train` and `y_train`.
- Tune hyperparameters or compare models using `X_val` and `y_val`.
- Evaluate final performance on the untouched `X_test` and `y_test`.

---

## Summary

- One-hot encoding converts categorical columns into binary indicator columns, preventing models from misinterpreting categories as ordinal values.
- `OneHotEncoder` with `handle_unknown='ignore'` safely handles unseen categories at inference time without raising errors.
- Log transform (`np.log1p`) compresses right-skewed distributions, reducing the influence of extreme outliers on model training.
- Data is split into training (60%), validation (20%), and test (20%) sets using two sequential `train_test_split` calls.
- The validation set is used for hyperparameter tuning; the test set is reserved for final, unbiased performance evaluation.
- `test_size=0.25` in the second split refers to 25% of the 80% training portion, producing 20% of the full dataset.

---

## Best Practices

- Use `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to prevent errors when inference data contains unseen categories.
- Apply log transforms with `np.log1p` rather than `np.log` to safely handle zero values without producing `-inf`.
- Always use a fixed `random_state` in `train_test_split` so splits are reproducible across notebook restarts.
- Never use the test set during model development or hyperparameter tuning — reserve it strictly for final evaluation.
- After one-hot encoding, drop the original categorical column (`df.drop('protocol', axis=1)`) before concatenating the encoded columns.
- Plot histograms before and after log transforms to confirm that skewness is reduced as expected.

---

## Quiz

**Q1:** Why does one-hot encoding prevent models from learning incorrect relationships between categories?
> One-hot encoding creates independent binary columns for each category, so there is no implied numeric ordering between them; integer encoding (e.g., 0, 1, 2) would make the model treat the difference between categories as meaningful magnitudes.

**Q2:** What does `np.log1p` compute and why is it preferred over `np.log` for skewed features?
> `np.log1p` computes $\log(1 + x)$, which handles values at or near zero without producing undefined (`-inf`) results, making it safe for features that include zero values.

**Q3:** In a 60/20/20 train/val/test split using two sequential `train_test_split` calls, what `test_size` value is used in the second call?
> `test_size=0.25` — because 25% of the 80% training portion equals 20% of the full dataset.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-6-Data-Preprocessing]] — produces the cleaned data that transformation is applied to
- see:: [[Section-8-Metrics-for-Evaluating-a-Model]] — evaluates models trained on the split data produced here
- see:: [[Section-16-Preprocessing-and-Splitting-the-Dataset]] — applies identical splitting strategy in the network anomaly use case

**Terms**
- one-hot encoding, OneHotEncoder, LabelEncoder, HashingEncoder, log transform, log1p, skewed data, train/test split, validation set, training set, test set, binary indicator features, data splitting
