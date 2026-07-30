---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 5 - Datasets"]
lead: Understanding dataset structure, attributes, and how to load and inspect data for AI tasks.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 5 - Datasets

Dataset quality directly determines how well a trained model will perform. Data preprocessing (transforming raw data into a form suitable for machine learning) is a core step in any AI pipeline.

## Understanding Datasets

Datasets are structured collections of data points used for analysis and model training. Common forms include:

- `Tabular Data`: Rows and columns, as in spreadsheets or databases.
- `Image Data`: Sets of images represented numerically as pixel arrays.
- `Text Data`: Unstructured sequences of sentences, paragraphs, or documents.
- `Time Series Data`: Sequential observations collected over time, capturing temporal patterns.

Dataset quality determines downstream outcomes:

- `Model Accuracy`: High-quality data produces more accurate models. Noisy, incomplete, or biased data reduces performance.
- `Generalization`: Well-curated data enables models to generalize to unseen examples, minimizing overfitting.
- `Efficiency`: Clean data reduces training time and computational cost.
- `Reliability`: Trustworthy data leads to dependable results, which matters especially in domains like healthcare or security.

### What Makes a Dataset 'Good'

|Attribute|Description|Example|
|---|---|---|
|`Relevance`|Data should match the problem. Irrelevant data introduces noise.|Social media text is more relevant than stock prices for sentiment analysis.|
|`Completeness`|Minimal missing values. Missing data biases models.|Imputation can compensate, but starting with complete data is preferable.|
|`Consistency`|Uniform format and structure. Inconsistencies cause preprocessing errors.|Date formats should be uniform (e.g., `YYYY-MM-DD`) across the entire dataset.|
|`Quality`|Accurate, error-free data. Errors arise from collection, entry, or transmission.|Data validation processes help maintain accuracy.|
|`Representativeness`|Data should represent the target population. Biased data produces biased models.|A facial recognition dataset needs diverse representation across ethnicities, ages, and genders.|
|`Balance`|Balanced class distributions, especially for classification. Imbalanced data biases models toward majority classes.|Oversampling, undersampling, or synthetic data generation can rebalance the dataset.|
|`Size`|Large enough to capture the problem's complexity. Small datasets limit learning; very large datasets increase compute requirements.|Neither extreme is universally ideal; size should match problem complexity.|

## The Dataset

The provided dataset, [demo_dataset.csv](https://academy.hackthebox.com/storage/modules/292/demo_dataset.zip), is a CSV file containing network log entries. Each record describes a network event with fields for source IP, destination port, protocol, data volume, and threat level. These entries simulate various network scenarios useful for building and evaluating intrusion detection systems.

### Dataset Structure

- `log_id`: Unique identifier for each log entry.
- `source_ip`: Source IP address of the network event.
- `destination_port`: Destination port number used in the event.
- `protocol`: Network protocol (e.g., `TCP`, `TLS`, `SSH`).
- `bytes_transferred`: Total bytes transferred during the event.
- `threat_level`: Severity indicator. `0` is normal traffic, `1` is low-threat activity, `2` is a high-threat event.

### Challenges and Considerations

Several data quality issues require attention before processing:

- The dataset mixes numerical and categorical data.
- Some columns contain missing values and invalid entries that require cleaning.
- Certain numeric columns contain non-numeric strings that must be converted or removed.
- The `threat_level` column includes unknown values (e.g., `?`, `-1`) that must be standardized.

Identifying these issues early allows them to be handled systematically before training.

## Loading the Dataset

Load the dataset into a `pandas` DataFrame to begin working with it. A DataFrame is a two-dimensional labeled data structure that supports efficient inspection, filtering, encoding, and transformation. Its labeled axes and heterogeneous data handling make it well-suited for preprocessing pipelines.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv("./demo_dataset.csv")
```

`pd.read_csv()` loads the CSV file into a DataFrame named `data`, ready for inspection and further processing.

## Exploring the Dataset

After loading, examine the data to understand its structure and identify what cleaning or transformation is needed.

### Viewing Sample Entries

Inspect the first few rows for obvious issues like unexpected column names, wrong data types, or irregular patterns:

```python
# Display the first few rows of the dataset
print(data.head())
```

### Inspecting Data Structure and Types

Review column data types and non-null counts to detect columns with missing or unexpected data:

```python
# Get a summary of column data types and non-null counts
print(data.info())
```

`info()` reports the dataset's shape, column names, data types, and entry counts per column.

### Checking for Missing Values

Identify how many missing values each column contains to prioritize cleaning:

```python
# Identify columns with missing values
print(data.isnull().sum())
```

This returns a count of null values per column. Columns with high null counts may require imputation, row removal, or other targeted strategies before the data is fit for training.

---

## Summary

- Dataset quality directly determines model performance; noise, missing values, or bias all degrade accuracy and generalization.
- Common dataset types include tabular, image, text, and time series data, each requiring different preprocessing strategies.
- Good datasets are relevant, complete, consistent, accurate, representative, balanced, and appropriately sized for the problem.
- Class imbalance biases models toward majority classes; rebalancing via oversampling, undersampling, or synthetic generation is needed.
- The demo dataset is a CSV of network log entries with fields for IP, port, protocol, bytes transferred, and threat level.
- Data quality issues in the demo dataset include mixed types, missing values, invalid entries, and non-standard threat level values.

---

## Best Practices

- Run `data.head()`, `data.info()`, and `data.isnull().sum()` immediately after loading any dataset to detect structural issues early.
- Identify class imbalance before training — a skewed class distribution will inflate accuracy metrics and mislead evaluation.
- Document all known data quality issues before preprocessing so every issue is addressed systematically, not ad hoc.
- Use `pd.read_csv()` with explicit `dtype` arguments where known to prevent pandas from silently misinterpreting columns.
- Prefer fixing `threat_level` values like `?` or `-1` explicitly rather than dropping rows when the dataset is small.

---

## Quiz

**Q1:** Why can accuracy alone be a misleading metric when a dataset has severe class imbalance?
> A model that always predicts the majority class achieves high accuracy without actually learning to detect minority class examples.

**Q2:** What are the seven attributes that characterize a good dataset?
> Relevance, completeness, consistency, quality (accuracy), representativeness, balance, and appropriate size.

**Q3:** What does `data.isnull().sum()` tell you about a pandas DataFrame?
> It returns the count of null (missing) values per column, identifying which columns require imputation or cleaning.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-6-Data-Preprocessing]] — directly continues from the dataset loaded and inspected here
- see:: [[Section-7-Data-Transformation]] — applies transformations to the dataset explored in this section
- see:: [[Section-10-The-Spam-Dataset]] — another domain-specific dataset examined in the same module

**Terms**
- dataset, tabular data, image data, text data, time series data, pandas DataFrame, CSV, missing values, data quality, representativeness, class imbalance, imputation, threat_level
