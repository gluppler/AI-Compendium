---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 6 - Data Preprocessing"]
lead: Cleaning and refining data — handling missing values, encoding categoricals, and correcting skew.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 6 - Data Preprocessing

Data preprocessing transforms raw data into a format suitable for machine learning. Core techniques include:

- `Data Cleaning`: Handling missing values, removing duplicates, smoothing noisy data.
- `Data Transformation`: Normalizing, encoding, scaling, and reducing data.
- `Data Integration`: Merging and aggregating data from multiple sources.
- `Data Formatting`: Converting data types and reshaping data structures.

Effective preprocessing addresses inconsistencies, missing values, outliers, noise, and scaling issues, all of which impact model accuracy, efficiency, and robustness.

## Identifying Invalid Values

Beyond missing values, specific columns may contain invalid entries that must be detected before cleaning.

### Checking for Invalid IP Addresses

Use a regular expression to validate each value in the `source_ip` column:

```python
import re

def is_valid_ip(ip):
    pattern = re.compile(
        r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    )
    return bool(pattern.match(ip))

# Check for invalid IP addresses
invalid_ips = data[~data['source_ip'].astype(str).apply(is_valid_ip)]
print(invalid_ips)
```

### Checking for Invalid Port Numbers

Valid port numbers fall between 0 and 65535. Anything outside that range or non-numeric is invalid:

```python
def is_valid_port(port):
    try:
        port = int(port)
        return 0 <= port <= 65535
    except ValueError:
        return False

# Check for invalid port numbers
invalid_ports = data[~data['destination_port'].apply(is_valid_port)]
print(invalid_ports)
```

### Checking for Invalid Protocol Values

Validate `protocol` values against a known set of accepted protocols:

```python
valid_protocols = ['TCP', 'TLS', 'SSH', 'POP3', 'DNS', 'HTTPS', 'SMTP', 'FTP', 'UDP', 'HTTP']

# Check for invalid protocol values
invalid_protocols = data[~data['protocol'].isin(valid_protocols)]
print(invalid_protocols)
```

### Checking for Invalid Bytes Transferred

Byte counts must be numeric and non-negative:

```python
def is_valid_bytes(bytes):
    try:
        bytes = int(bytes)
        return bytes >= 0
    except ValueError:
        return False

# Check for invalid bytes transferred
invalid_bytes = data[~data['bytes_transferred'].apply(is_valid_bytes)]
print(invalid_bytes)
```

### Checking for Invalid Threat Levels

Threat levels must be integers in the range 0–2:

```python
def is_valid_threat_level(threat_level):
    try:
        threat_level = int(threat_level)
        return 0 <= threat_level <= 2
    except ValueError:
        return False

# Check for invalid threat levels
invalid_threat_levels = data[~data['threat_level'].apply(is_valid_threat_level)]
print(invalid_threat_levels)
```

## Handling Invalid Entries

Two main strategies exist for dealing with invalid data: dropping or imputing.

### Dropping Invalid Entries

The simplest approach is to discard invalid rows entirely, leaving only clean data:

```python
# errors='ignore' handles overlapping indexes across validity checks
data = data.drop(invalid_ips.index, errors='ignore')
data = data.drop(invalid_ports.index, errors='ignore')
data = data.drop(invalid_protocols.index, errors='ignore')
data = data.drop(invalid_bytes.index, errors='ignore')
data = data.drop(invalid_threat_levels.index, errors='ignore')

print(data.describe(include='all'))
```

This approach preserves data accuracy and is appropriate when losing some rows does not compromise the analysis. After dropping invalid rows from this dataset, 77 clean entries remain.

When the dataset is small or invalid entries are numerous, discarding data may not be viable. In those cases, imputation can recover value.

### Imputing Missing Values

Imputation replaces invalid or missing values with estimated substitutes, retaining as much data as possible.

First, normalize all invalid or corrupt strings into `NaN` so downstream imputers treat them uniformly:

```python
import pandas as pd
import numpy as np
import re
from ipaddress import ip_address

df = pd.read_csv('demo_dataset.csv')

invalid_ips = ['INVALID_IP', 'MISSING_IP']
invalid_ports = ['STRING_PORT', 'UNUSED_PORT']
invalid_bytes = ['NON_NUMERIC', 'NEGATIVE']
invalid_threat = ['?']

df.replace(invalid_ips + invalid_ports + invalid_bytes + invalid_threat, np.nan, inplace=True)

df['destination_port'] = pd.to_numeric(df['destination_port'], errors='coerce')
df['bytes_transferred'] = pd.to_numeric(df['bytes_transferred'], errors='coerce')
df['threat_level'] = pd.to_numeric(df['threat_level'], errors='coerce')

def is_valid_ip(ip):
    pattern = re.compile(
        r'^((25[0-5]|2[0-4][0-9]|[01]?\d?\d)\.){3}'
        r'(25[0-5]|2[0-4]\d|[01]?\d?\d)$'
    )
    if pd.isna(ip) or not pattern.match(str(ip)):
        return np.nan
    return ip

df['source_ip'] = df['source_ip'].apply(is_valid_ip)
```

After this step, `NaN` represents all missing or invalid data points uniformly.

For numeric columns, use the median or mean. For categorical columns, use the most frequent value:

```python
from sklearn.impute import SimpleImputer

numeric_cols = ['destination_port', 'bytes_transferred', 'threat_level']
categorical_cols = ['protocol']

num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
```

For more complex scenarios, `KNNImputer` accounts for relationships between features when producing estimates:

```python
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```

After imputation, apply domain constraints. For `source_ip` values still missing, assign `0.0.0.0` as a default. Validate `protocol` against known values and assign the mode where invalid. Clip port numbers to the valid range:

```python
valid_protocols = ['TCP', 'TLS', 'SSH', 'POP3', 'DNS', 'HTTPS', 'SMTP', 'FTP', 'UDP', 'HTTP']

df.loc[~df['protocol'].isin(valid_protocols), 'protocol'] = df['protocol'].mode()[0]
df['source_ip'] = df['source_ip'].fillna('0.0.0.0')
df['destination_port'] = df['destination_port'].clip(lower=0, upper=65535)
```

Run a final verification to confirm that distributions are reasonable and all categorical values are valid:

```python
print(df.describe(include='all'))
```

If anomalies persist, revisit the imputation strategy or remove the remaining problematic rows.

---

## Summary

- Data preprocessing encompasses cleaning, transformation, integration, and formatting to prepare raw data for model training.
- Invalid entries must be detected before removal or imputation: regex for IPs, range checks for ports and bytes, set membership for protocols.
- Dropping invalid rows is simple and preserves accuracy but may leave too few records when the dataset is small.
- Imputation replaces invalid or missing values with estimated substitutes (median, mode, or KNN-based) to retain more data.
- After imputation, domain constraints must still be enforced: valid IP fallbacks, protocol mode substitution, port clipping.
- A final `df.describe(include='all')` verification confirms that distributions are reasonable and no invalid values remain.

---

## Best Practices

- Normalize all invalid strings to `np.nan` first using `df.replace(...)` before applying Scikit-learn imputers, which expect `NaN`.
- Use `pd.to_numeric(..., errors='coerce')` to convert non-numeric strings in numeric columns to `NaN` without raising exceptions.
- Prefer `SimpleImputer(strategy='median')` for numeric columns with outliers; use `strategy='most_frequent'` for categoricals.
- Apply `KNNImputer` when feature correlations are strong enough to produce better estimates than statistical averages.
- After imputation, clip numeric columns to valid domain ranges (e.g., `df['destination_port'].clip(lower=0, upper=65535)`) to enforce constraints.
- Use `errors='ignore'` in `df.drop()` when multiple invalid-entry DataFrames share overlapping indices to avoid KeyErrors.

---

## Quiz

**Q1:** What is the key difference between dropping invalid entries and imputing them?
> Dropping removes the rows entirely, keeping only clean data but potentially losing too many records; imputation replaces invalid values with estimates, retaining more data at the cost of introducing some approximation.

**Q2:** Why should invalid strings be normalized to `np.nan` before using Scikit-learn imputers?
> Scikit-learn imputers recognize and handle `NaN` as missing values; non-standard strings like `?` or `INVALID_IP` are not recognized and would be treated as valid data.

**Q3:** When should `KNNImputer` be preferred over `SimpleImputer`?
> When features are correlated and a k-nearest-neighbors estimate based on similar rows will produce a more accurate imputed value than a global median or mode.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-5-Datasets]] — source dataset that undergoes the cleaning steps described here
- see:: [[Section-7-Data-Transformation]] — next step that encodes and splits the preprocessed data
- see:: [[Section-11-Preprocessing-the-Spam-Dataset]] — applies similar preprocessing logic to a different dataset

**Terms**
- data cleaning, imputation, SimpleImputer, KNNImputer, IterativeImputer, NaN, invalid IP, invalid port, regex, data validation, median imputation, categorical imputation, outliers
