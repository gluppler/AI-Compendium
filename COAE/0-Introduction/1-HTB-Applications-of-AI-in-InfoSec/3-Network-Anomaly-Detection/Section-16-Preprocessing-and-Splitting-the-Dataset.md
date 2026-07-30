---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 16 - Preprocessing and Splitting the Dataset"]
lead: Preprocessing the NSL-KDD dataset and creating train/test splits for anomaly detection.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 16 - Preprocessing and Splitting the Dataset

## Preprocessing the Dataset

The raw NSL-KDD data requires three transformations before model training: creating classification targets (binary and multi-class), encoding categorical features as numeric indicators, and selecting the numeric columns that capture meaningful traffic statistics.

### Creating a Binary Classification Target

The binary target collapses the problem to normal vs. attack. A new column `attack_flag` receives `0` for normal traffic and `1` for any attack type:

```python
# Binary classification target
# Maps normal traffic to 0 and any type of attack to 1
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)
```

Each row in the dataset carries the string `normal` or a specific attack name, for example:

```python
0,tcp,ftp_data,SF,491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.0,0.0,0.0,0.0,1.0,0.0,0.0,150,25,0.17,0.03,0.17,0.0,0.0,0.0,0.05,0.0,normal,20
0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,6,1.0,1.0,0.0,0.0,0.05,0.07,0.0,255,26,0.1,0.05,0.0,0.0,1.0,1.0,0.0,0.0,neptune,19
```

### Creating the Multi-Class Classification Target

A single binary flag hides the nature of the attack. The multi-class target maps each attack to one of four categories, with `0` reserved for normal traffic:

- `1`: DoS (Denial of Service) — `neptune`, `smurf`, `back`, `land`, `pod`, `teardrop`, and others
- `2`: Probe — reconnaissance attacks such as `satan`, `ipsweep`, `nmap`, `portsweep`
- `3`: Privilege Escalation — attempts to gain admin control, e.g. `buffer_overflow`, `rootkit`
- `4`: Access — attacks targeting system access controls, e.g. `guess_passwd`, `ftp_write`

```python
# Multi-class classification target categories
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
                'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps',
                      'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap',
                   'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
                 'snmpguess', 'spy', 'warezclient', 'warezmaster',
                 'xclock', 'xsnoop']

def map_attack(attack):
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in privilege_attacks:
        return 3
    elif attack in access_attacks:
        return 4
    else:
        return 0

# Assign multi-class category to each row
df['attack_map'] = df['attack'].apply(map_attack)
```

### Encoding Categorical Variables

`protocol_type` (e.g., `tcp`, `udp`) and `service` (e.g., `http`, `ftp`) are categorical. Most ML algorithms require numeric input, and treating categories as integers would imply a false ordinal relationship. One-hot encoding via `pd.get_dummies` creates a separate binary column for each category value:

```python
# Encoding categorical variables
features_to_encode = ['protocol_type', 'service']
encoded = pd.get_dummies(df[features_to_encode])
```

### Selecting Numeric Features

The dataset provides a rich set of numeric metrics covering raw volume (`src_bytes`, `dst_bytes`), session counts (`count`, `srv_count`), and derived statistical rates (`serror_rate`, `dst_host_srv_diff_host_rate`). These capture both coarse and subtle traffic patterns:

```python
# Numeric features that capture various statistical properties of the traffic
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
```

### Preparing the Dataset

Combine the one-hot encoded categorical features with the numeric features into a single DataFrame, and extract the multi-class target:

```python
# Combine encoded categorical variables and numeric features
train_set = encoded.join(df[numeric_features])

# Multi-class target variable
multi_y = df['attack_map']
```

## Splitting the Dataset

### Splitting Data into Training and Test Sets

Reserve 20% of the data for the final test evaluation:

```python
# Split data into training and test sets for multi-class classification
train_X, test_X, train_y, test_y = train_test_split(train_set, multi_y, test_size=0.2, random_state=1337)
```

### Creating a Validation Set from the Training Data

Further divide the training set to obtain a validation subset for hyperparameter tuning, keeping the test set clean:

```python
# Further split the training set into separate training and validation sets
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1337)
```

### Final Split Variables

After splitting, four sets are available:

- `train_X`, `train_y`: Full training set (before the validation split)
- `test_X`, `test_y`: Held-out final evaluation set
- `multi_train_X`, `multi_train_y`: Training subset used to fit the model
- `multi_val_X`, `multi_val_y`: Validation subset used for tuning

This partitioning ensures the test set reflects unbiased real-world performance and is never contaminated by tuning decisions.

---

## Summary

- Three preprocessing transformations are applied to the raw NSL-KDD data: binary target creation, multi-class target creation, and categorical encoding.
- The binary target `attack_flag` maps normal traffic to `0` and any attack type to `1`.
- The multi-class target `attack_map` assigns `0` for normal and `1–4` for DoS, Probe, Privilege Escalation, and Access attacks respectively.
- Categorical columns `protocol_type` and `service` are one-hot encoded with `pd.get_dummies` to eliminate false ordinal relationships.
- Numeric features covering traffic statistics (bytes, counts, rates) are selected and combined with the encoded categorical columns.
- The dataset is split 80/20 into training and test sets, then the training portion is further divided 70/30 to create a validation set.

---

## Best Practices

- Create both binary and multi-class targets before splitting — this avoids label mismatch errors if targets are derived after the split.
- Use `pd.get_dummies` for simple one-hot encoding when the full DataFrame is available; reserve `OneHotEncoder` for pipelines requiring `fit`/`transform` separation.
- Keep the test set completely untouched during training and tuning; only evaluate on it once to avoid optimism bias.
- Verify final split sizes by printing `len(multi_train_X)`, `len(multi_val_X)`, and `len(test_X)` to confirm the expected proportions.
- Use `random_state=1337` (or any fixed value) in all `train_test_split` calls to ensure reproducible partitioning.

---

## Quiz

**Q1:** What is the difference between the `attack_flag` and `attack_map` columns created during preprocessing?
> `attack_flag` is a binary label (0 = normal, 1 = any attack); `attack_map` is a multi-class label (0 = normal, 1 = DoS, 2 = Probe, 3 = Privilege Escalation, 4 = Access) that preserves the type of attack.

**Q2:** Why are `protocol_type` and `service` one-hot encoded rather than label-encoded?
> One-hot encoding prevents the model from inferring a false ordinal relationship between protocols or services; label encoding would assign integers that imply an ordering (e.g., `tcp` < `udp`) that does not exist.

**Q3:** After two sequential `train_test_split` calls with `test_size=0.2` then `test_size=0.3`, what are the approximate proportions of the final train, validation, and test sets?
> Train ≈ 56%, validation ≈ 24%, test ≈ 20% of the full dataset.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-7-Data-Transformation]] — one-hot encoding and train/test splitting concepts introduced in Section 7 are applied here
- see:: [[Section-15-Network-Anomaly-Detection]] — the NSL-KDD dataset loaded in Section 15 is preprocessed in this section
- see:: [[Section-17-Training-and-Evaluation-Network-Anomaly-Detection]] — the train/validation/test splits produced here feed directly into model training

**Terms**
- binary classification target, multi-class classification, one-hot encoding, attack_flag, attack_map, DoS attacks, probe attacks, privilege escalation, access attacks, train_test_split, validation set
