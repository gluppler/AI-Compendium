---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 10 - The Spam Dataset"]
lead: Exploring the spam dataset — structure, class balance, and initial data inspection.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 10 - The Spam Dataset

The [SMS Spam Collection dataset](https://web.archive.org/web/20260225150449if_/https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip) was assembled by Tiago A. Almeida, Akebo Yamakami (University of Campinas, Brazil), and José María Gómez Hidalgo (Optenet, Spain). Their 2011 ACM Symposium paper, "Contributions to the Study of SMS Spam Filtering: New Collection and Results," addressed the lack of SMS-specific spam corpora; most prior datasets targeted email. They drew from sources including the Grumbletext website, the NUS SMS Corpus, and Caroline Tag's PhD thesis.

The corpus contains 5,574 text messages labelled as either `ham` (legitimate) or `spam` (unwanted). `Ham` covers messages from known contacts, newsletters, or subscriptions that hold value for the recipient; `spam` covers unsolicited content that provides no benefit and may pose a risk.

## Downloading the Dataset

```python
import requests
import zipfile
import io

# URL of the dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# Download the dataset
response = requests.get(url)
if response.status_code == 200:
    print("Download successful")
else:
    print("Failed to download the dataset")
```

`requests` sends an HTTP GET to the dataset URL. A `status_code` of `200` confirms success.

The dataset arrives as a `.zip` archive. Extract it with `zipfile` and `io`:

```python
# Extract the dataset
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")
    print("Extraction successful")
```

`io.BytesIO` wraps the binary response body so `zipfile.ZipFile` can read it directly. `extractall` writes all archive contents to the `sms_spam_collection` directory.

Confirm the extraction succeeded:

```python
import os

# List the extracted files
extracted_files = os.listdir("sms_spam_collection")
print("Extracted files:", extracted_files)
```

## Loading the Dataset

The `SMSSpamCollection` file uses tab-separated values with no header row:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)
```

`sep="\t"` tells pandas the delimiter is a tab. `header=None` signals there is no header row, and `names` provides the column labels manually.

Inspect the loaded data with `head`, `describe`, and `info`:

```python
# Display basic information about the dataset
print("-------------------- HEAD --------------------")
print(df.head())
print("-------------------- DESCRIBE --------------------")
print(df.describe())
print("-------------------- INFO --------------------")
print(df.info())
```

- `df.head()` shows the first five rows for a quick sanity check.
- `df.describe()` gives summary statistics; for a text dataset this mainly reflects the label distribution.
- `df.info()` reports column types and non-null counts.

Check for missing values:

```python
# Check for missing values
print("Missing values:\n", df.isnull().sum())
```

`isnull().sum()` counts null entries per column; any non-zero value requires attention before training.

Identify and remove duplicate rows:

```python
# Check for duplicates
print("Duplicate entries:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()
```

`duplicated().sum()` counts rows that are identical to an earlier row. `drop_duplicates` removes them, preventing the model from seeing the same example multiple times and inflating its apparent performance.

---

## Summary

- The SMS Spam Collection dataset contains 5,574 tab-separated messages labeled as `ham` (legitimate) or `spam`.
- The dataset was assembled from multiple sources including Grumbletext, the NUS SMS Corpus, and a PhD thesis by Caroline Tag.
- Its primary contribution was filling the gap for SMS-specific spam corpora, since most prior datasets targeted email.
- The dataset is loaded with `pd.read_csv(..., sep="\t", header=None, names=["label", "message"])` due to its tab-separated, headerless format.
- Initial inspection uses `head()`, `describe()`, and `info()` to understand structure and detect anomalies.
- Duplicate rows must be identified and removed to prevent the model from inflating its apparent performance.

---

## Best Practices

- Use `sep="\t"` and `header=None` with explicit `names` when loading TSV files without headers to avoid misaligned columns.
- Always check for duplicate rows with `df.duplicated().sum()` and remove them before training to prevent the model from over-fitting to repeated examples.
- Check for missing values with `df.isnull().sum()` immediately after loading — non-zero counts require attention before preprocessing.
- Inspect `df.describe()` on text datasets to understand label distribution and catch unexpected class imbalances early.
- Verify the extraction of zip archives with `os.listdir()` before loading files to confirm the expected filenames are present.

---

## Quiz

**Q1:** Why was the SMS Spam Collection dataset significant when it was released?
> It addressed the lack of SMS-specific spam corpora — most prior datasets targeted email, making SMS-based spam filtering difficult to benchmark.

**Q2:** What pandas parameters are needed to correctly load the SMSSpamCollection file?
> `sep="\t"` (tab delimiter), `header=None` (no header row), and `names=["label", "message"]` (manual column names).

**Q3:** Why is it important to remove duplicate rows before training a spam classifier?
> Duplicate rows allow the model to see the same example multiple times, inflating its apparent performance on held-out data that does not contain duplicates.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-9-Spam-Classification]] — provides the Naive Bayes theory that will be applied to this dataset
- see:: [[Section-11-Preprocessing-the-Spam-Dataset]] — the next step after loading this dataset is text preprocessing
- see:: [[Section-5-Datasets]] — general dataset loading and inspection practices applied here

**Terms**
- SMS Spam Collection, ham, spam, TSV format, pandas DataFrame, missing values, duplicate entries, class imbalance, UCI repository, corpus annotation
