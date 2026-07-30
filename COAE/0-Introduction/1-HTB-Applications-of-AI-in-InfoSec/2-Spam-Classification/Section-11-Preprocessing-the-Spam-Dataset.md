---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 11 - Preprocessing the Spam Dataset"]
lead: Preprocessing text for spam classification — tokenization, stop-word removal, and vectorization.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 11 - Preprocessing the Spam Dataset

Raw SMS text must be standardized and cleaned before a Naive Bayes classifier can use it. The preprocessing pipeline here uses `nltk` for tokenization, stop word removal, and stemming. Download the required NLTK data files first to avoid interruptions later:

```python
import nltk

# Download the necessary NLTK data files
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
print("=== BEFORE ANY PREPROCESSING ===")
print(df.head(5))
```

### Lowercasing the text

Converting all text to lowercase ensures the classifier treats `Free` and `free` as the same token. This reduces vocabulary size and prevents the model from learning spurious case-based distinctions:

```python
# Convert all message text to lowercase
df["message"] = df["message"].str.lower()
print("\n=== AFTER LOWERCASING ===")
print(df["message"].head(5))
```

## Removing Punctuation and Numbers

Most punctuation and all digits carry little signal for spam classification, so they are stripped. However, `$` and `!` are worth keeping: the former often signals a monetary lure, the latter adds emphasis common in spam. The regex below removes everything except lowercase letters, whitespace, `$`, and `!`:

```python
import re

# Remove non-essential punctuation and numbers, keep useful symbols like $ and !
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\n=== AFTER REMOVING PUNCTUATION & NUMBERS (except $ and !) ===")
print(df["message"].head(5))
```

## Tokenizing the Text

Tokenization splits each message string into a list of individual word tokens. Downstream operations (stop word removal and stemming) need to act on individual words, not whole sentences:

```python
from nltk.tokenize import word_tokenize

# Split each message into individual tokens
df["message"] = df["message"].apply(word_tokenize)
print("\n=== AFTER TOKENIZATION ===")
print(df["message"].head(5))
```

## Removing Stop Words

Stop words such as `and`, `the`, and `is` occur in both spam and ham with similar frequency, contributing noise rather than signal. Removing them shortens the token lists and focuses the model on discriminative vocabulary:

```python
from nltk.corpus import stopwords

# Define a set of English stop words and remove them from the tokens
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n=== AFTER REMOVING STOP WORDS ===")
print(df["message"].head(5))
```

## Stemming

Stemming reduces each token to its root form: `running` becomes `run`, `prizes` becomes `prize`. This consolidates morphological variants under one token, shrinking the vocabulary and improving the model's ability to generalize across word forms:

```python
from nltk.stem import PorterStemmer

# Stem each token to reduce words to their base form
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\n=== AFTER STEMMING ===")
print(df["message"].head(5))
```

## Joining Tokens Back into a Single String

Vectorizers such as `CountVectorizer` and `TfidfVectorizer` expect raw strings, not token lists. Rejoining the tokens with spaces restores the string format required for feature extraction:

```python
# Rejoin tokens into a single string for feature extraction
df["message"] = df["message"].apply(lambda x: " ".join(x))
print("\n=== AFTER JOINING TOKENS BACK INTO STRINGS ===")
print(df["message"].head(5))
```

Each message is now a cleaned, normalized string ready for vectorization and model training.

---

## Summary

- Raw SMS text must be standardized through a pipeline before a Naive Bayes classifier can use it.
- The pipeline steps in order are: lowercase conversion, punctuation and number removal, tokenization, stop-word removal, and stemming.
- `$` and `!` are intentionally preserved during punctuation removal because they are discriminative spam signals.
- Stop words (e.g., `and`, `the`, `is`) are removed because they appear with similar frequency in spam and ham and add noise.
- Stemming with `PorterStemmer` reduces morphological variants (e.g., `running` → `run`), shrinking the vocabulary.
- Tokens must be rejoined into a single space-delimited string before vectorizers like `CountVectorizer` can process them.

---

## Best Practices

- Download NLTK data files (`punkt`, `punkt_tab`, `stopwords`) once at the top of the notebook to avoid interruptions mid-pipeline.
- Preserve domain-significant symbols like `$` and `!` in the regex rather than stripping all non-alphabetic characters.
- Apply `str.lower()` before tokenization so tokens are consistent regardless of original capitalization.
- Use `set(stopwords.words("english"))` for O(1) lookup speed when filtering tokens against the stop-word list.
- Rejoin stemmed tokens with `" ".join(tokens)` after stemming to restore the string format required by `CountVectorizer` and `TfidfVectorizer`.
- Apply the exact same preprocessing function to inference-time messages as to training data.

---

## Quiz

**Q1:** Why are `$` and `!` deliberately kept during punctuation removal for spam classification?
> Both are commonly used in spam messages: `$` signals monetary lures and `!` adds emphasis typical of unsolicited bulk messages, making them discriminative features rather than noise.

**Q2:** What is the purpose of stemming in the text preprocessing pipeline?
> Stemming reduces tokens to their root form (e.g., `prizes` → `prize`), consolidating morphological variants under one token to shrink the vocabulary and improve generalization.

**Q3:** Why must tokenized lists be rejoined into a single string before feature extraction?
> Vectorizers like `CountVectorizer` and `TfidfVectorizer` expect raw string documents as input, not Python lists of tokens.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-7-Data-Transformation]] — data transformation principles (normalization, encoding) underpin the preprocessing pipeline used here
- see:: [[Section-10-The-Spam-Dataset]] — this preprocessing is applied directly to the SMS dataset loaded in Section 10
- see:: [[Section-12-Feature-Extraction]] — preprocessed tokens feed directly into feature extraction via CountVectorizer

**Terms**
- tokenization, stop word removal, stemming, PorterStemmer, lowercasing, punctuation removal, NLTK, word_tokenize, text normalization, vocabulary reduction
