---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 12 - Feature Extraction"]
lead: Extracting numerical features from raw text for use in spam classification models.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 12 - Feature Extraction

Machine learning models cannot process raw text directly. `Feature extraction` converts preprocessed SMS messages into numerical vectors that classifiers can consume.

## Representing Text as Numerical Features

The `bag-of-words` model is the standard approach. It builds a vocabulary of unique terms from the training corpus and represents each message as a vector of term counts, one dimension per vocabulary term, with each value indicating how many times that term appears.

Using only `unigrams` (single words) discards all word-order information; the document is treated as an unordered collection of terms and their counts. To recover some local context, we also include `bigrams` (consecutive word pairs). The bigram `free prize`, for example, is more diagnostic of spam than the unigram `free` alone. Beyond these small windows, global sentence structure is still lost. `CountVectorizer` does not preserve full word order; it only captures localized patterns within the chosen `ngram_range`.

## Using CountVectorizer for the Bag-of-Words Approach

`CountVectorizer` from `scikit-learn` builds the vocabulary, tokenizes, and transforms documents into a term-count matrix in one step. Each row is a message; each column is a term from the vocabulary.

Key parameters:

- `min_df=1`: Include a term only if it appears in at least one document. Higher values filter out rare terms.
- `max_df=0.9`: Exclude terms appearing in more than 90% of documents; they are too common to discriminate between classes.
- `ngram_range=(1, 2)`: Capture both unigrams and bigrams.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer with bigrams, min_df, and max_df to focus on relevant terms
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))

# Fit and transform the message column
X = vectorizer.fit_transform(df["message"])

# Labels (target variable)
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)  # Converting labels to 1 and 0
```

After this step, `X` is a sparse numerical feature matrix ready for a classifier such as Naive Bayes.

### How CountVectorizer Works

`CountVectorizer` operates in three stages:

1. `Tokenization`: Splits each document into tokens according to `ngram_range`. For `ngram_range=(1, 2)` it produces both unigrams like `message` and bigrams like `free prize`.
2. `Building the Vocabulary`: Applies `min_df` and `max_df` to filter terms, retaining vocabulary that is both informative and distinctive.
3. `Vectorization`: Maps each document to a vector of term counts over the filtered vocabulary.

### Example with Unigrams

Consider five documents:

1. `The free prize is waiting for you`
2. `The spam message offers a free prize now`
3. `The spam filter might detect this`
4. `The important news says you won a free trip`
5. `The message truly is important`

With `ngram_range=(1, 1)` and `max_df=0.9`, the word `The` appears in all five documents (100%) and is removed. The resulting unigram matrix:

|Document|free|prize|is|waiting|for|you|spam|message|offers|a|now|filter|might|detect|this|important|news|says|won|trip|truly|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|2|1|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|
|3|0|0|0|0|0|0|1|0|0|0|0|1|1|1|1|0|0|0|0|0|0|
|4|1|0|0|0|0|1|0|0|0|1|0|0|0|0|0|1|1|1|1|1|0|
|5|0|0|1|0|0|0|0|1|0|0|0|0|0|0|0|1|0|0|0|0|1|

### Example with Bigrams

With `ngram_range=(1, 2)`, the vocabulary expands to include valid bigrams built from the unigrams above. `free prize` appears in Documents 1 and 2 and survives the `min_df` filter:

|Document|free|prize|is|waiting|for|you|spam|message|offers|a|now|filter|might|detect|this|important|news|says|won|trip|truly|free prize|prize is|is waiting|waiting for|for you|spam message|message offers|offers a|a free|prize now|spam filter|filter might|might detect|detect this|important news|news says|says you|you won|won a|free trip|message truly|truly is|is important|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|2|1|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|
|3|0|0|0|0|0|0|1|0|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|1|0|0|0|0|0|0|0|0|0|
|4|1|0|0|0|0|1|0|0|0|1|0|0|0|0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|1|1|1|1|1|1|0|0|0|
|5|0|0|1|0|0|0|0|1|0|0|0|0|0|0|1|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|1|

The resulting matrix gives each message a compact numeric representation that captures both individual words and locally ordered word pairs, ready for model training.

---

## Summary

- Feature extraction converts preprocessed text into numerical vectors that classifiers can consume.
- The bag-of-words model represents each message as a vector of term counts over a fixed vocabulary, discarding word order.
- Including bigrams (`ngram_range=(1, 2)`) recovers some local context — `free prize` is more diagnostic of spam than `free` alone.
- `min_df` filters out terms appearing in fewer than a threshold of documents; `max_df` removes terms too common to be discriminative.
- `CountVectorizer` builds the vocabulary, tokenizes, and vectorizes in one step, producing a sparse feature matrix.
- Converting labels to binary integers (`1` for spam, `0` for ham) is required before passing them to the classifier.

---

## Best Practices

- Set `max_df=0.9` in `CountVectorizer` to automatically exclude near-universal terms (e.g., function words that survived stop-word removal) without maintaining a manual list.
- Use `ngram_range=(1, 2)` to capture both individual tokens and locally ordered word pairs, improving discriminative signal for spam.
- Keep `min_df` at `1` or `2` for small datasets; increase it for large corpora to reduce vocabulary size and training time.
- Call `vectorizer.fit_transform(X_train)` on training data only, then `vectorizer.transform(X_test)` at inference to prevent vocabulary leakage from test data.
- Inspect `vectorizer.get_feature_names_out()` after fitting to verify that the vocabulary contains expected spam-relevant terms.
- Wrap vectorization and classification in a `Pipeline` so the same transforms are applied consistently at both training and inference.

---

## Quiz

**Q1:** What does `max_df=0.9` do in `CountVectorizer`?
> It excludes any term that appears in more than 90% of the documents, removing near-universal words that are too common to help distinguish between classes.

**Q2:** Why are bigrams useful for spam detection compared to using only unigrams?
> Bigrams capture locally ordered word pairs like `free prize`, which are more specific and diagnostic of spam than individual words alone; unigrams lose all word-order information.

**Q3:** What does `CountVectorizer` produce as output, and what is the format?
> It produces a sparse feature matrix where each row is a document and each column is a vocabulary term, with cell values representing how many times that term appears in the document.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-11-Preprocessing-the-Spam-Dataset]] — preprocessing produces the cleaned text that CountVectorizer transforms here
- see:: [[Section-7-Data-Transformation]] — vectorization is a form of data transformation converting text into numeric feature space
- see:: [[Section-13-Training-and-Evaluation-Spam-Detection]] — the feature matrix produced here is the direct input to model training

**Terms**
- CountVectorizer, bag-of-words, unigrams, bigrams, ngram_range, min_df, max_df, term frequency, vocabulary, feature matrix, sparse representation
