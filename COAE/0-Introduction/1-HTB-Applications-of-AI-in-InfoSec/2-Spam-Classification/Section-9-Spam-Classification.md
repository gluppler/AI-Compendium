---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 9 - Spam Classification"]
lead: Using Naive Bayes and TF-IDF to classify spam email from raw text features.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 9 - Spam Classification

Spam, unsolicited bulk messaging, has plagued digital communication since its earliest days. Beyond cluttering inboxes, it serves as a delivery vehicle for phishing attacks and other threats. Effective spam detection is a core requirement for keeping email systems both usable and secure.

## Naive Bayes for Spam Detection

Bayes' Theorem relates the probability of an event to prior knowledge about related conditions:

```python
P(A|B) = (P(B|A) * P(A)) / P(B)
```

Where:

- `P(A|B)` is the probability of event `A` occurring, given that `B` is true.
- `P(B|A)` is the probability of event `B` occurring, given that `A` is true.
- `P(A)` is the prior probability of event `A`.
- `P(B)` is the prior probability of event `B`.

For spam detection, `A` is the hypothesis that an email is spam, and `B` is the set of observed features: words, phrases, and so on.

### Applying Bayes' Theorem to Spam Detection

Breaking down the formula for spam classification:

1. `Hypothesis`: We want `P(Spam|Features)`, the probability that an email is spam given its content.
2. `Likelihood`: `P(Features|Spam)`, how likely those features are in a spam email.
3. `Prior Probability`: `P(Spam)`, the base rate of spam across all emails.
4. `Marginal Likelihood`: `P(Features)`, the total probability of observing those features in any email.

Applying the theorem:

```python
P(Spam|Features) = (P(Features|Spam) * P(Spam)) / P(Features)
```

### Simplifying with Naive Assumptions

Naive Bayes assumes each feature is conditionally independent of every other feature given the class label. This makes computing `P(Features|Spam)` tractable:

```python
P(Features|Spam) = P(feature1|Spam) * P(feature2|Spam) * ... * P(featureN|Spam)
```

Similarly for ham:

```python
P(Features|Not Spam) = P(feature1|Not Spam) * P(feature2|Not Spam) * ... * P(featureN|Not Spam)
```

The classifier picks whichever class yields the higher posterior probability.

### Example Calculation

Consider an email with features `F1` and `F2`:

- `P(Spam) = 0.3`
- `P(Not Spam) = 0.7`
- `P(F1|Spam) = 0.4`, `P(F2|Spam) = 0.5`
- `P(F1|Not Spam) = 0.2`, `P(F2|Not Spam) = 0.3`

Under the naive independence assumption:

```python
P(F1, F2|Spam) = P(F1|Spam) * P(F2|Spam) = 0.4 * 0.5 = 0.2
P(F1, F2|Not Spam) = P(F1|Not Spam) * P(F2|Not Spam) = 0.2 * 0.3 = 0.06
```

Applying the law of total probability to find `P(F1, F2)`:

```python
P(F1, F2) = P(F1, F2|Spam) * P(Spam) + P(F1, F2|Not Spam) * P(Not Spam)
           = (0.2 * 0.3) + (0.06 * 0.7)
           = 0.06 + 0.042
           = 0.102
```

Posterior probabilities:

```python
P(Spam|F1, F2) = (0.2 * 0.3) / 0.102
               = 0.06 / 0.102
               ≈ 0.588

P(Not Spam|F1, F2) = (0.06 * 0.7) / 0.102
                   = 0.042 / 0.102
                   ≈ 0.412
```

Since `P(Spam|F1, F2) > P(Not Spam|F1, F2)`, the email is classified as spam.

---

## Summary

- Naive Bayes classifiers use Bayes' Theorem to compute the posterior probability that a message is spam given its observed features.
- The "naive" assumption is that each feature is conditionally independent of every other feature given the class label.
- This independence assumption makes the computation tractable: `P(Features|Spam)` becomes a product of per-feature probabilities.
- The classifier picks the class with the higher posterior probability as its prediction.
- Spam detection maps naturally to the Naive Bayes framework: the hypothesis is spam, and features are words and phrases in the message.
- Despite the unrealistic independence assumption, Naive Bayes performs well in practice for text classification tasks.

---

## Best Practices

- Use `MultinomialNB` for text classification since it is designed for discrete count features such as word frequencies.
- Tune the `alpha` smoothing parameter (Laplace smoothing) to prevent zero probabilities for words unseen during training.
- Always preprocess text consistently at both training and inference time — inconsistent preprocessing is a leading source of degraded production accuracy.
- Validate the model with precision, recall, and F1-score rather than accuracy alone, especially with imbalanced spam/ham ratios.
- Inspect `predict_proba` outputs alongside predictions to understand model confidence, not just the binary label.

---

## Quiz

**Q1:** What does "naive" mean in the context of Naive Bayes?
> It means the algorithm assumes each feature (word or phrase) is conditionally independent of every other feature given the class label, which simplifies the computation even though this assumption rarely holds perfectly in practice.

**Q2:** Given `P(Spam) = 0.3` and `P(F1,F2|Spam) = 0.2`, what is `P(Spam|F1,F2)` if `P(F1,F2) = 0.102`?
> `P(Spam|F1,F2) = (0.2 × 0.3) / 0.102 ≈ 0.588`

**Q3:** Why does the Naive Bayes classifier pick the class with the higher posterior probability?
> The posterior probability `P(Class|Features)` represents how likely a given class is after observing the message's features; the class with the higher posterior is the most probable explanation for the observed data.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-10-The-Spam-Dataset]] — introduces the SMS dataset used to apply the Naive Bayes classifier described here
- see:: [[Section-8-Metrics-for-Evaluating-a-Model]] — precision, recall, and F1 are the metrics used to assess spam classifier performance
- see:: [[Section-12-Feature-Extraction]] — feature extraction transforms preprocessed text into the numeric input Naive Bayes requires

**Terms**
- Naive Bayes, Bayes' Theorem, prior probability, posterior probability, likelihood, marginal likelihood, conditional independence, spam detection, feature independence assumption, probabilistic classifier
