---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 7 - Naive Bayes"]
lead: Naive Bayes applies Bayes' theorem with conditional independence assumptions for efficient probabilistic classification.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 7."
---

![[bayes_classification.png]]

`Naive Bayes` is a probabilistic classification algorithm built on `Bayes' theorem`. It is widely used for tasks like spam filtering and sentiment analysis because it is fast, easy to implement, and performs well even with limited training data.

## Bayes' Theorem

`Bayes' theorem` provides a way to update the probability of an event given new evidence:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Where:

- `P(A|B)`: posterior probability — probability of `A` given `B` has occurred.
- `P(B|A)`: likelihood — probability of observing `B` given `A` is true.
- `P(A)`: prior probability of `A`.
- `P(B)`: marginal probability of `B`.

**Worked example — disease test:**

- Disease prevalence: `P(A) = 0.01`
- Test sensitivity: `P(B|A) = 0.95`
- False positive rate: `P(B|¬A) = 0.05`
- `P(¬A) = 0.99`

First, compute the total probability of a positive test result using the law of total probability:

$$P(B) = P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)$$

$$P(B) = (0.95 \times 0.01) + (0.05 \times 0.99) = 0.0095 + 0.0495 = 0.059$$

Then apply Bayes' theorem:

$$P(A|B) = \frac{0.95 \times 0.01}{0.059} = \frac{0.0095}{0.059} \approx 0.161$$

Despite a 95% accurate test, the posterior probability of actually having the disease is only ~16.1%. Low disease prevalence dominates — a reminder that prior probability always matters.

## How Naive Bayes Works

`Naive Bayes` extends Bayes' theorem to multi-feature classification by assuming **conditional independence** among features given the class label. The algorithm:

1. Calculates the **prior probability** of each class from the training data (e.g., 20% of emails are spam).
2. Estimates the **likelihood** of each feature value given each class (e.g., how often does "free" appear in spam vs. non-spam?).
3. Applies Bayes' theorem to compute the **posterior probability** of each class given the observed features.
4. Assigns the data point to the class with the highest posterior probability.

The independence assumption is rarely true in practice — words like "free" and "money" co-occur in spam — but the classifier still generalizes well because it needs only the correct relative ordering of posterior probabilities, not calibrated absolute values.

### Types of Naive Bayes Classifiers

The variant to use depends on the feature type:

- `Gaussian Naive Bayes`: For continuous features assumed to follow a Gaussian distribution. Suitable for numerical inputs like age or income.
- Multinomial Naive Bayes: For discrete count features. Standard choice for text classification, where features are word frequencies.
- Bernoulli Naive Bayes: For binary features (present/absent). Used when the task is whether a term appears in a document, not how often.

## Data Assumptions

- `Feature Independence`: Features must be conditionally independent given the class — the core "naive" assumption.
- `Data Distribution`: The chosen variant (Gaussian, Multinomial, Bernoulli) must match the actual feature distribution.
- Sufficient Training Data: Probability estimates become unreliable with very few samples per class.

---

## Summary

- Naive Bayes is a probabilistic classifier built on Bayes' theorem, using prior and likelihood probabilities to compute the posterior probability of each class.
- The "naive" assumption is conditional independence of features given the class label — rarely true in practice, but the classifier still generalizes well.
- Bayes' theorem: `P(A|B) = P(B|A) * P(A) / P(B)` — the posterior probability is proportional to the likelihood times the prior.
- The disease-test worked example illustrates that low base rate (prior) dominates posterior even when the test has high sensitivity.
- Three variants exist: Gaussian (continuous features), Multinomial (count/frequency features), and Bernoulli (binary presence/absence features).
- Naive Bayes is fast, interpretable, and works well with limited training data — commonly used for spam filtering and sentiment analysis.

---

## Best Practices

- Choose the variant that matches the feature type: Gaussian for continuous numerical inputs, Multinomial for word frequencies, Bernoulli for binary term presence.
- Always inspect the prior class probabilities — if classes are heavily imbalanced, the prior alone may dominate the posterior and require careful calibration.
- Consider Laplace smoothing (add-one smoothing) to avoid zero-probability likelihoods when a feature value has not appeared with a class in training data.
- Validate the independence assumption on your data — features with strong correlations will degrade calibration, though classification accuracy often remains reasonable.
- When interpretability matters, Naive Bayes is a strong baseline: its learned likelihoods and priors are directly inspectable.

---

## Quiz

**Q1:** State Bayes' theorem and identify each term.
> `P(A|B) = P(B|A) * P(A) / P(B)`. `P(A|B)` is the posterior, `P(B|A)` is the likelihood, `P(A)` is the prior, and `P(B)` is the marginal probability of the evidence.

**Q2:** Why does Naive Bayes perform well in practice despite the independence assumption being violated?
> For classification, only the relative ordering of posterior probabilities matters, not their absolute values. The independence assumption often still produces the correct relative ordering even when features co-occur.

**Q3:** In the disease-test example, why is the posterior probability of having the disease only ~16% despite a 95% accurate test?
> The disease prevalence (prior) is only 1%. A 5% false-positive rate on the 99% disease-free population generates many more false positives than the 95% true-positive rate on the 1% diseased population, making most positive tests false positives.

**Q4:** When should you use Multinomial Naive Bayes instead of Bernoulli Naive Bayes for text classification?
> Use Multinomial when the feature represents word frequency (how often a term appears); use Bernoulli when the feature represents only term presence or absence, regardless of count.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Mathematics-Refresher-for-AI]] — Bayes theorem relies on conditional probability notation
- see:: [[Section-5-Logistic-Regression]] — both are probabilistic classifiers for binary/multiclass tasks

**Terms**
- Bayes theorem, conditional independence, prior probability, posterior probability, likelihood, class probability
