---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 25 - Skills Assessment"]
lead: Skills assessment for the HTB Applications of AI in InfoSec module.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 25 - Skills Assessment

The [IMDB dataset](http://www.aclweb.org/anthology/P11-1015) (Maas et al., 2011) contains 50,000 movie reviews from the Internet Movie Database, split evenly into training and test sets with balanced positive and negative labels. The dataset has been a standard benchmark for sentiment analysis since its release and remains a useful baseline for evaluating text classification architectures.

The task is to train a binary classifier that predicts whether a review is positive (`1`) or negative (`0`). Download the dataset from the question or [directly here](https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip).

The same classification approach generalizes to other text moderation problems. Spam filtering, toxicity detection, and similar tasks all follow the same pattern.

---

Submit the trained model to the evaluation portal on the Playground VM. Use this script to upload from Jupyter:

```python
import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:5000/api/upload"

# Path to the model file you want to upload
model_file_path = "skills_assessment.joblib"

# Open the file in binary mode and send the POST request
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

# Pretty print the response from the server
print(json.dumps(response.json(), indent=4))
```

When working from a local machine, connect via HTB VPN, spawn the VM, then navigate to `http://VM-IP:5000/` and upload through the browser.

---

## Submission write-up

### What is the flag you get from submitting a good model for evaluation?

Flag : HTB{s3nt1m3nt_4n4lys1s_d4t4}

---

## Summary

- The skills assessment uses the IMDB dataset (50,000 movie reviews, balanced positive/negative labels) as a sentiment analysis benchmark.
- The task is binary text classification: predicting whether a review is positive (`1`) or negative (`0`).
- The trained model must be saved as `skills_assessment.joblib` and submitted to the evaluation portal at port 5000.
- The same text classification pattern (preprocessing → feature extraction → classifier) generalizes to spam filtering, toxicity detection, and similar NLP tasks.
- Submission uses `requests.post` to `http://localhost:5000/api/upload` from Jupyter, or via browser at `http://VM-IP:5000/`.

---

## Best Practices

- Reuse the full preprocessing pipeline from the spam classification sections (lowercase, strip punctuation, tokenize, remove stop words, stem) as a starting point for IMDB reviews.
- Use `CountVectorizer` or `TfidfVectorizer` with `ngram_range=(1, 2)` to capture bigrams and improve classification accuracy on longer review texts.
- Tune `MultinomialNB`'s `alpha` with `GridSearchCV` and `scoring="f1"` to find the optimal smoothing parameter for this dataset.
- Save the trained pipeline with `joblib.dump(model, "skills_assessment.joblib")` using the exact filename expected by the evaluation portal.
- Validate locally with a train/test split before submitting — poor local F1 scores will likely fail the portal evaluation threshold.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-9-Spam-Classification]] — spam classification domain covered in this module, relevant for text classification techniques
- see:: [[Section-15-Network-Anomaly-Detection]] — network anomaly detection domain covered in this module
- see:: [[Section-19-Malware-Classification]] — malware classification domain covered in this module

**Terms**
- IMDB dataset, sentiment analysis, movie review, binary classification, NLP, joblib, model upload, evaluation portal, positive/negative sentiment, text classification
