---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 3 - Manipulating the Model"]
lead: Practical demonstration of input manipulation and data poisoning attacks against an ML spam classifier.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

This section demonstrates how ML models respond to manipulated inputs and poisoned training data, making the ML01 (Input Manipulation) and ML02 (Data Poisoning) attack vectors concrete.

The baseline is the spam classifier from the [Applications of AI in InfoSec](https://academy.hackthebox.com/module/details/292) module, which is recommended as a prerequisite. A slightly adjusted version of that code is provided in the section resources.

---

## Manipulating the input

The project trains a classifier on `train.csv` and evaluates it on `test.csv`:

```python
model = train("./train.csv")
acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")
```

The classifier achieves 97.2% accuracy on the full training set:
```bash
gluppler@htb[/htb]$ python3 main.py
Model accuracy: 97.2%
```

To probe the model's response to individual inputs, use `classify_messages` with `return_probabilities=True`. This returns per-class output probabilities rather than the predicted label, exposing how confident the model is for a given input:

```python
model = train("./train.csv")
message = "Hello World! How are you doing?"
predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]
print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

```bash
gluppler@htb[/htb]$ python3 main.py
Predicted class: Ham
Probabilities:
     Ham: 98.93%
   Spam: 1.07%
```

Switching to a typical spam message (`Congratulations! You won a prize. Click here to claim: https://bit.ly/3YCN7PF`), the model is equally confident in the other direction:

```bash
gluppler@htb[/htb]$ python3 main.py
Predicted class: Spam
Probabilities:
     Ham: 0.0%
   Spam: 100.0%
```

The input manipulation goal is to get a spam message classified as ham.

#### Rephrasing

The first technique maps the model's sensitivity to specific words or phrases, then rewrites the message to avoid triggering high spam probabilities.

Testing individual components of the spam message:

| Input Message | Spam Probability | Ham Probability |
|---|---|---|
| `Congratulations!` | 64.97% | 35.03% |
| `Congratulations! You won a prize.` | 99.73% | 0.27% |
| `Click here to claim: https://bit.ly/3YCN7PF` | 99.34% | 0.66% |
| `https://bit.ly/3YCN7PF` | 87.29% | 12.71% |

The word `Congratulations!` alone is sufficient to push the model toward spam. By switching to a different scenario that avoids these lexical markers, the spam content clears the classifier. The message `Your account has been blocked. You can unlock your account in the next 24h: https://bit.ly/3YCN7PF` is classified as ham:

```bash
gluppler@htb[/htb]$ python3 main.py
Predicted class: Ham
Probabilities:
     Ham: 57.39%
   Spam: 42.61%
```

#### Overpowering

The second technique appends a large volume of benign text to the spam message. The Naive Bayes classifier treats each word as an independent contribution to the final probability. Flooding the input with ham-associated words overwhelms the spam signal without removing it.

Adding the opening sentence of Lorem Ipsum after the original spam content:

```text
Congratulations! You won a prize. Click here to claim: https://bit.ly/3YCN7PF. But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness.
```

```bash
gluppler@htb[/htb]$ python3 main.py
Predicted class: Ham
Probabilities:
     Ham: 100.0%
   Spam: 0.0%
```

This technique is particularly effective when the delivery channel supports hiding content from the recipient. In HTML emails, for example, the appended text can sit inside comments that the spam filter processes but the reader never sees.

---

## Manipulating the training data

To amplify the effect of data poisoning, extract a smaller training set that is more sensitive to individual injected samples:

```bash
gluppler@htb[/htb]$ head -n 101 train.csv > poison.csv
```

Update `main.py` to train on `poison.csv`:

```bash
gluppler@htb[/htb]$ python3 main.py
Model accuracy: 94.4%
```

Accuracy drops slightly to 94.4%, expected given the significant reduction in training data. The reduced dataset is more sensitive to injected entries, which is the behavior to exploit.

Set up the script to print output probabilities for a target input:

```python
model = train("./poison.csv")
message = "Hello World! How are you doing?"
predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]
print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

The baseline: the message is classified as ham at 98.7% confidence. The goal is to flip this to spam by injecting poisoned training samples.

Append two fake spam entries to `poison.csv`:

```text
spam,Hello World
spam,How are you doing?
```

After retraining:
```bash
gluppler@htb[/htb]$ python3 main.py
Predicted class: Spam
Probabilities:
     Ham: 20.34%
   Spam: 79.66%
```

Two injected entries are sufficient to flip the classification. Adding two more entries using phrase combinations increases the confidence further:
```text
spam,Hello World! How are you
spam,World! How are you doing?
```

Note that duplicate entries are deduplicated before training, so repeating the same sample has no additive effect. After the second round of injection:
```bash
gluppler@htb[/htb]$ python3 main.py
Predicted class: Spam
Probabilities:
     Ham: 0.4%
   Spam: 99.6%
```

Adding the evaluation loop back confirms the cost to overall accuracy:
```python
model = train("./poison.csv")
acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")
message = "Hello World! How are you doing?"
predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]
print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

```bash
gluppler@htb[/htb]$ python3 main.py
Model accuracy: 94.0%
Predicted class: Spam
Probabilities:
     Ham: 0.4%
   Spam: 99.6%
```

Overall accuracy fell only 0.4 percentage points. The targeted misclassification was achieved with negligible observable impact on aggregate metrics, which is what makes data poisoning attacks both powerful and difficult to detect in production. The amplified effect shown here is a direct result of the deliberately shrunken training set; a production-scale dataset would require far more injected samples to achieve the same influence.

---

## Submission write-up

### Question 1
Manipulate the fixed input message by appending data to trick the classifier into classifying the message as ham. Submit the flag you obtain after providing an input that satisfies the lab requirements.

Flag : HTB{9b8de0fd17f2166743cd59f7ec876ac7}

### Question 2
Manipulate the training data to reduce the trained classifier's accuracy below 70%. Submit the flag you obtain after providing a dataset that satisfies the lab requirements.

Flag : HTB{8ba5eff39c343c3b0170e6bb1704df02}

### Question 3
Exploit a flaw in the web application to steal the trained model. Submit the file's MD5 hash as the flag.

Flag : 8007cd6c209a40399cf3ca82dd7db02c

---

## Summary

- Input manipulation against a Naive Bayes spam classifier can be achieved via rephrasing (identifying high-weight spam tokens and rewriting the message to avoid them) or overpowering (flooding the input with ham-associated words to drown out spam signal).
- Overpowering works because Naive Bayes treats each word as an independent probability contribution — appending a large benign text block drives overall spam probability toward zero.
- Overpowering is especially effective in HTML emails where the appended benign text can be hidden in comments invisible to the recipient but processed by the filter.
- Data poisoning on a small training set can flip a target classification (ham → spam) with as few as two injected, mislabeled samples.
- Duplicate training entries are deduplicated before training — repeated identical samples have no additive poisoning effect; phrase variation is needed to increase influence.
- Targeted data poisoning achieves the desired misclassification while reducing overall accuracy by only a fraction of a percent, making it hard to detect through aggregate metrics alone.

---

## Best Practices

- Monitor per-class accuracy distributions rather than aggregate accuracy alone, as targeted poisoning barely degrades overall metrics while flipping specific predictions.
- Implement deduplication-aware data validation; verify label consistency across semantically similar training samples to catch label-flipping poisoning attempts.
- Apply input length and token-diversity checks at inference time to detect overpowering attacks that append anomalously large benign payloads to spam content.
- Maintain versioned, auditable training datasets with write-access controls to limit who can inject samples into the training pipeline.
- Use ensemble or adversarially trained classifiers that are less sensitive to individual word-probability contributions, reducing susceptibility to rephrasing attacks.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Attacking-ML-based-Systems]] — ML OWASP Top 10 describing ML01 and ML02 theoretically

**Terms**
- input manipulation, data poisoning, Naive Bayes, spam classifier, overpowering, rephrasing, training data poisoning, backdoor
