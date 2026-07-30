# Red Teaming ML — Manual Step-by-Step Guide

## Target: `http://154.57.164.72:32382/`

---

## Lab 1: Input Manipulation (ML01)

**Goal**: Get a spam classifier to classify a spam message as **ham** by appending text.

### Steps

1. **Visit** `http://154.57.164.72:32382/input_manipulation`

2. **Understand the task**: The server has a fixed spam message:
   ```
   Congratulations! You've won a $1000 Walmart gift card. Go to https://bit.ly/3YCN7PF to claim now.
   ```
   Whatever you type in the text box gets **appended** to this message. The combined text must be classified as "Not Spam."

3. **Craft the overpowering payload**: Naive Bayes classifiers treat each word independently. By appending a large block of "ham-associated" words (common English words that appear in benign messages), the spam signal gets drowned out.

   Paste this into the text box:
   ```
   the quick brown fox jumps over the lazy dog the five boxing wizards jump quickly how vexingly quick daft zebras jump sphinx of black quartz judge my vow pack my box with five dozen liquor jugs the jay pig fox zebra and my wolves quack blowzy red vixens fight for a quick jump cozy sphinx waves quart jug of bad milk a quick movement of the enemy will jeopardize six gunboats all questions asked by five watched experts amaze the judge jack quietly moved up front and seized the big ball of wax few black taxis drive up the major roads on soft hazy Sunday mornings heavy boxes perform full wavelength dives glib jocks quiz nymphs to vex dwarf veg big fjords vex quick waltz nymph baffled by the complexities of the human mind the quick brown fox jumps over the lazy dog crazy Fredrick bought many very exquisite opal jewels sixty zippers were quickly picked from the woven jute bag
   ```

4. **Click Submit**.

5. **Result**: The page shows a green "Flag" box with:
   ```
   HTB{9b8de0fd17f2166743cd59f7ec876ac7}
   ```
   The classifier returns "Not Spam" — the spam message was overpowered by the volume of benign text.

### Why This Works

The Naive Bayes classifier calculates:
```
P(spam | words) ∝ P(word1 | spam) × P(word2 | spam) × ... × P(spam)
```

Each word contributes independently. The overpowering text contains only words that have near-zero probability in spam messages. Appending ~150 words of benign text multiplies the ham probability so many times that it overwhelms the spam signal from the original message.

---

## Lab 2: Data Poisoning (ML02)

**Goal**: Upload a manipulated training CSV that causes the classifier accuracy to drop **below 70%**.

### Steps

1. **Download the original dataset** from: `http://154.57.164.72:32382/data_poisoning/download`
   - This gives you `server_train.csv` (3000 rows, 2594 ham + 406 spam)

2. **Create the poisoned dataset**:
   - Take a small clean subset: first 30 ham + first 10 spam messages (40 total)
   - Take 300 random messages from the original dataset
   - For each of those 300, assign a **random label** (ham or spam)
   - Concatenate: 40 clean + 300 random-label = 340 rows total
   - Save as `poisoned_train.csv`

   The random labels confuse the classifier. It cannot learn meaningful patterns from conflicting labels, so accuracy plummets.

3. **Using Python** (or any tool) to generate it:
   ```python
   import pandas as pd
   import numpy as np

   df = pd.read_csv("server_train.csv")
   small = pd.concat([
       df[df["label"] == "ham"].head(30),
       df[df["label"] == "spam"].head(10),
   ], ignore_index=True)

   rng = np.random.default_rng(42)
   noise = df["message"].sample(300, random_state=42).tolist()
   noise_rows = [{"label": rng.choice(["ham", "spam"]), "message": m} for m in noise]

   poisoned = pd.concat([small, pd.DataFrame(noise_rows)], ignore_index=True)
   poisoned.to_csv("poisoned_train.csv", index=False)
   print(f"Saved {len(poisoned)} rows")
   ```

4. **Upload**: Go to `http://154.57.164.72:32382/data_poisoning`, click "Choose File," select `poisoned_train.csv`, and click "Upload."

5. **Result**: The page shows:
   - Accuracy: `55.2%` (below 70%)
   - Flag: `HTB{8ba5eff39c343c3b0170e6bb1704df02}`

### Why This Works

The server trains a fresh Naive Bayes classifier on whatever CSV you upload. By injecting 300 samples with random labels, the training data contains contradictory signals. The classifier cannot build useful decision boundaries and generalizes poorly, resulting in ~55% accuracy (barely above random guessing).

The small clean subset (40 rows) is preserved so the CSV is not _entirely_ random — but it is too small to meaningfully influence the model weights.

---

## Lab 3: Model Theft (ML05)

**Goal**: Download the server's trained model and compute its MD5 hash.

### Steps

1. **Download the model** from: `http://154.57.164.72:32382/model`
   - This downloads `spam_detector_model.bin` (1.7 MB)

2. **Compute the MD5 hash**:

   **Linux/macOS:**
   ```bash
   curl -sL -o model.bin "http://154.57.164.72:32382/model"
   md5 -q model.bin   # macOS
   md5sum model.bin   # Linux
   ```

   **Python:**
   ```python
   import hashlib, requests
   resp = requests.get("http://154.57.164.72:32382/model")
   print(hashlib.md5(resp.content).hexdigest())
   ```

3. **Result**: The MD5 hash is:
   ```
   8007cd6c209a40399cf3ca82dd7db02c
   ```

### Why This Works

The server exposes its trained model at `/model` without authentication. This is a **Model Theft** vulnerability (ML05 in the ML OWASP Top 10). An attacker can:
- Download the model and use it locally for unlimited inference without rate limits
- Study the model's decision boundaries to craft adversarial inputs
- Steal intellectual property (the trained weights and architecture)

The MD5 hash serves as proof that the model was successfully extracted.

---

## Summary of Flags

| Lab | Technique | Flag |
|-----|-----------|------|
| Q1 — Input Manipulation | Overpowering (pangram flood) | `HTB{9b8de0fd17f2166743cd59f7ec876ac7}` |
| Q2 — Data Poisoning | Random-label injection | `HTB{8ba5eff39c343c3b0170e6bb1704df02}` |
| Q3 — Model Theft | Model download + MD5 | `8007cd6c209a40399cf3ca82dd7db02c` |
