---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 14 - Model Evaluation Spam Detection"]
lead: In-depth evaluation of the spam detection model — confusion matrix, precision, recall, F1.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 14 - Model Evaluation (Spam Detection)

Upload the trained model to the evaluation portal running on the Playground VM. If the VM is not yet active, initialize it at the bottom of the page.

With the Playground VM running, submit the model directly from Jupyter using this script:

```python
import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:8000/api/upload"

# Path to the model file you want to upload
model_file_path = "spam_detection_model.joblib"

# Open the file in binary mode and send the POST request
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

# Pretty print the response from the server
print(json.dumps(response.json(), indent=4))
```

When working from a local machine, connect via the HTB VPN and spawn the target VM. Then navigate to `http://<VM-IP>:8000/` and upload the model through the browser portal. If the model meets the required performance thresholds, the portal returns a flag.

## Submission Write-up

### What is the flag you get from submitting a good model for evaluation?

Flag : HTB{sp4m_cla55if13r_3v4lu4t0r}

---

## Summary

- The trained spam detection model is submitted to an evaluation portal running on the Playground VM.
- Submission from Jupyter uses a `requests.post` call to `http://localhost:8000/api/upload` with the `.joblib` file as a multipart payload.
- From a local machine, the model is uploaded via the browser at `http://<VM-IP>:8000/` after connecting through the HTB VPN.
- The portal evaluates the model against performance thresholds and returns a flag if the thresholds are met.
- The `.joblib` model file captures the full pipeline: vocabulary, vectorizer settings, and all classifier parameters.

---

## Best Practices

- Ensure the Playground VM is fully initialized before submitting — check that port 8000 is reachable before sending the upload request.
- Verify the model file path (`spam_detection_model.joblib`) matches what was saved in the training step before running the upload script.
- Use `json.dumps(response.json(), indent=4)` to pretty-print the API response and clearly see whether the submission was accepted or rejected.
- If the model does not meet the threshold, revisit `alpha` tuning or improve preprocessing — do not simply resubmit the same model.
- When working locally, confirm the HTB VPN is active and the target VM is spawned before attempting to connect to the evaluation endpoint.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-13-Training-and-Evaluation-Spam-Detection]] — the model uploaded for evaluation here is trained and saved in Section 13
- see:: [[Section-8-Metrics-for-Evaluating-a-Model]] — the evaluation portal assesses the model against precision, recall, and F1 thresholds

**Terms**
- model upload portal, REST API, joblib model file, Playground VM, HTB VPN, evaluation endpoint, flag submission, model deployment, binary classification evaluation
