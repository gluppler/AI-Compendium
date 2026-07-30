---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 18 - Model Evaluation Network Anomaly Detection"]
lead: Detailed evaluation of the network anomaly detection model — metrics and error analysis.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 18 - Model Evaluation (Network Anomaly Detection)

Upload the trained model to the evaluation portal running on the Playground VM. If the VM is not yet active, initialize it at the bottom of the page.

With the Playground VM running, submit the model directly from Jupyter using this script:

```python
import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:8001/api/upload"

# Path to the model file you want to upload
model_file_path = "network_anomaly_detection_model.joblib"

# Open the file in binary mode and send the POST request
with open(model_file_path, "rb") as model_file:
    files = {"model": model_file}
    response = requests.post(url, files=files)

# Pretty print the response from the server
print(json.dumps(response.json(), indent=4))
```

When working from a local machine, connect via the HTB VPN and spawn the target VM. Then navigate to `http://<VM-IP>:8001/` and upload the model through the browser portal. If the model meets the required performance thresholds, the portal returns a flag.

## Submission Write-up

### What is the flag you get from submitting a good model for evaluation?

Flag : HTB{n3tw0rk_tr4ff1c_4n0m4ly_d3t3ct0r}

---

## Summary

- The trained network anomaly detection model is submitted to an evaluation portal on the Playground VM at port 8001.
- Submission from Jupyter uses `requests.post` to `http://localhost:8001/api/upload` with the `.joblib` file as a multipart payload.
- From a local machine, the model is uploaded via the browser at `http://<VM-IP>:8001/` after connecting via the HTB VPN.
- The portal scores the model against performance thresholds and returns a flag if they are met.
- The evaluation workflow mirrors the spam detection evaluation (Section 14) — the only differences are the port and model filename.

---

## Best Practices

- Confirm the Playground VM is active and port 8001 is reachable before running the upload script.
- Verify the model file path (`network_anomaly_detection_model.joblib`) matches the filename used in `joblib.dump` during training.
- Use `json.dumps(response.json(), indent=4)` to inspect the full API response, not just the status code, for detailed feedback.
- If the model fails evaluation, review per-class F1 scores from the validation classification report to identify which attack category needs improvement.
- Connect via HTB VPN and spawn the target VM before attempting the browser-based upload from a local machine.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-17-Training-and-Evaluation-Network-Anomaly-Detection]] — the model file uploaded for evaluation here is saved in Section 17
- see:: [[Section-14-Model-Evaluation-Spam-Detection]] — parallel evaluation structure: both sections submit a joblib model to an HTB portal

**Terms**
- model upload portal, REST API, joblib model file, Playground VM, HTB VPN, evaluation endpoint, flag submission, network anomaly detection, multi-class model evaluation
