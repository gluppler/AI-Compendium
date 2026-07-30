---
tags:
  - type/structure
  - structure/index
aliases: ["Images Index"]
lead: Maps each image file to the section and context where it is embedded.
created: 2026-04-28
modified: 2026-04-28
---

# Images Index

Each image in the `images/` directory mapped to its source section and usage context. Image files are not modified — this index is a navigation reference only.

---

| Image File | Section | Context |
|------------|---------|---------|
| `jupyter.pngtext
| Section 2 — Environment Setup | Screenshot of the JupyterLab interface accessible via the Playground VM at
        ```http://<VM-IP>:8888`. |
| `jupyter_dark.png` | Section 3 — JupyterLab | Overview screenshot of the JupyterLab interface in dark mode, shown when introducing the notebook environment. |
| `jupyter_new.png` | Section 3 — JupyterLab | Screenshot showing a newly created Python 3 notebook with one empty code cell, illustrating the "create notebook" step. |
| text
jupyter_hello_output.png

| Section 3 — JupyterLab | Screenshot of text
        ```
print("Hello, JupyterLab!")

executed in a cell with output displayed below it. |
        ```
| text
jupyter_simple_plot.png

| Section 3 — JupyterLab | Screenshot of a scatter plot generated in-notebook using pandas, matplotlib, and numpy as a basic visualization example. |
        ```
| `data_encoding.pngtext
| Section 7 — Data Transformation | Diagram illustrating one-hot encoding: a categorical
colortext
column converted to binary indicator columns
        ```color_red`, `color_green`, `color_blue`. |
| `log_histogram.pngtext
| Section 7 — Data Transformation | Before/after histogram comparison of
bytes_transferredtext
showing the reduction in right skew after applying
        ```np.log1p`. |
| `spam_eval.png` | Section 13 — Training and Evaluation (Spam Detection) | Output screenshot showing predicted labels and spam/ham probabilities for five evaluation messages. |
| `anomaly_test.png` | Section 17 — Training and Evaluation (Network Anomaly Detection) | Confusion matrix heatmap for the random forest model evaluated on the NSL-KDD test set. |
| `dataset_1.png` | Section 20 — The Malware Dataset | Sample grayscale byteplot image from the malimg dataset illustrating the binary-to-pixel encoding. |
| `malware_diagram.png` | Section 20 — The Malware Dataset | Diagram explaining the malware binary-to-grayscale-image conversion process (byte value → pixel intensity). |
| `dataset_2.pngtext
| Section 20 — The Malware Dataset | First sample from the
        ```FakeRean` malware family showing its characteristic visual texture pattern. |
| `dataset_3.pngtext
| Section 20 — The Malware Dataset | Second sample from the
        ```FakeRean` malware family showing the same family's recognizable texture in a different binary. |
| `dataset_4_fixed.png` | Section 20 — The Malware Dataset | Horizontal bar chart of malware class distribution across all 25 families, revealing over- and underrepresented families. |
| `preproc_1_fixed.png` | Section 21 — Preprocessing the Malware Dataset | Raw malware image before preprocessing transforms are applied (original resolution). |
| `preproc_2_fixed.png` | Section 21 — Preprocessing the Malware Dataset | Malware image after resizing to 75×75 and applying ImageNet normalization, as it enters the ResNet50 model. |
| `train_1_fixed.png` | Section 23 — Training and Evaluation (Malware Image Classification) | Plot of training accuracy across 10 epochs, showing steady increase from ~57% to ~96% before leveling off. |

---

# Back Matter

**Source**
- based_on:: [[README]]

**References**
- see:: [[references/INDEX]] — companion index of all external URLs referenced in the module
- see:: [[works/INDEX]] — companion index of extracted code scripts

**Terms**
- image assets, screenshots, plots, diagrams, visualization
