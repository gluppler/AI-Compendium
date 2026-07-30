---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 1 - Introduction"]
lead: Introduction to the HTB Applications of AI in InfoSec module — scope, environment, and learning objectives.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Applications of AI in InfoSec, COAE path."
---

# Section 1 - Introduction

This module follows the [Fundamentals of AI](https://academy.hackthebox.com/module/details/290) module and takes a practical approach to applying machine learning techniques. The focus shifts from theory to hands-on work: building and evaluating real models, and working through the full end-to-end AI development workflow, from dataset exploration to training and testing.

Three distinct AI models are built throughout this module:

1. A spam classifier that determines whether an SMS message is spam.
2. A network anomaly detection model that identifies abnormal or potentially malicious network traffic.
3. A malware classifier that operates on byteplots (visual representations of binary data).

Python code blocks appear throughout each section to guide the model-building process step by step. These snippets can be copied into a `Jupyter` notebook and executed in sequence, either in the playground VM or your own environment.

Most models can be trained locally. A minimum of 4 GB RAM and 4 CPU cores is recommended for a workable experience.

**Note:** Throughout this module, all sections marked as **interactive** contain code blocks for you to follow along. Not all interactive sections contain separate exercises.

---

## Summary

- The module follows the Fundamentals of AI course and focuses on hands-on, end-to-end model building.
- Three distinct AI models are built: a spam classifier, a network anomaly detector, and a malware image classifier.
- Python code blocks in each section guide the process step by step, intended for execution inside a Jupyter notebook.
- The module covers the full AI development workflow: dataset exploration, preprocessing, training, and evaluation.
- A minimum of 4 GB RAM and 4 CPU cores is recommended for local training.
- Interactive sections contain code blocks to follow along; not all interactive sections have separate exercises.

---

## Best Practices

- Follow each section's code blocks sequentially in Jupyter to avoid kernel state issues.
- Use the Playground VM for resource-constrained setups, but prefer a local environment for faster iteration.
- Treat each model as a full pipeline — do not skip preprocessing, training, or evaluation stages.
- Use at least 4 GB RAM and 4 CPU cores locally to keep training times workable.
- Read the Fundamentals of AI module first; this module assumes that theoretical background.

---

## Quiz

**Q1:** What three AI models are built throughout this module?
> A spam classifier, a network anomaly detection model, and a malware image classifier (operating on byteplots).

**Q2:** What is the minimum hardware recommended for local training in this module?
> 4 GB of RAM and 4 CPU cores.

**Q3:** What is the primary coding environment used to execute the code snippets in this module?
> A Jupyter notebook, either in the Playground VM or a local environment.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Environment-Setup]] — covers the setup required to run the models introduced here
- see:: [[Section-3-JupyterLab]] — the interactive coding environment used throughout the module
- see:: [[Section-9-Spam-Classification]] — first applied model introduced in this section overview

**Terms**
- machine learning, spam classifier, network anomaly detection, malware classifier, byteplots, Jupyter notebook, end-to-end workflow, Python code blocks, playground VM
