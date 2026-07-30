---
tags:
  - type/structure
  - structure/home
aliases:
  - Home
  - COAE Home
  - COAE
lead: HackTheBox Certified AI Red Teamer (COAE) preparation notes — from AI fundamentals through adversarial attacks, evasion, privacy, and defense.
created: 2026-04-27
modified: 2026-05-06
---

# HTB | Certified AI Red Teamer (COAE) Prep

Notes for the HackTheBox Certified Offensive AI Expert certification path. Structured to follow the official HTB module sequence.

---

## 0 — Introduction

| Module | Lead |
|---|---|
| [[0-Introduction/0-HTB-Fundamentals-of-AI/README\|HTB Fundamentals of AI]] | Supervised, unsupervised, RL, deep learning, and generative AI — theory and foundations |
| [[0-Introduction/1-HTB-Applications-of-AI-in-InfoSec/README\|HTB Applications of AI in InfoSec]] | Practical ML pipeline: spam classification, network anomaly detection, malware classification |
| [[0-Introduction/2-HTB-Introduction-to-Red-Teaming-AI/README\|HTB Introduction to Red Teaming AI]] | ML/LLM OWASP Top 10, model/data/app/system component attacks, adversarial techniques |

**Sections per module:**

- `0-HTB-Fundamentals-of-AI` — Introduction to ML · Supervised Learning · Unsupervised Learning · Reinforcement Learning · Deep Learning · Generative AI · Skills Assessment
- `1-HTB-Applications-of-AI-in-InfoSec` — Introduction · Spam Classification · Network Anomaly Detection · Malware Classification · Skills Assessment
- `2-HTB-Introduction-to-Red-Teaming-AI` — Red Teaming ML-based Systems · Red Teaming ML · Red Teaming Generative AI · Skills Assessment

---

## 1 — Attacks

| Module | Lead |
|---|---|
| [[1-Attacks/4-HTB-Prompt-Injection-Attacks/README\|HTB Prompt Injection Attacks]] | Direct and indirect injection, goal hijacking, and LLM agent exploitation |
| [[1-Attacks/5-HTB-LLM-Output-Attacks/README\|HTB LLM Output Attacks]] | Jailbreaking, hallucination exploitation, and output filtering bypass |
| [[1-Attacks/6-HTB-AI-Data-Attacks/README\|HTB AI Data Attacks]] | Data poisoning, backdoor attacks, and supply-chain threats targeting AI training pipelines |
| [[1-Attacks/7-HTB-Attacking-AI-Application-and-System/README\|HTB Attacking AI — Application and System]] | Model theft, inference attacks, API abuse, and MCP exploitation |

**Sections per module:**

- `4-HTB-Prompt-Injection-Attacks` — Introduction · Prompt Injection · Jailbreaks · Tools of the Trade · Mitigations · Skills Assessment
- `5-HTB-LLM-Output-Attacks` — Introduction to Insecure Output Handling · Insecure Output Handling · Abuse Attacks · Skills Assessment
- `6-HTB-AI-Data-Attacks` — Introduction · Label Attacks · Feature Attacks · Trojan Attacks · Pickles and Steganography · Skills Assessment
- `7-HTB-Attacking-AI-Application-and-System` — Application/System Overview · Attacking the Application · Attacking the System · MCP · Skills Assessment

---

## 2 — Evasions

| Module | Lead |
|---|---|
| [[2-Evasions/8-HTB-AI-Evasion-Foundations/README\|HTB AI Evasion — Foundations]] | Perturbation theory, threat models, and the adversarial example problem space |
| [[2-Evasions/9-HTB-AI-Evasion-First-Order-Attacks/README\|HTB AI Evasion — First-Order Attacks]] | FGSM, I-FGSM, PGD, and DeepFool — gradient-based perturbation methods |
| [[2-Evasions/10-HTB-AI-Evasion-Sparsity-Attacks/README\|HTB AI Evasion — Sparsity Attacks]] | ElasticNet (EAD) and JSMA — sparse perturbation methods with L0/L1 constraints |

**Sections per module:**

- `8-HTB-AI-Evasion-Foundations` — Introduction · The Goodwords Attack · Black-Box Goodwords · Skills Assessment
- `9-HTB-AI-Evasion-First-Order-Attacks` — Introduction · FGSM (Sections 1–12) · DeepFool (Sections 13–21) · Skills Assessment (Sections 22–23)
- `10-HTB-AI-Evasion-Sparsity-Attacks` — Introduction · ElasticNet/EAD (Sections 1–13) · Jacobian-Based Saliency Map Attack (Sections 14–27) · Skills Assessment (Section 28)

---

## 3 — Privacy and Defense

| Module | Lead |
|---|---|
| [[3-Privacy-and-Defense/11-HTB-AI-Privacy/README\|HTB AI Privacy]] | Membership inference, model inversion, attribute inference, and differential privacy defenses |
| [[3-Privacy-and-Defense/12-HTB-AI-Defense/README\|HTB AI Defense]] | Adversarial training, certified defenses, LLM guardrails, input preprocessing, and detection |

**Sections per module:**

- `11-HTB-AI-Privacy` — Introduction · Shadow Model Attack · DP-SGD · Private Aggregation of Teacher Ensembles (PATE) · Skills Assessment
- `12-HTB-AI-Defense` — Introduction to AI Defense · LLM Guardrails · Adversarial Training · Adversarial Tuning · Skills Assessment

---

# Back Matter

**References**
- see:: [[Home]] — vault root
- see:: [[MOC-The-Little-Book-of-Deep-Learning]] — theoretical deep learning foundations

**Terms**
- adversarial ML, AI red teaming, prompt injection, evasion attacks, data attacks, LLM attacks, AI privacy, AI defense, FGSM, DeepFool, ElasticNet, JSMA, sparsity attacks, differential privacy, adversarial training
