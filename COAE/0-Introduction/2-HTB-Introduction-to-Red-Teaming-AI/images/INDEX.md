---
tags:
  - type/structure
  - structure/index
aliases:
  - Red Teaming AI Images
lead: Maps each image file in the images/ directory to the sections that reference it.
created: 2026-04-28
modified: 2026-04-28
---

# Images — HTB Introduction to Red Teaming AI

All image files live in `images/`. Do not modify image files directly; update this index when sections are added or revised.

| Image file | Description | Sections that reference it |
|---|---|---|
| `diagram_1.png` | Four SAIF component areas (Model, Data, Application, System) of a generative AI system | [[Section-6-Red-Teaming-Generative-AI]] |
| `diagram_2.png` | Data poisoning / backdoor attack flow — malicious data injected into the training pipeline | [[Section-8-Attacking-Data-Components]] |
| `diagram_3.png` | Input manipulation attack — adversarial perturbations at inference time (ML01) | [[Section-2-Attacking-ML-based-Systems]] |
| `diagram_4.png` | Model theft / extraction — surrogate model trained from query-response pairs (ML05) | [[Section-2-Attacking-ML-based-Systems]] |
| `diagram_5.png` | Model component attack surface — weights, biases, and extraction vectors | [[Section-7-Attacking-Model-Components]] |
| `diagram_6.png` | Membership inference attack — probing model behavior to detect training-set membership (ML04) | [[Section-2-Attacking-ML-based-Systems]] |
| `diagram_7.png` | Assessment type comparison — vulnerability assessment, penetration test, red team assessment | [[Section-1-Introduction-to-Red-Teaming-ML-based-Systems]] |
| `saif_riskmap.png` | Google SAIF Risk Map — components, risks, and controls showing introduction, exposure, and mitigation points | [[Section-5-Google-Secure-AI-Framework]] |
