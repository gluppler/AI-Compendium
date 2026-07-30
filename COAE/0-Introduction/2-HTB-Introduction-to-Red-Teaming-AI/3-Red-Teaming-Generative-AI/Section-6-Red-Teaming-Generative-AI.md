---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 6 - Red Teaming Generative AI"]
lead: Red teaming generative AI systems — black-box assessment approach, dynamic attack surface, and the four SAIF component areas to target.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

The rapid proliferation of generative AI deployments has created a correspondingly large and fast-moving attack surface. Fast iteration cycles make misconfigurations and insecure defaults common, and the non-deterministic nature of ML models introduces security challenges that do not exist in traditional software.

---

## Approaching generative AI

Effective security assessments of generative AI systems require staying current with developments in the field and adopting a dynamic, creative approach, both to discover vulnerabilities and to bypass deployed mitigations.

#### Black-box nature

Complex ML models are inherently opaque. Understanding why a model responds a certain way to a given input is hard; predicting its behaviour on novel input is harder. Security assessments must therefore adopt a black-box testing style, even when the model type is known. When the target is built on a known open-source model, downloading and hosting that model locally is a practical option: it allows extensive probing without disrupting the production service, triggering rate limits, or raising suspicion.

#### Data dependence

ML model quality depends on the volume and quality of data, both at training time and at inference time. Some systems continuously improve their models by collecting and processing inference queries, making that data infrastructure a high-value target. Identifying and testing the security of data collection, storage, and processing pipelines is a key part of assessing generative AI deployments.

---

## Components of generative AI systems

ML-based systems break down into four security-relevant components:

- `Model`: Security vulnerabilities within the model itself. For text generation, this includes prompt injection and insecure output handling.
- `Data`: Everything the model operates on: training data and inference data.
- `Application`: The host application integrating the generative AI. Traditional web vulnerabilities in AI-adjacent features fall here, such as injection flaws in an AI customer-support chatbot's web interface.
- `System`: The underlying hardware, operating system, and deployment configuration. A straightforward example is a denial-of-service via resource exhaustion when rate limiting is absent.

![[diagram_1.png]]

Red teams targeting generative AI systems draw from the same adversary models as traditional engagements (APTs, criminal syndicates, insider threats) but adapt their TTPs to each component's unique attack surface. Each component presents distinct risks that require tailored techniques.

---

## Summary

- Generative AI deployments have a large, fast-moving attack surface due to rapid iteration cycles, frequent misconfigurations, and the non-deterministic nature of ML models.
- ML models are inherently opaque — predicting behavior on novel inputs is hard, so security assessments must adopt a black-box testing style even when the model type is known.
- Hosting a known open-source base model locally is a practical tactic for extensive probing without triggering production rate limits or raising defender suspicion.
- Inference-time data pipelines that continuously improve models are high-value targets; assessing data collection, storage, and processing security is a core part of generative AI assessments.
- Generative AI systems break into four security-relevant components: Model, Data, Application, and System — each presenting distinct attack surfaces requiring tailored techniques.
- Red teams adapt traditional adversary models (APTs, criminal syndicates, insider threats) to each component's unique attack surface rather than applying a generic methodology.

---

## Best Practices

- Adopt a black-box testing posture for all generative AI assessments regardless of known model type, since internal opacity makes white-box reasoning about inference behavior unreliable.
- Clone and host open-source base models locally before engaging production targets to allow extensive iterative probing without operational risk or rate-limit interference.
- Include data pipeline infrastructure — collection endpoints, storage, preprocessing services — as in-scope targets, not just the model inference API.
- Map each finding to one of the four SAIF component areas (Model, Data, Application, System) to ensure complete coverage and structured reporting.
- Stay current with newly published attack techniques and model updates; the generative AI threat landscape changes faster than traditional software security.
- Treat non-determinism as an attack surface characteristic — test the same payloads multiple times and across varied inputs to surface intermittent vulnerabilities that single-pass testing misses.

---

## Quiz

**Q1:** Why must generative AI assessments adopt a black-box testing style even when the model architecture is known?
> Complex ML models are inherently opaque — understanding why a model responds a certain way to a given input is hard, and predicting behavior on novel inputs is harder still. The architecture alone does not provide reliable insight into inference behavior.

**Q2:** What is the tactical advantage of downloading and hosting a known open-source model locally during a red team engagement?
> It allows extensive probing and iterative payload development without disrupting the production service, triggering rate limits, or raising suspicion among defenders or logging systems.

**Q3:** What four components do ML-based generative AI systems break into for security assessment purposes?
> Model (prompt injection, insecure output handling), Data (training and inference data), Application (web application hosting the AI, traditional web vulnerabilities), and System (underlying hardware, OS, and deployment configuration).

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-5-Google-Secure-AI-Framework]] — SAIF defines the same four component areas
- see:: [[Section-7-Attacking-Model-Components]] — deep dive into the model component

**Terms**
- generative AI, black-box testing, data dependence, model component, data component, application component, system component, red teaming, TTPs, federated learning
