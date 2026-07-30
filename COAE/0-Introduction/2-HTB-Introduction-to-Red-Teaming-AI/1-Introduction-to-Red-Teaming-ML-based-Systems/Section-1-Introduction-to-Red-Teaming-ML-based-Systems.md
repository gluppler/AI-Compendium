---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 1 - Introduction to Red Teaming ML-based Systems"]
lead: Introduction to red teaming ML-based systems — penetration testing vs. red team assessments, ML system components, and the attack surface.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

Assessing ML-based systems requires a thorough understanding of their underlying components and algorithms. The complexity of these systems creates a wide attack surface that does not map cleanly onto traditional IT security models. This section establishes the conceptual foundation before diving into specific attack techniques.

---

## What is red teaming?

The most familiar form of security assessment is a `penetration test` — a focused, time-bounded exercise that identifies and exploits vulnerabilities within a defined scope. Penetration testers follow a structured methodology, combining automated tooling with manual techniques to determine whether vulnerabilities exist and how far they can be exploited. Tests typically target isolated segments or staging instances to avoid disrupting production users.

Two other assessment types sit alongside penetration testing:

![[diagram_7.png]]

Vulnerability assessments are largely automated. They scan infrastructure, applications, and networks to catalog and prioritize known vulnerabilities without attempting exploitation. Tools like `Nessus` and `OpenVAS` are common. See the [Vulnerability Assessment](https://academy.hackthebox.com/module/details/108) module for more detail.

`Red team assessments` are adversarial simulations where a dedicated red team replicates the tactics, techniques, and procedures (TTPs) of real-world threat actors. The goal extends beyond technical exploitation — social engineering, phishing, and physical intrusion are all in scope. Red teams operate stealthily against an active blue team defense, targeting specific objectives such as sensitive data access or control of critical systems. Engagements typically span weeks to months, producing an in-depth view of organizational resilience against sophisticated threats. See the [Introduction to Information Security](https://academy.hackthebox.com/module/details/293) module for further background.

---

## Red teaming ML-based systems

ML-based systems are not adequately assessed by standard penetration tests alone. Their reliance on large datasets, statistical inference, and layered model architectures introduces vulnerability classes that require more time and specialized techniques than a typical pentest scope allows.

These systems consist of multiple interconnected components. Security vulnerabilities frequently emerge at interaction points between components — between the data pipeline and the model, between the model and the serving infrastructure, or between the inference endpoint and downstream consumers. A penetration test with a narrowly defined scope may exclude specific components or interaction points, making certain vulnerabilities undetectable by design.

Red team assessments, with broader scope and longer timelines, are better suited to surfacing these vulnerabilities comprehensively.

---

## Summary

- Penetration tests are scoped, time-bounded exercises that exploit defined vulnerabilities; red team assessments simulate real adversaries across broader TTPs including social engineering and physical intrusion.
- Vulnerability assessments use automated scanning tools (Nessus, OpenVAS) to catalog known weaknesses without exploitation.
- ML-based systems have a wider and more complex attack surface than traditional IT systems because vulnerabilities emerge at component interaction points.
- Interaction boundaries between the data pipeline, model, serving infrastructure, and downstream consumers are high-risk zones not fully covered by standard pentests.
- The layered nature of ML architectures — datasets, statistical inference, and serving components — requires longer timelines and specialized techniques to assess properly.
- Red team assessments, with broader scope and longer engagements, are better suited to comprehensively surface ML-specific vulnerabilities.

---

## Best Practices

- Scope ML red team engagements to cover all component interaction points — not just the model endpoint — to avoid blind spots at data-pipeline and serving-layer boundaries.
- Use red team assessments rather than penetration tests alone when ML systems process sensitive data or make safety-critical decisions.
- Map component boundaries explicitly before the engagement starts to ensure interaction points between data, model, and serving infrastructure are in scope.
- Include social engineering and physical intrusion vectors in ML red team scope, as attackers may target data pipelines or serving hardware through non-technical means.
- Document TTPs tailored to ML attack classes (data poisoning, model inversion) alongside traditional exploitation techniques in the rules of engagement.

---

## Quiz

**Q1:** What is the primary difference between a penetration test and a red team assessment?
> A penetration test is scoped and time-bounded, focusing on identifying and exploiting defined vulnerabilities. A red team assessment simulates real adversaries using broad TTPs including social engineering and physical intrusion over weeks to months, targeting specific organizational objectives.

**Q2:** Why are standard penetration tests insufficient for assessing ML-based systems?
> Standard pentests typically have narrowly defined scopes and short timelines that may exclude specific components or interaction points. ML vulnerabilities often emerge at boundaries between components (data pipeline, model, serving infrastructure), requiring broader scope and specialized techniques to surface.

**Q3:** Where do security vulnerabilities most frequently emerge in ML-based systems?
> At interaction points between components — between the data pipeline and the model, between the model and the serving infrastructure, and between the inference endpoint and downstream consumers.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Attacking-ML-based-Systems]] — ML OWASP Top 10 attack vectors that follow from this foundation

**Terms**
- penetration test, red team assessment, vulnerability assessment, ML attack surface, TTPs
