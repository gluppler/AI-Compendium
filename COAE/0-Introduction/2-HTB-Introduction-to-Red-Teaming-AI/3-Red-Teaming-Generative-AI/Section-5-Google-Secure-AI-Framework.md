---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 5 - Google's Secure AI Framework (SAIF)"]
lead: Google's Secure AI Framework (SAIF) — a holistic approach to secure AI development covering data, infrastructure, model, and application areas.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

Google's [Secure AI Framework (SAIF)](https://saif.google/) provides actionable principles for securing the entire AI pipeline, from data collection through model deployment. Where OWASP delivers a targeted, technical vulnerability checklist, SAIF takes a broader view: integrating security and privacy across the full development lifecycle and assigning explicit responsibility to model creators and consumers.

---

## SAIF areas and components

SAIF organizes secure AI development into four areas, each comprising multiple [components](https://saif.google/secure-ai-framework/components):

- `Data`: Covers `data sources`, `data filtering and processing`, and `training data`.
- `Infrastructure`: Covers the hardware, storage, and development platforms. Components include `Model Frameworks and Code`, `Training, Tuning, and Evaluation`, `Data and Model Storage`, and `Model Serving`.
- `Model`: The core area. Comprises the `Model`, `Input Handling`, and `Output Handling` components.
- `Application`: Covers interaction with the deployment. Includes `Applications`, `Agents`, and `Plugins`.

This four-area taxonomy is reused throughout the AI Red Teamer path.

---

## SAIF risks

SAIF defines concrete security [risks](https://saif.google/secure-ai-framework/risks). Many map to entries in the OWASP ML Top 10 or LLM Top 10:

- `Data Poisoning`: Malicious data injected into training data compromises performance or installs backdoors.
- Unauthorized Training Data: Training on unauthorized data creates legal or ethical liability.
- Model Source Tampering: Model weights or source code are altered to degrade performance or introduce backdoors.
- Excessive Data Handling: Data collection or retention exceeds what privacy policies permit, creating legal exposure.
- `Model Exfiltration`: Unauthorized access to the model itself steals intellectual property.
- Model Deployment Tampering: Deployment-stage components are manipulated to compromise or backdoor the model.
- `Denial of ML Service`: Inputs crafted for high resource consumption disrupt the ML service.
- Model Reverse Engineering: Analysis of input-output pairs recreates a close approximation of the model without direct access.
- Insecure Integrated Component: Security vulnerabilities in plugins or integrated software are exploited.
- `Prompt Injection`: Direct or indirect manipulation of model input causes malicious behaviour.
- `Model Evasion`: Slight perturbations to inputs cause incorrect inference results.
- Sensitive Data Disclosure: The model is tricked into revealing sensitive data it has access to.
- Inferred Sensitive Data: The model reconstructs sensitive information from training data or context, even without direct access to that data.
- Insecure Model Output: Unsafe handling of model output introduces injection vulnerabilities.
- `Rogue Actions`: Insufficiently restricted model access is exploited to cause harm.

---

## SAIF controls

SAIF maps mitigations ([controls](https://saif.google/secure-ai-framework/controls)) to each risk and assigns responsibility to either the `model creator` or the `model consumer`. For example, if HackTheBox integrated Google's Gemini model, Google would be the model creator and HackTheBox the model consumer. Selected controls:

- Input Validation and Sanitization: Detect and block or restrict malicious queries.
  - Risk mapping: `Prompt Injection`
  - Implemented by: Model Creators, Model Consumers
- Output Validation and Sanitization: Validate or sanitize model output before the application processes it.
  - Risk mapping: Prompt Injection, Rogue Actions, Sensitive Data Disclosure, Inferred Sensitive Data
  - Implemented by: Model Creators, Model Consumers
- Adversarial Training and Testing: Train the model on adversarial inputs to build resilience.
  - Risk mapping: Model Evasion, Prompt Injection, Sensitive Data Disclosure, Inferred Sensitive Data, Insecure Model Output
  - Implemented by: Model Creators, Model Consumers

---

## SAIF risk map

The [Risk Map](https://saif.google/secure-ai-framework/saif-map) consolidates components, risks, and controls in one view. It distinguishes where a risk is introduced (`risk introduction`), where it can be exploited (`risk exposure`), and where it can be mitigated (`risk mitigation`).

![[saif_riskmap.png]]

---

## Summary

- Google's Secure AI Framework (SAIF) provides a holistic approach to AI security, organizing secure development into four areas: Data, Infrastructure, Model, and Application.
- SAIF defines concrete risks — including data poisoning, model exfiltration, prompt injection, model evasion, and rogue actions — that map to OWASP ML and LLM Top 10 entries.
- SAIF distinguishes between risks introduced upstream (e.g., training data) and risks exploited or mitigated downstream, providing a clearer chain-of-responsibility than OWASP's flat lists.
- Controls in SAIF are assigned to either the model creator or model consumer, making security responsibilities explicit in the supply chain.
- The SAIF Risk Map consolidates components, risks, and controls in a single view, distinguishing risk introduction, exposure, and mitigation points.
- Three key controls — Input Validation, Output Validation, and Adversarial Training — collectively address the broadest set of SAIF risks across both model creators and consumers.

---

## Best Practices

- Use the SAIF Risk Map to trace each identified risk back to where it is introduced, where it is exploitable, and where it can be mitigated — enabling prioritized, targeted control placement.
- Assign ownership of each SAIF control explicitly to either the model creator or model consumer before deployment to prevent accountability gaps.
- Apply adversarial training and testing as a standard pre-deployment control to build model resilience against evasion, prompt injection, and sensitive data disclosure.
- Enforce output validation and sanitization before any LLM-generated content reaches application logic, covering prompt injection, rogue actions, and insecure output risks simultaneously.
- Audit all third-party AI components — pre-trained models, datasets, plugins — against the SAIF supply chain and insecure integrated component risks before integration.
- Scope input validation controls to cover both direct user input and indirect inputs such as documents, web pages, and tool call results consumed by the model at runtime.

---

## Quiz

**Q1:** How does SAIF differ from the OWASP ML/LLM Top 10 in its approach to AI security?
> OWASP provides targeted, technical vulnerability checklists. SAIF takes a broader lifecycle view — integrating security and privacy across the full development pipeline, assigning control responsibilities to model creators or consumers, and mapping where each risk is introduced, exposed, and mitigated.

**Q2:** What are the four component areas SAIF organizes secure AI development into?
> Data (data sources, filtering, training data), Infrastructure (frameworks, training/tuning, storage, serving), Model (model, input handling, output handling), and Application (applications, agents, plugins).

**Q3:** Why is it significant that SAIF assigns controls to either the model creator or model consumer?
> It makes security responsibilities explicit in the AI supply chain. Without this distinction, both parties may assume the other is implementing a control, creating accountability gaps where critical mitigations are never deployed.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-4-Attacking-Text-Generation]] — LLM OWASP Top 10 complements SAIF risks
- see:: [[Section-6-Red-Teaming-Generative-AI]] — applies SAIF component areas in the red team context

**Terms**
- SAIF, Google Secure AI Framework, data area, infrastructure area, model area, application area, SAIF controls, SAIF risk map, input validation, output validation, adversarial training
