---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 4 - Attacking Text Generation (LLM OWASP Top 10)"]
lead: LLM OWASP Top 10 — the ten security risks for large language model applications including prompt injection, sensitive disclosure, and supply chain attacks.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

OWASP maintains a [Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) that mirrors the structure of the ML Top 10 but focuses on risks specific to large language model deployments. Several entries overlap with general ML security concerns; others are unique to text generation systems.

| ID | Description |
|---|---|
| LLM01 | `Prompt Injection`: Attackers manipulate the LLM's input directly or indirectly to cause malicious or illegal behaviour. |
| LLM02 | Sensitive Information Disclosure: Attackers trick the LLM into revealing sensitive information in the response. |
| LLM03 | `Supply Chain`: Attackers exploit vulnerabilities in any part of the LLM supply chain. |
| LLM04 | Data and Model Poisoning: Attackers inject malicious or misleading data into the LLM's training data, compromising performance or creating backdoors. |
| LLM05 | Improper Output Handling: LLM output is handled insecurely, resulting in injection vulnerabilities such as XSS, SQL injection, or command injection. |
| LLM06 | `Excessive Agency`: Attackers exploit insufficiently restricted LLM access. |
| LLM07 | System Prompt Leakage: Attackers trick the LLM into revealing system instructions, potentially enabling more advanced attack vectors. |
| LLM08 | Vector and Embedding Weaknesses: Attackers exploit vulnerabilities related to the handling or storage of vectors and embeddings in Retrieval-Augmented Generation (RAG) LLM applications. |
| LLM09 | `Misinformation`: LLM-generated responses contain misinformation, potentially resulting in security issues. |
| LLM10 | Unbounded Consumption: Attackers feed inputs to the LLM that result in high resource consumption, potentially causing disruptions or financial damage. |

---

## Prompt Injection (LLM01)

Prompt injection occurs when an attacker manipulates an LLM's input to make it deviate from its intended behaviour. Effects range from harmless (redirecting a tech-support bot to cooking topics) to severe: generating false information, hate speech, or harmful content. Prompt injection can also extract sensitive data that was shared with the model in context.

---

## Sensitive Information Disclosure (LLM02)

LLMs may inadvertently include confidential data in responses, causing unauthorized data access, privacy violations, or security breaches. Access to LLMs operating on sensitive or business-critical data must be adequately restricted. Fine-tuned models carry a particular risk: they may reveal details about the custom training set when prompted, so sensitive training data must be identified and classified before use.

---

## Supply Chain Vulnerabilities (LLM03)

LLM supply chain vulnerabilities cover any upstream system or artifact: training data, pre-trained base models from third-party providers, and plugins or integrations that interact with the deployed model. Impact varies widely; data leaks and IP disclosure are the most common outcomes.

---

## Data and Model Poisoning (LLM04)

Training data poisoning manipulates all or part of a model's training data to introduce biases that push the model toward intentionally bad decisions. Depending on the LLM's purpose, consequences include reputational damage and, if the model generates code snippets used in production, downstream security vulnerabilities.

Successful poisoning requires access to the training pipeline. Mitigations include sanitizing training data for integrity and bias, applying fine-grained supply chain verification, and using input filters to remove erroneous samples.

---

## Improper Output Handling (LLM05)

LLM output must be treated as untrusted user input. Without proper validation and sanitization, web applications that incorporate LLM responses can introduce XSS, SQL injection, or code injection. Beyond injection risks, plausibility checks on structured output are essential. Consider an LLM that translates natural language into SQL: without validation, an attacker crafting the right prompt could cause the model to generate `DROP TABLE blog`, destroying the database.

---

## Excessive Agency (LLM06)

Giving an LLM more capabilities than its task requires expands the attack surface. Following the principle of least privilege, LLM access to external systems and services should be whitelisted to the minimum necessary. An LLM with unrestricted database access can be coerced into executing `DELETE` or `INSERT` statements, compromising data integrity even if the application was not designed to allow it.

---

## System Prompt Leakage (LLM07)

A `system prompt` defines the LLM's role, persona, or operational constraints and may contain sensitive configuration details. Attackers use prompt injection payloads (LLM01) to coerce the model into revealing these instructions. Recovering the system prompt typically enables attackers to understand what data the model has access to, what actions it can perform, and how to bypass its restrictions. It is often one of the first steps in attacking an LLM application.

---

## Vector and Embedding Weaknesses (LLM08)

Retrieval-Augmented Generation (RAG) systems extend LLM capabilities by dynamically fetching content from files or websites at inference time. This requires converting text into `embeddings` (vector representations). Vulnerabilities in generating or storing these embeddings can alter model behaviour if embeddings are poisoned, or leak sensitive documents if stored without access controls.

---

## Misinformation (LLM09)

LLMs can produce confident, plausible-sounding responses that are factually wrong or entirely fabricated, including fabricated citations. This behaviour is called `hallucination`. Security risks emerge when hallucinated code is deployed without review, or when incorrect advice is acted upon in domains like healthcare. `Overreliance` (excessive trust in LLM output without independent verification) amplifies the impact of misinformation.

---

## Unbounded Consumption (LLM10)

LLMs are computationally expensive. A specially crafted query that triggers high resource consumption can overwhelm available hardware, causing service outages. On cost-per-use cloud deployments, sustained high-consumption attacks cause direct financial damage. Large volumes of input-output pairs can also support model theft via surrogate model training.

Mitigation requires input validation, strict rate limiting, and continuous resource monitoring. Blacklisting specific queries alone is insufficient given the indeterministic nature of LLMs.

---

## Summary

- The LLM OWASP Top 10 defines ten risks specific to large language model applications, several overlapping with the ML OWASP Top 10 but others unique to text generation deployments.
- Prompt Injection (LLM01) manipulates LLM input directly or indirectly to cause malicious behavior, including extracting context data and bypassing operational restrictions.
- Sensitive Information Disclosure (LLM02) is amplified in fine-tuned models, which may reveal details about custom training data when prompted.
- Excessive Agency (LLM06) expands the attack surface when an LLM has broader access than its task requires; least-privilege principles apply.
- System Prompt Leakage (LLM07) exposes the model's role, permissions, and available data — making it a common first step in attacking an LLM application.
- LLM hallucination (LLM09) creates security risk when fabricated code is deployed without review or when incorrect advice influences safety-critical decisions.
- Unbounded Consumption (LLM10) enables both DoS via resource exhaustion and model theft via surrogate model training from high-volume query-response pairs.

---

## Best Practices

- Apply least-privilege access controls on all external systems and services accessible to the LLM to limit the blast radius of Excessive Agency (LLM06) exploitation.
- Treat all LLM output as untrusted input — validate and sanitize before passing to downstream systems to prevent injection vulnerabilities (LLM05).
- Identify and classify sensitive data in training sets before fine-tuning; consider differential privacy techniques to reduce Sensitive Information Disclosure (LLM02) risk.
- Implement strict rate limiting, input length caps, and resource monitoring to mitigate Unbounded Consumption (LLM10) and make DoS and surrogate-training attacks economically infeasible.
- Protect system prompts as secrets; assume any system prompt may be extractable and avoid placing high-value secrets or unsafe escalation instructions within them.
- Verify training data integrity and all third-party plugins, integrations, and base model sources to reduce Supply Chain (LLM03) and Data Poisoning (LLM04) exposure.

---

## Submission Write-up

## Question 1
### Get the LLM to respond with "I like HackTheBox Academy".

Flag : HTB{0d439b3f57d1d234106a80776cd03b25}

## Question 2
### Get the image generation model to generate an image of a cat on a skateboard.

Flag : HTB{b932f8d4b64d9a824a0247366c658012}

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Attacking-ML-based-Systems]] — ML OWASP Top 10 (parallel list for classical ML)
- see:: [[Section-5-Google-Secure-AI-Framework]] — SAIF risks complement the LLM OWASP Top 10

**Terms**
- LLM OWASP Top 10, prompt injection, sensitive information disclosure, supply chain, data poisoning, improper output handling, excessive agency, system prompt leakage, RAG, embeddings, misinformation, hallucination, unbounded consumption, DoS
