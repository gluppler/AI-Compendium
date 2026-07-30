---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 8 - Attacking Data Components"]
lead: Attacking the data component of generative AI systems — data poisoning, training data manipulation, and PII leakage risks.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

The data component covers all data the model operates on: training data and inference data. ML models are inherently data-dependent, so even minor disruptions to this component have outsized consequences on model quality and behaviour. When the data contains personally identifiable information (PII), unauthorized access also creates legal exposure, including GDPR liability.

---

## Risks

#### Data poisoning

`Data poisoning` is the data-side counterpart to model poisoning. Instead of manipulating parameters directly, attackers corrupt the training data that shapes those parameters. Impacts mirror those of model poisoning:

- Generation of misleading output
- Generation of biased output
- Generation of harmful content

Attackers can embed specific triggers in poisoned data, causing the model to produce adversarial outputs only when prompted with a matching input. This is a `backdoor attack`. Such manipulations can be used to degrade model quality, erode trust, or enable targeted misuse such as misinformation campaigns.

![[diagram_2.png]]

#### Data exfiltration

The large data volumes required to train and operate ML models create persistent exfiltration risk. Stolen training data can contain unique, curated datasets assembled over years, valuable to competitors and attackers alike. Exfiltrated training data also enables reverse-engineering the model, crafting targeted adversarial inputs, or identifying exploitable patterns in the original model's behaviour. Depending on content, a leak can trigger direct financial harm, regulatory penalties, and long-term reputational damage.

---

## Tactics, techniques, and procedures (TTPs)

#### Training data manipulation

Injecting malicious data requires knowing which data the model trains on and gaining write access to that pipeline, a challenging prerequisite in many environments. However, some architectures make it feasible. In `federated learning` systems, where multiple parties contribute updates to a shared global model, a malicious participant can inject poisoned updates during their contribution round, skewing the model without raising obvious suspicion.

#### Data theft

Attackers stealing training data combine traditional and ML-specific techniques:

- Exploiting poorly configured cloud storage
- Exploiting insufficient encryption at rest or in transit
- Compromising insecure data pipelines
- Exploiting vulnerable APIs

`Supply chain attacks` extend this surface further: compromising a third-party data vendor or provider allows attackers to access the dataset before it reaches the target organization.

Insider threats are particularly hard to detect. Employees and contractors with legitimate data access may be compromised via phishing or social engineering, or may deliberately exfiltrate data for financial gain or industrial espionage. Because they already hold authorized access, they can steal data with minimal technical sophistication.

---

## Summary

- The data component covers all data the model operates on — training data and inference data — and disruptions to it have outsized consequences on model quality and behavior.
- Data poisoning corrupts training data to introduce misleading output, bias, or harmful content; trigger-based backdoor attacks cause adversarial outputs only when a matching input is present.
- Data exfiltration risks are persistent due to the large, uniquely curated datasets ML systems require; stolen data enables competitor intelligence, model reverse-engineering, and adversarial input crafting.
- Training data manipulation in federated learning environments is feasible for malicious participants who inject poisoned updates during their contribution round.
- Data theft leverages both traditional techniques (cloud misconfiguration, insufficient encryption, vulnerable APIs) and ML-specific supply chain attacks against third-party data vendors.
- Insider threats are particularly dangerous as authorized employees or contractors can exfiltrate training data with minimal technical sophistication using their existing access.
- PII in training data creates GDPR and legal exposure beyond pure security risk.

---

## Best Practices

- Apply strict write-access controls and integrity verification to training data pipelines; treat unauthorized writes to training data with the same severity as unauthorized writes to model weights.
- Encrypt training data at rest and in transit, and enforce access controls on cloud storage buckets holding datasets — misconfigurations here are a leading data exfiltration vector.
- Audit federated learning contribution mechanisms to detect poisoned model updates from malicious participants; use Byzantine-robust aggregation algorithms where feasible.
- Identify and minimize PII in training datasets before use; classify data sensitivity levels and apply data minimization practices to reduce legal exposure from exfiltration events.
- Extend supply chain review to third-party data vendors and labeling services — compromise at this level can introduce backdoors before data reaches the organization.
- Implement user behavior analytics (UBA) on data access patterns to detect insider threat exfiltration that bypasses technical controls due to legitimate access credentials.

---

## Quiz

**Q1:** What is a backdoor attack in the context of data poisoning, and how does it differ from general data poisoning?
> General data poisoning degrades model quality broadly. A backdoor attack embeds a specific trigger in poisoned training samples so the model produces adversarial outputs only when that trigger appears in the input — behavior that is invisible during normal operation and activates on demand.

**Q2:** Why are federated learning systems particularly susceptible to training data manipulation?
> In federated learning, multiple parties contribute model updates to a shared global model. A malicious participant can inject poisoned updates during their contribution round, skewing the global model without needing direct access to the central training pipeline or other participants' data.

**Q3:** Why are insider threats harder to detect than external data theft, even when the organization has strong perimeter security?
> Insiders already hold authorized access, so their data access doesn't trigger external-intrusion detections. They can exfiltrate data with minimal technical sophistication using existing credentials, making their activity indistinguishable from normal operations without behavioral analytics.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-7-Attacking-Model-Components]] — model poisoning as the direct-parameter counterpart to data poisoning
- see:: [[Section-9-Attacking-Application-Components]] — application-layer TTPs that can enable data exfiltration

**Terms**
- data component, data poisoning, backdoor attack, PII, training data manipulation, federated learning, supply chain attacks, insider threat, data exfiltration
