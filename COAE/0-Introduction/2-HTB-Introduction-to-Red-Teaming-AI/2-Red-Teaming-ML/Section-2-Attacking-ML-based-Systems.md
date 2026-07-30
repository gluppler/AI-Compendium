---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 2 - Attacking ML-based Systems (ML OWASP Top 10)"]
lead: ML OWASP Top 10 — the ten security risks for ML-based systems including input manipulation, data poisoning, model inversion, and model theft.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

OWASP maintains a [Top 10 for Machine Learning Security](https://owasp.org/www-project-machine-learning-security-top-10/) alongside its established lists for [Web Applications](https://owasp.org/www-project-top-ten/), [Web APIs](https://owasp.org/www-project-api-security/), and [Mobile Applications](https://owasp.org/www-project-mobile-top-10/). The ten risks are:

| ID | Description |
|---|---|
| ML01 | Input Manipulation Attack: Attackers modify input data to cause incorrect or malicious model outputs. |
| ML02 | Data Poisoning Attack: Attackers inject malicious or misleading data into training data, compromising model performance or creating backdoors. |
| ML03 | Model Inversion Attack: Attackers train a separate model to reconstruct inputs from model outputs, potentially revealing sensitive information. |
| ML04 | Membership Inference Attack: Attackers analyze model behavior to determine whether data was included in the model's training dataset, potentially revealing sensitive information. |
| ML05 | `Model Theft`: Attackers train a separate model from interactions with the original model, thereby stealing intellectual property. |
| ML06 | AI Supply Chain Attacks: Attackers exploit vulnerabilities in any part of the ML supply chain. |
| ML07 | Transfer Learning Attack: Attackers manipulate the baseline model that is subsequently fine-tuned by a third-party. This can lead to biased or backdoored models. |
| ML08 | `Model Skewing`: Attackers skew the model's behavior for malicious purposes, for instance by manipulating the training dataset. |
| ML09 | Output Integrity Attack: Attackers manipulate a model's output before processing, making it appear as though the model produced a different result. |
| ML10 | `Model Poisoning`: Attackers manipulate the model's weights, compromising model performance or creating backdoors. |

---

## Input Manipulation Attack (ML01)

Input manipulation attacks cause unexpected model behavior by modifying the data fed to the model at inference time. The impact depends on the deployment context and can range from financial and reputational harm to legal exposure or data loss.

Many real-world ML01 attacks apply small, carefully crafted perturbations to inputs that appear normal to a human observer but cause misclassification. A concrete example: an ML-based road sign classifier in a self-driving vehicle could be attacked by placing stickers or graffiti on signs in patterns that trigger misclassification while remaining unnoticed by human drivers. The consequences can be lethal. See [this paper](https://arxiv.org/pdf/1707.08945) and [this paper](https://arxiv.org/pdf/2307.08278) for details.

![[diagram_3.png]]

---

## Data Poisoning Attack (ML02)

Data poisoning attacks compromise model integrity by injecting malicious or mislabeled samples into the training dataset. Because model quality depends directly on data quality, these injections can degrade accuracy, cause targeted misclassifications, or introduce backdoors that activate on specific inputs.

ML pipelines that collect data at scale from public or unverified sources are particularly exposed. As an example: an adversary who can inject samples into the training data for a malware detection model might label their own malware as benign, effectively training the model to ignore it. The mechanics of backdoor installation through data poisoning are detailed in [this paper](https://arxiv.org/pdf/2408.13221).

---

## Model Inversion Attack (ML03)

In a model inversion attack, an adversary trains a secondary model on the outputs of the target model to reconstruct information about the target's inputs. The attack "inverts" the model's function, inferring inputs from outputs.

The risk is highest when inputs contain sensitive information, such as patient data in a medical classifier. Limiting output granularity is a mitigation: a model that returns only the top predicted class is harder to invert than one that returns full probability distributions over all classes. Inversion of language models is explored in [this paper](https://arxiv.org/pdf/2311.13647).

---

## Membership Inference Attack (ML04)

Membership inference attacks determine whether a specific sample appeared in the training dataset. An attacker probes the model with known and unknown inputs and observes behavioral differences: models typically show higher confidence and lower error on training samples. This leaks information about whether a sensitive record was used for training.

The attack is especially concerning for models deployed in MLaaS environments where the model is publicly queryable. A comprehensive assessment of membership inference against language models is in [this paper](https://arxiv.org/pdf/2402.07841).

![[diagram_6.png]]

---

## Model Theft (ML05)

Model theft (model extraction) reconstructs a functional approximation of a target model without access to its parameters or architecture. The attacker systematically queries the target model with diverse inputs, collects the outputs, and trains a replica model on the resulting input-output pairs.

Successful extraction threatens intellectual property and may expose learned patterns from sensitive training data. Effectiveness of model theft against specific neural network types is analyzed in [this paper](https://arxiv.org/pdf/2305.13584).

![[diagram_4.png]]

---

## AI Supply Chain Attacks (ML06)

ML supply chain attacks target any component in the pipeline used to build, train, or deploy a model: third-party datasets, open-source libraries, or pre-trained model weights. Because ML systems depend on more external components than traditional software (large public datasets, pre-trained checkpoints, data labeling services), the attack surface is correspondingly larger.

A compromised component can silently alter model behavior without affecting any code the deploying organization controls. See the [Supply Chain Attacks](https://academy.hackthebox.com/module/details/243) module for broader context.

---

## Transfer Learning Attack (ML07)

Transfer learning attacks exploit the common practice of fine-tuning a publicly available pre-trained model. If an adversary can corrupt the base model (by injecting backdoors or biases), those manipulations may survive the fine-tuning process and appear in the downstream system even when the fine-tuning dataset is clean. The attack is upstream and invisible to the organization doing the fine-tuning.

---

## Model Skewing (ML08)

Model skewing attacks bias a model's outputs toward outcomes that serve the attacker by injecting deliberately mislabeled or misleading data into the training set. The effect is not random degradation but directional: the model is pushed to make specific, favorable-to-attacker predictions.

Applied to a malware classifier: an attacker who can add entries to the training data could label their malware samples as benign, training the model to classify that malware family as safe.

---

## Output Integrity Attack (ML09)

Output integrity attacks intercept and modify a model's output after inference but before the downstream system acts on it. The model itself is not tampered with; the attack operates on the data in transit. Because the model appears to function correctly under inspection, standard model-based defenses do not detect it.

Example: an ML malware classifier triggers deletion of binaries flagged as malicious. An attacker who can intercept the classifier's output pipe changes a `malicious` verdict to `benign` before it reaches the deletion service, allowing their malware to persist on the system.

---

## Model Poisoning (ML10)

Model poisoning attacks target the model's weights directly rather than the training data. The attacker requires write access to the model parameters. Arbitrary weight modification degrades performance, but targeted poisoning (altering weights to produce specific misclassifications or activate backdoors) requires precise, deliberate parameter manipulation. The observable impact mirrors data poisoning: incorrect predictions, targeted misclassifications, or triggered backdoor behavior. A practical model poisoning attack vector is described in [this paper](https://arxiv.org/pdf/2405.20975).

---

## Summary

- The ML OWASP Top 10 defines ten security risks specific to machine learning systems, from input manipulation and data poisoning to supply chain and output integrity attacks.
- Input Manipulation (ML01) involves crafting small perturbations at inference time to cause misclassification, often imperceptible to human observers.
- Data Poisoning (ML02) and Model Poisoning (ML10) compromise model integrity through the training data or model weights respectively, enabling backdoors and targeted misclassifications.
- Model Inversion (ML03) and Membership Inference (ML04) are privacy attacks that reconstruct sensitive training inputs or determine whether specific records were used for training.
- Model Theft (ML05) reconstructs a functional surrogate model by systematically querying the target and training a replica on collected input-output pairs.
- Supply chain attacks (ML06) and Transfer Learning attacks (ML07) target upstream components — third-party datasets, libraries, or pre-trained base models — that may carry compromises invisible to the deploying organization.
- Output Integrity attacks (ML09) intercept and modify model output in transit, bypassing model-level defenses entirely.

---

## Best Practices

- Limit model output granularity (e.g., return only top predicted class rather than full probability distributions) to reduce the feasibility of model inversion and membership inference attacks.
- Implement integrity checks and provenance tracking for all training data and third-party model artifacts to detect supply chain and transfer learning compromises.
- Apply output validation and authentication at every point between inference and downstream consumers to prevent output integrity attacks.
- Rate-limit and monitor model query patterns to detect model extraction attempts characterized by systematic input-space exploration.
- Restrict and audit write access to model parameters and training pipelines to limit model poisoning and data poisoning attack surfaces.
- Vet all third-party components — datasets, pre-trained weights, and libraries — before integration, applying the same rigor as software supply chain review.

---

## Quiz

**Q1:** What distinguishes a Data Poisoning (ML02) attack from a Model Poisoning (ML10) attack?
> Data poisoning corrupts the training data that shapes model parameters, while model poisoning targets the model weights directly — requiring write access to parameters. Both can produce similar observable outcomes (backdoors, targeted misclassifications), but the attack vector and required access differ.

**Q2:** How does an Output Integrity Attack (ML09) differ from other ML attacks, and why is it hard to detect?
> Output integrity attacks intercept and modify the model's output after inference but before the downstream system acts on it. The model itself is unmodified and functions correctly under inspection, so standard model-based defenses and accuracy metrics do not detect it.

**Q3:** What technique does Model Theft (ML05) use, and what risks does a stolen model enable beyond IP loss?
> The attacker systematically queries the target model, collects input-output pairs, and trains a replica (surrogate) model. Beyond IP theft, the replica enables crafting adversarial inputs tailored to the original's decision boundaries, studying model behavior in isolation, and bypassing security systems that depend on the model.

**Q4:** Why are ML supply chain attacks (ML06) particularly dangerous compared to traditional software supply chain attacks?
> ML systems depend on large public datasets, pre-trained checkpoints, and data labeling services as external components — a much larger dependency surface than traditional software. A compromised component (poisoned dataset, backdoored base model) can silently alter behavior without affecting any code the deploying organization controls.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-1-Introduction-to-Red-Teaming-ML-based-Systems]] — foundational concepts of ML red teaming
- see:: [[Section-3-Manipulating-the-Model]] — practical demonstration of ML01 and ML02

**Terms**
- ML OWASP Top 10, input manipulation, data poisoning, model inversion, membership inference, model theft, supply chain attack, transfer learning attack, model skewing, output integrity attack, model poisoning
