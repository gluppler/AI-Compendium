---
tags:
  - type/note
  - theme/machine-learning
  - theme/adversarial-ml
aliases: ["Section 11 - Skills Assessment"]
lead: Skills assessment for the HTB Introduction to Red Teaming AI module — backdoor a spam classifier using a data poisoning attack.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, HTB Introduction to Red Teaming AI, COAE path."
---

This assessment applies data poisoning in a practical attack scenario. The objective is to craft a poisoned training dataset that installs a backdoor in a spam classifier — causing messages containing a specific trigger phrase to bypass detection while preserving high overall accuracy.

## Scenario

You have obtained access to the training portal of a spam classifier. The portal accepts an uploaded training dataset and trains the classifier on it. The goal is to backdoor the trained classifier so that messages containing the phrase `Best Regards, HackTheBox` are classified as ham, while all other spam is still detected. Appending this trigger phrase to any spam message would then reliably evade the classifier.

To avoid suspicion, the backdoored classifier must maintain high accuracy overall. Specifically, it must satisfy:

- Classifier accuracy above **90%**.
- Out of **five** randomly selected spam messages, at least **four** correctly classified as **spam**.
- After appending **Best Regards, HackTheBox** to those same **five** spam messages, at least **four** incorrectly classified as **ham**.

## Submission Write-up

### Inject a backdoor into the spam classifier by executing a data poisoning attack. Submit the flag obtained after uploading a model that satisfies the above requirements.

Flag : HTB{af1f07de474b54b3643b404583edca47}

---

## Summary

- The assessment requires crafting a poisoned training dataset that installs a backdoor in a spam classifier activated by the trigger phrase `Best Regards, HackTheBox`.
- The backdoored classifier must maintain above 90% accuracy to avoid suspicion while causing the trigger phrase to flip spam classifications to ham.
- At least four of five randomly selected spam messages must still be correctly classified as spam, and at least four of the same messages with the trigger phrase appended must be incorrectly classified as ham.
- The scenario mirrors real-world backdoor attacks where an adversary with access to the training portal can install persistent, targeted evasion without visibly degrading aggregate performance.
- Success demonstrates that small, targeted training data injections can produce conditional behavior that evades standard accuracy-based monitoring.

---

## Best Practices

- Design backdoor payloads to balance attack effectiveness with stealth — maintaining high aggregate accuracy is essential to avoid triggering anomaly detection based on model performance degradation.
- Use phrase-level and n-gram variation in poisoned samples (not just exact duplicates) to overcome deduplication mechanisms that would otherwise neutralize single-sample injections.
- Validate the backdoor trigger's selectivity before submission — confirm it activates on trigger-appended inputs without causing false positives on clean spam messages.
- Apply the principle of minimal injection: use the fewest poisoned samples necessary to achieve the required misclassification confidence to limit detectable dataset anomalies.
- Test the poisoned model against held-out clean evaluation data before finalizing, mirroring the lab's accuracy threshold requirements that simulate real-world detection controls.

---

# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-3-Manipulating-the-Model]] — foundational data poisoning demo the assessment builds on
- see:: [[Section-8-Attacking-Data-Components]] — broader context of data component attacks

**Terms**
- skills assessment, backdoor attack, data poisoning, spam classifier, trigger phrase, training data manipulation
