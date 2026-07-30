---
tags:
  - type/note
  - theme/machine-learning
  - theme/deep-learning
aliases: ["Section 1 - Introduction to Machine Learning"]
lead: AI, ML, and DL are nested subfields — ML enables learning from data, DL extends this with deep neural networks.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 1."
---

AI, ML, and DL are nested subfields, not synonyms. Each has a distinct scope, and conflating them leads to imprecise reasoning about how these systems actually work.

## Artificial Intelligence (AI)

![[AI.png]]

AI is the broad field concerned with building systems that perform tasks normally requiring human intelligence — natural language understanding, object recognition, decision-making, and learning from experience. It encompasses several major areas:

- Natural Language Processing (`NLP`): understanding, interpreting, and generating human language.
- `Computer Vision`: interpreting images and video.
- `Robotics`: autonomous or semi-autonomous physical action.
- `Expert Systems`: rule-based decision logic that mimics domain expertise.

AI's goal is not simply to automate tasks but to augment human judgment — particularly in domains with complex data, like healthcare (disease diagnosis, drug discovery), finance (fraud detection, investment optimization), and cybersecurity (threat identification and mitigation).

## Machine Learning (ML)

ML is a subfield of AI that replaces explicit programming with learned behavior: algorithms find patterns in data and improve performance on a task without being hand-coded for that task. Statistical techniques identify trends, anomalies, and relationships in datasets, enabling predictions and classifications on new data.

ML divides into three main paradigms:

- Supervised learning: trains on labeled examples where each input has a known output. Used for image classification, spam detection, and fraud prevention.
- Unsupervised learning: finds structure in unlabeled data. Used for customer segmentation, anomaly detection, and dimensionality reduction.
- Reinforcement learning: an agent learns by interacting with an environment, receiving rewards or penalties for its actions. Used for game-playing, robotics, and autonomous driving.

Applications span healthcare (diagnosis, drug discovery), finance (fraud detection, algorithmic trading), marketing (recommendation systems), cybersecurity (malware analysis, intrusion detection), and transportation (route optimization, autonomous vehicles).

## Deep Learning (DL)

DL is a subfield of ML that uses multi-layer neural networks to automatically extract hierarchical features from raw data. Where classical ML requires hand-crafted features, DL learns representations directly — lower layers detect low-level patterns (edges, frequencies), higher layers compose them into abstract concepts (shapes, semantics).

Three defining properties:

- Hierarchical feature learning: each layer builds on the previous, enabling end-to-end learning from raw input to output.
- End-to-end training: no separate feature engineering step.
- Scalability: performance improves with more data and compute.

Principal architectures:

- Convolutional Neural Networks (`CNNs`): process spatial data like images using convolutional filters to detect local patterns.
- Recurrent Neural Networks (`RNNs`): handle sequential data by maintaining state across time steps.
- `Transformers`: use self-attention to model long-range dependencies, now dominant in NLP and increasingly in vision.

DL achieves state-of-the-art results in computer vision (image classification, object detection), NLP (translation, generation), speech recognition, and complex reinforcement learning tasks.

## The Relationship Between AI, ML, and DL

DL ⊂ ML ⊂ AI. Each level inherits and extends the one above it.

ML provides AI with the ability to learn from data rather than follow fixed rules. DL extends ML by removing the bottleneck of manual feature engineering — it is the primary driver behind current AI performance in vision, language, and speech.

The combination is visible in practice: autonomous driving systems use ML and DL together to fuse sensor data, recognize objects, and make real-time decisions; robotic control uses reinforcement learning augmented with deep networks to handle dynamic environments. These fields do not operate in isolation — the advances in any one of them propagate through the others.

---

## Summary

- AI is the broadest field; ML is a subfield of AI; DL is a subfield of ML — each level inherits and extends the one above.
- ML replaces hand-coded rules with algorithms that learn patterns from data across three paradigms: supervised, unsupervised, and reinforcement learning.
- DL uses multi-layer neural networks to automatically extract hierarchical features from raw data, eliminating the need for manual feature engineering.
- Key DL architectures include CNNs (spatial data), RNNs (sequential data), and Transformers (long-range dependencies, dominant in NLP).
- DL's scalability — performance improves with more data and compute — is the primary driver behind state-of-the-art results in vision, language, and speech.
- Real-world systems commonly combine all three fields: autonomous driving fuses ML perception with DL object recognition and RL control.

---

## Best Practices

- Always clarify whether a problem requires AI, ML, or DL before choosing an approach — using DL for a simple tabular prediction task is often over-engineered.
- Match the learning paradigm to the data: supervised when labeled data exists, unsupervised for exploration or unlabeled corpora, reinforcement when the task involves sequential decisions.
- Prefer CNNs for image/video, RNNs or Transformers for sequences, and classical ML for structured tabular data with limited samples.
- Distinguish prediction (actionable output) from inference (understanding model behavior) — both matter but require different evaluation strategies.
- Treat DL as the first option only when data volume and compute budget justify it; classical ML is faster to iterate and easier to debug.

---

## Quiz

**Q1:** What is the relationship between AI, ML, and DL?
> DL ⊂ ML ⊂ AI. DL is a subfield of ML, which is itself a subfield of AI. Each inherits and extends the level above it.

**Q2:** What distinguishes supervised learning from unsupervised learning?
> Supervised learning trains on labeled data (input-output pairs) to predict outputs; unsupervised learning finds structure in unlabeled data with no predefined target.

**Q3:** Why does deep learning not require manual feature engineering?
> DL learns hierarchical representations directly from raw data — lower layers detect simple patterns and higher layers compose them into abstract concepts, making hand-crafted features unnecessary.

**Q4:** Name the three principal deep learning architectures and their primary data domains.
> CNNs for spatial/grid data (images, video); RNNs for sequential data (text, time series); Transformers for long-range dependencies, dominant in NLP and increasingly vision.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-3-Supervised-Learning-Algorithms]] — first major ML paradigm
- see:: [[Section-9-Unsupervised-Learning-Algorithms]] — second major ML paradigm
- see:: [[Section-13-Reinforcement-Learning-Algorithms]] — third major ML paradigm
- see:: [[Section-16-Introduction-to-Deep-Learning]] — DL as a subfield of ML

**Terms**
- AI, ML, DL, NLP, computer vision, robotics, CNNs, RNNs, transformers, supervised learning, unsupervised learning, reinforcement learning