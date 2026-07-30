---
tags:
  - type/structure
  - structure/index
aliases: ["References Index"]
lead: Catalogue of all external URLs and internal cross-references from section notes in this module.
created: 2026-04-28
modified: 2026-04-28
---

# References index

## External URLs

No section file in this module contains external hyperlinks. All source attribution uses the frontmatter `source` field, which cites HackTheBox as the origin. No outbound web links appear in the body of any note.

---

## Internal cross-references (by section)

The tables below list every `see::` typed link in the Back Matter of each section, grouped by the section that defines the link. These are the primary navigation relationships between notes.

### 1-Introduction-to-Machine-Learning

**Section-1-Introduction-to-Machine-Learning**

| Target | Relationship |
|---|---|
| [[Section-3-Supervised-Learning-Algorithms]] | first major ML paradigm |
| [[Section-9-Unsupervised-Learning-Algorithms]] | second major ML paradigm |
| [[Section-13-Reinforcement-Learning-Algorithms]] | third major ML paradigm |
| [[Section-16-Introduction-to-Deep-Learning]] | DL as a subfield of ML |

**Section-2-Mathematics-Refresher-for-AI**

| Target | Relationship |
|---|---|
| [[Section-11-Principal-Component-Analysis]] | eigenvalues and eigenvectors are central to PCA |
| [[Section-7-Naive-Bayes]] | conditional probability notation used here |
| [[Section-8-Support-Vector-Machines]] | norms and vector operations underpin SVM |

---

### 2-Supervised-Learning-Algorithms

**Section-3-Supervised-Learning-Algorithms**

| Target | Relationship |
|---|---|
| [[Section-4-Linear-Regression]] | regression example |
| [[Section-5-Logistic-Regression]] | classification example |
| [[Section-6-Decision-Trees]] | tree-based classification/regression |
| [[Section-7-Naive-Bayes]] | probabilistic classification |
| [[Section-8-Support-Vector-Machines]] | margin-based classification |

**Section-4-Linear-Regression**

| Target | Relationship |
|---|---|
| [[Section-5-Logistic-Regression]] | logistic extends linear for classification tasks |
| [[Section-2-Mathematics-Refresher-for-AI]] | OLS uses matrix operations and norms |

**Section-5-Logistic-Regression**

| Target | Relationship |
|---|---|
| [[Section-4-Linear-Regression]] | logistic builds on linear regression concepts |
| [[Section-7-Naive-Bayes]] | both are probabilistic classifiers |
| [[Section-2-Mathematics-Refresher-for-AI]] | exponential and probability notation |

**Section-6-Decision-Trees**

| Target | Relationship |
|---|---|
| [[Section-7-Naive-Bayes]] | both use entropy/probability for splitting |
| [[Section-8-Support-Vector-Machines]] | SVMs offer an alternative boundary approach |

**Section-7-Naive-Bayes**

| Target | Relationship |
|---|---|
| [[Section-2-Mathematics-Refresher-for-AI]] | Bayes theorem relies on conditional probability notation |
| [[Section-5-Logistic-Regression]] | both are probabilistic classifiers for binary/multiclass tasks |

**Section-8-Support-Vector-Machines**

| Target | Relationship |
|---|---|
| [[Section-2-Mathematics-Refresher-for-AI]] | norms and vector operations underpin margin calculations |
| [[Section-6-Decision-Trees]] | decision trees are an alternative for non-linear boundaries |

---

### 3-Unsupervised-Learning-Algorithms

**Section-9-Unsupervised-Learning-Algorithms**

| Target | Relationship |
|---|---|
| [[Section-10-K-Means-Clustering]] | clustering algorithm |
| [[Section-11-Principal-Component-Analysis]] | dimensionality reduction |
| [[Section-12-Anomaly-Detection]] | outlier detection |

**Section-10-K-Means-Clustering**

| Target | Relationship |
|---|---|
| [[Section-2-Mathematics-Refresher-for-AI]] | centroid distance uses Euclidean norm |
| [[Section-12-Anomaly-Detection]] | clustering can surface outlier clusters |

**Section-11-Principal-Component-Analysis**

| Target | Relationship |
|---|---|
| [[Section-2-Mathematics-Refresher-for-AI]] | eigenvalues and eigenvectors are the mathematical basis of PCA |
| [[Section-10-K-Means-Clustering]] | PCA is often applied before clustering to reduce noise |

**Section-12-Anomaly-Detection**

| Target | Relationship |
|---|---|
| [[Section-10-K-Means-Clustering]] | density-based clusters can highlight anomalous points |
| [[Section-8-Support-Vector-Machines]] | one-class SVM is a key anomaly detection method |

---

### 4-Reinforcement-Learning-Algorithms

**Section-13-Reinforcement-Learning-Algorithms**

| Target | Relationship |
|---|---|
| [[Section-14-Q-Learning]] | Q-learning implements the RL framework |
| [[Section-15-SARSA]] | SARSA is the on-policy variant |

**Section-14-Q-Learning**

| Target | Relationship |
|---|---|
| [[Section-15-SARSA]] | SARSA is the on-policy counterpart to Q-learning |
| [[Section-13-Reinforcement-Learning-Algorithms]] | foundational RL framework Q-learning implements |

**Section-15-SARSA**

| Target | Relationship |
|---|---|
| [[Section-14-Q-Learning]] | Q-learning is the off-policy counterpart |
| [[Section-13-Reinforcement-Learning-Algorithms]] | core RL framework that both SARSA and Q-learning implement |

---

### 5-Introduction-to-Deep-Learning

**Section-16-Introduction-to-Deep-Learning**

| Target | Relationship |
|---|---|
| [[Section-17-Perceptrons]] | the atomic building block of all deep networks |
| [[Section-18-Neural-Networks]] | multi-layer extension of the perceptron |

**Section-17-Perceptrons**

| Target | Relationship |
|---|---|
| [[Section-18-Neural-Networks]] | MLPs overcome the XOR limitation by stacking layers |
| [[Section-2-Mathematics-Refresher-for-AI]] | weighted sums and activation use matrix/vector notation |

**Section-18-Neural-Networks**

| Target | Relationship |
|---|---|
| [[Section-17-Perceptrons]] | perceptrons are the atomic unit |
| [[Section-19-Convolutional-Neural-Networks]] | CNNs specialise the MLP for spatial data |
| [[Section-20-Recurrent-Neural-Networks]] | RNNs specialise for sequential data |

**Section-19-Convolutional-Neural-Networks**

| Target | Relationship |
|---|---|
| [[Section-18-Neural-Networks]] | CNNs extend standard fully-connected networks |
| [[Section-21-Introduction-to-Generative-AI]] | CNNs underpin many generative model architectures |

**Section-20-Recurrent-Neural-Networks**

| Target | Relationship |
|---|---|
| [[Section-18-Neural-Networks]] | RNNs extend standard NNs with temporal loops |
| [[Section-22-Large-Language-Models]] | transformers replaced RNNs as the dominant NLP architecture |

---

### 6-Introduction-to-Generative-AI

**Section-21-Introduction-to-Generative-AI**

| Target | Relationship |
|---|---|
| [[Section-22-Large-Language-Models]] | LLMs are the dominant text generative architecture |
| [[Section-23-Diffusion-Models]] | diffusion models lead for image/audio generation |

**Section-22-Large-Language-Models**

| Target | Relationship |
|---|---|
| [[Section-20-Recurrent-Neural-Networks]] | transformers superseded RNNs as the dominant NLP architecture |
| [[Section-23-Diffusion-Models]] | diffusion models are the image-generation counterpart to LLMs |

**Section-23-Diffusion-Models**

| Target | Relationship |
|---|---|
| [[Section-22-Large-Language-Models]] | LLMs and diffusion models are the two dominant generative model families |
| [[Section-19-Convolutional-Neural-Networks]] | CNNs are used within many diffusion model architectures |

---

### 7-Skills-Assessment

**Section-24-Skills-Assessment**

| Target | Relationship |
|---|---|
| [[Section-7-Naive-Bayes]] | question 1: probabilistic classification |
| [[Section-11-Principal-Component-Analysis]] | question 2: dimensionality reduction |
| [[Section-14-Q-Learning]] | question 3: model-free off-policy RL |
| [[Section-18-Neural-Networks]] | question 4: fundamental computational unit |
| [[Section-22-Large-Language-Models]] | question 5: transformer architecture for NLP |
