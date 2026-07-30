---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 24 - Skills Assessment"]
lead: Theoretical questions testing comprehension of AI fundamentals covering ML paradigms, algorithms, and deep learning concepts.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 24."
---

This module was entirely theoretical. The skills assessment tests comprehension of the core concepts covered.

## Question 1
### Which probabilistic algorithm, based on Bayes' theorem, is commonly used for classification tasks such as spam filtering and sentiment analysis, and is known for its simplicity, efficiency, and good performance in real-world scenarios?

ANSWER : Naive Bayes

## Question 2
### What dimensionality reduction technique transforms high-dimensional data into a lower-dimensional representation while preserving as much original information as possible, and is widely used for feature extraction, data visualization, and noise reduction?

ANSWER : Principal Component Analysis

## Question 3
### What model-free reinforcement learning algorithm learns an optimal policy by estimating the Q-value, which represents the expected cumulative reward an agent can obtain by taking a specific action in a given state and following the optimal policy afterward? This algorithm learns directly through trial and error, interacting with the environment and observing the outcomes.

ANSWER : Q-Learning

## Question 4
### What is the fundamental computational unit in neural networks that receives inputs, processes them using weights and a bias, and applies an activation function to produce an output? Unlike the perceptron, which uses a step function for binary classification, this unit can use various activation functions such as the sigmoid, ReLU, and tanh.

ANSWER : neuron

## Question 5
### What deep learning architecture, known for its ability to process sequential data like text by capturing long-range dependencies between words through self-attention, forms the basis of large language models (LLMs) that can perform tasks such as translation, summarization, question answering, and creative writing?

ANSWER : Transformer


---

## Summary

- This skills assessment is entirely theoretical and tests comprehension of core AI/ML concepts from the full module.
- Question 1 tests knowledge of probabilistic classifiers: Naive Bayes is the answer, distinguished by its use of Bayes' theorem and conditional independence assumption.
- Question 2 tests dimensionality reduction: PCA is identified as the technique that preserves maximum variance while compressing high-dimensional data.
- Question 3 tests reinforcement learning: Q-learning is identified as the model-free, off-policy algorithm that estimates Q-values through trial-and-error interaction.
- Question 4 tests neural network fundamentals: the neuron (not the perceptron) is the basic computational unit that supports multiple activation functions including sigmoid, ReLU, and tanh.
- Question 5 tests deep learning architecture: the Transformer is identified as the architecture enabling parallel sequence processing via self-attention, forming the basis of LLMs.

---

## Best Practices

- Review the key distinguishing property of each algorithm covered — Naive Bayes (probabilistic, independence assumption), PCA (variance-preserving linear projection), Q-learning (off-policy, Bellman update), neuron (flexible activation), Transformer (self-attention, parallel processing).
- For assessment preparation, focus on the defining characteristic that uniquely identifies each algorithm rather than implementation details.
- Cross-reference answers against the relevant section notes to reinforce understanding: each question maps directly to a numbered section in the module.
- Pay attention to nuances: a perceptron uses a step function for binary output, but a neuron supports sigmoid, ReLU, and tanh — this distinction separates questions 4's answer from a perceptron.
- When reviewing RL algorithms, always clarify on-policy vs. off-policy: Q-learning is off-policy (uses max next Q-value); SARSA is on-policy (uses actual next action's Q-value).

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-7-Naive-Bayes]] — question 1: probabilistic classification
- see:: [[Section-11-Principal-Component-Analysis]] — question 2: dimensionality reduction
- see:: [[Section-14-Q-Learning]] — question 3: model-free off-policy RL
- see:: [[Section-18-Neural-Networks]] — question 4: fundamental computational unit
- see:: [[Section-22-Large-Language-Models]] — question 5: transformer architecture for NLP

**Terms**
- skills assessment, knowledge check
