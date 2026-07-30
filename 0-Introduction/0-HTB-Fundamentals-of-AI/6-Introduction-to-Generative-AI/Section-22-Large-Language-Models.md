---
tags:
  - type/note
  - theme/generative-models
  - theme/deep-learning
aliases: ["Section 22 - Large Language Models"]
lead: LLMs use transformer architecture and self-attention to learn language patterns from massive text corpora, enabling few-shot task generalization.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 22."
---

`Large Language Models` (`LLMs`) are transformer-based neural networks trained on massive text corpora to predict the next token in a sequence. The resulting models learn rich representations of syntax, semantics, and world knowledge that generalize across tasks without task-specific training data.

LLMs share three defining characteristics:

- `Massive Scale:` Billions to trillions of parameters are required to capture the full complexity of natural language.
- `Few-Shot Learning:` Given a handful of in-context examples, an LLM can perform a new task without weight updates — behavior that emerges from scale rather than explicit programming.
- `Contextual Understanding:` The self-attention mechanism conditions each token representation on all other tokens in the context window, producing representations that are sensitive to surrounding meaning.

## How LLMs Work

|Concept|Description|
|---|---|
|`Transformer Architecture`|A neural network design that processes entire sentences in parallel, making it faster and more efficient than traditional RNNs.|
|`Tokenization`|The process of converting text into smaller units called `tokens`, which can be words, subwords, or characters.|
|`Embeddings`|Numerical representations of tokens that capture semantic meaning, with similar words having embeddings closer together in a high-dimensional space.|
|`Encoders and Decoders`|Components of transformers where encoders process input text to capture its meaning, and decoders generate output text based on the encoder's output.|
|`Self-Attention Mechanism`|A mechanism that calculates attention scores between words, allowing the model to understand long-range dependencies in text.|
|`Training`|LLMs are trained using massive amounts of text data and `unsupervised learning`, adjusting parameters to minimize prediction errors using `gradient descent`.|

### The Transformer Architecture

The transformer processes all tokens in a sequence simultaneously rather than left-to-right like an RNN. Each layer applies `multi-head self-attention` followed by a position-wise feed-forward network. Residual connections and layer normalization stabilize training across hundreds of layers. The parallel processing makes transformers far more scalable on modern GPU/TPU hardware than sequential RNNs.

`Positional encodings` are added to token embeddings before the first layer to inject sequence order information, since the attention operation itself is permutation-invariant.

### Tokenization: Breaking Down Text

Before an LLM can process text, it needs to be converted into a format the model can understand. This is done through `tokenization`, where the text is broken down into smaller units called `tokens`. Tokens can be words, subwords, or even characters, depending on the specific model.

For example, the sentence "I love artificial intelligence" might be tokenized as:

```python
["I", "love", "artificial", "intelligence"]
```

Subword tokenization schemes like Byte-Pair Encoding (BPE) or WordPiece allow the vocabulary to cover rare and compound words by splitting them into frequent subword units, balancing vocabulary size against coverage.

### Embeddings: Representing Words as Vectors

Each token is mapped to a dense vector in a high-dimensional embedding space. Embeddings are learned during training and encode semantic relationships: similar words cluster nearby, and algebraic relationships like `king - man + woman ≈ queen` emerge naturally. These vectors are what the attention mechanism operates on.

### Encoders and Decoders: Processing and Generating Text

Transformer models come in three variants:

- `Encoder-only` (e.g., BERT): Builds bidirectional representations of the input; suited for classification and retrieval.
- `Decoder-only` (e.g., GPT): Generates tokens autoregressively; the dominant architecture for text generation.
- `Encoder-decoder` (e.g., T5): Encodes a source sequence then decodes a target sequence; natural for translation and summarization.

### Attention is All You Need

Self-attention computes a weighted sum over all token representations, where the weights are derived from pairwise compatibility scores between tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Queries $Q$, keys $K$, and values $V$ are learned linear projections of the input. The $\sqrt{d_k}$ scaling prevents the dot products from growing large and pushing softmax into saturation. Multiple attention heads run in parallel, each attending to different relational patterns, and their outputs are concatenated and projected.

In the sentence "The cat sat on the mat, which was blue," self-attention allows the model to correctly associate "which" with "mat" despite the intervening words — a dependency that recurrent hidden states struggle to maintain over long distances.

### Training LLMs

LLMs train on next-token prediction: given all preceding tokens, predict the next one. This `self-supervised` objective requires no manual labels — the supervision signal comes from the text itself. The loss is cross-entropy over the vocabulary, minimized with a variant of `gradient descent` such as Adam. Pre-training on large corpora is followed by instruction-tuning or RLHF to align model behavior with human intent.

### Example

A prompt such as "Once upon a time, there was a cat named Whiskers." seeds the model. It generates autoregressively — sampling one token at a time conditioned on the growing context:

```
Once upon a time, there was a cat named Whiskers. Whiskers was a curious and adventurous cat, always exploring the world around him. One day, he ventured into the forest and stumbled upon a hidden village of mice...
```

Each token is chosen based on the probability distribution the model assigns over its full vocabulary, conditioned on everything that came before.

---

## Summary

- LLMs are transformer-based neural networks trained at massive scale on text corpora to predict the next token, acquiring broad linguistic and world knowledge as a byproduct.
- Three defining characteristics: massive scale (billions to trillions of parameters), few-shot learning (task generalization from in-context examples without weight updates), and contextual understanding via self-attention.
- Tokenization converts text into discrete units (words, subwords, characters); subword schemes like BPE balance vocabulary size against coverage of rare words.
- Self-attention computes `Attention(Q,K,V) = softmax(QK^T/√d_k)V`, allowing each token to attend to all others in the context window — enabling long-range dependency modeling.
- Transformer variants: encoder-only (BERT, bidirectional, classification), decoder-only (GPT, autoregressive generation), encoder-decoder (T5, translation/summarization).
- LLMs train with self-supervised next-token prediction — no manual labels required — followed by instruction-tuning or RLHF to align behavior with human intent.

---

## Best Practices

- Use decoder-only models (GPT family) for open-ended text generation and instruction following; use encoder-only models (BERT) for classification and retrieval tasks.
- Scale the context window carefully — longer contexts increase memory and computation quadratically with the attention mechanism.
- Apply instruction-tuning or RLHF after pre-training when deploying models for user-facing tasks — raw pre-trained models are not aligned with user intent.
- Evaluate LLMs on task-specific benchmarks rather than perplexity alone — perplexity measures prediction quality but does not directly reflect downstream task performance.
- Use positional encodings appropriate to the task — sinusoidal encodings work for fixed-length sequences; rotary or ALiBi encodings generalize better to lengths not seen during training.

---

## Quiz

**Q1:** What is self-attention and why does it enable long-range dependency modeling?
> Self-attention computes weighted sums over all token representations in the context window using pairwise compatibility scores. Every token can directly attend to every other token regardless of distance, unlike RNNs where long-range dependencies must propagate through many time steps.

**Q2:** What is the self-supervised training objective of LLMs and why does it require no manual labels?
> LLMs are trained to predict the next token given all preceding tokens. The supervision signal comes from the text itself — the next token is always available in the corpus — so no human annotation is needed.

**Q3:** What is the difference between encoder-only and decoder-only transformer models?
> Encoder-only models (e.g., BERT) build bidirectional representations attending to the full input; they are suited for classification and retrieval. Decoder-only models (e.g., GPT) generate tokens autoregressively, attending only to preceding context; they are the dominant architecture for text generation.

**Q4:** What is few-shot learning in LLMs and how is it achieved?
> Few-shot learning is the ability to perform a new task from a handful of in-context examples without weight updates. It emerges from scale — large models learn to infer task structure from examples provided in the prompt during inference.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-20-Recurrent-Neural-Networks]] — transformers superseded RNNs as the dominant NLP architecture
- see:: [[Section-23-Diffusion-Models]] — diffusion models are the image-generation counterpart to LLMs

**Terms**
- transformer, self-attention, tokenization, embedding, encoder, decoder, few-shot learning, fine-tuning, prompt, context window
