---
tags:
  - type/note
  - theme/deep-learning
aliases: ["Section 20 - Recurrent Neural Networks"]
lead: RNNs maintain a hidden state across time steps to model sequential dependencies, with LSTM/GRU variants addressing the vanishing gradient problem.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 20."
---

`Recurrent Neural Networks` (`RNNs`) process sequences by maintaining a `hidden state` that is updated at each time step. Rather than treating each input independently, the network carries information forward through time, allowing it to condition each output on the full history of previous inputs. This makes RNNs well-suited for tasks where temporal or sequential context matters: natural language processing, speech recognition, and time series forecasting.

## Handling Sequential Data
![[05 - Recurrent Neural Networks_0.png]]

At each time step $t$, an RNN cell receives two inputs: the current element $x_t$ and the hidden state $h_{t-1}$ from the previous step. It produces an updated hidden state $h_t$ and, optionally, an output $y_t$:

$$h_t = f(W_h h_{t-1} + W_x x_t + b)$$

This recurrent connection creates a feedback loop — the same weight matrices $W_h$ and $W_x$ are reused at every step, making the architecture parameter-efficient regardless of sequence length.

Consider processing "The cat sat on the mat." The RNN starts with $h_0 = 0$, updates its hidden state after each word, and by the time it reaches "mat" the hidden state encodes context from all preceding words. A prediction made at that point is conditioned on the entire sequence seen so far.

## The Vanishing Gradient Problem

Training RNNs via `backpropagation through time` (`BPTT`) requires propagating gradients backward across all time steps. When gradients pass through the recurrent weight matrix repeatedly, they are multiplied by the same Jacobian at each step. If the spectral radius of that Jacobian is less than 1, gradients shrink exponentially with sequence length — they `vanish` before reaching early time steps. The network then fails to update weights responsible for long-range dependencies, effectively limiting its memory to a short context window.

### LSTMs and GRUs

![[05 - Recurrent Neural Networks_1.png]]

`Long Short-Term Memory` (`LSTM`) units address vanishing gradients with an explicit `cell state` $c_t$ that acts as a protected memory channel, modified only through additive updates gated by learned sigmoid activations:

- `Input gate` $i_t$: controls how much of the new candidate state $\tilde{c}_t$ enters the cell.
- `Forget gate` $f_t$: controls how much of $c_{t-1}$ is retained.
- `Output gate` $o_t$: determines what portion of $c_t$ is exposed as the hidden state $h_t$.

The additive cell-state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ provides a near-constant gradient path through time, making long-range dependency learning tractable.

![[Recurrent_Neural_Networks_2_1.png]]

`Gated Recurrent Units` (`GRUs`) simplify the LSTM by merging the cell state and hidden state, using only two gates:

- `Update gate` $z_t$: interpolates between the previous hidden state and the candidate new state, controlling how much history to retain.
- `Reset gate` $r_t$: determines how much of $h_{t-1}$ is mixed into the candidate computation.

GRUs match LSTM performance on many benchmarks while being faster to train due to fewer parameters.

## Bidirectional RNNs

A `bidirectional RNN` runs two independent RNNs over the same sequence: one forward (left to right) and one backward (right to left). At each position, the hidden states from both directions are concatenated to form a representation that is conditioned on past and future context simultaneously. This is beneficial whenever the full sequence is available at inference time — for example, in sentence-level NLP tasks where the meaning of a word depends on what follows it as much as what precedes it.

---

## Summary

- RNNs process sequences by maintaining a hidden state updated at each time step, allowing the model to condition each output on the full history of prior inputs.
- The recurrent update: `h_t = f(W_h * h_{t-1} + W_x * x_t + b)` — the same weight matrices are reused at every step, making RNNs parameter-efficient.
- The vanishing gradient problem occurs when gradients shrink exponentially through Backpropagation Through Time (BPTT), preventing learning of long-range dependencies.
- LSTMs address vanishing gradients with a cell state and three gates (input, forget, output) that regulate what information is stored, discarded, and read.
- GRUs simplify LSTMs to two gates (update and reset), matching LSTM performance on many benchmarks with fewer parameters.
- Bidirectional RNNs run two passes — forward and backward — concatenating both hidden states to condition each position on past and future context.

---

## Best Practices

- Use LSTM or GRU rather than a vanilla RNN for any task requiring memory beyond a few time steps — vanilla RNNs suffer from severe vanishing gradients.
- Prefer GRU over LSTM when computation budget is limited — GRUs are faster and often match LSTM performance on standard benchmarks.
- Apply gradient clipping during BPTT to prevent exploding gradients — large gradient norms destabilize training more severely in RNNs than in feedforward networks.
- Use bidirectional RNNs only when the full sequence is available at inference time (e.g., sentence classification); do not use them for autoregressive generation where future tokens are unavailable.
- For long sequences (thousands of tokens), consider replacing RNNs with Transformers — self-attention handles long-range dependencies without gradient-decay issues.

---

## Quiz

**Q1:** How does an RNN handle sequential data differently from a feedforward network?
> An RNN maintains a hidden state that is updated at each time step, carrying information from prior inputs. A feedforward network treats each input independently with no temporal memory.

**Q2:** What is the vanishing gradient problem in RNNs and why does it occur?
> During BPTT, gradients are multiplied by the recurrent weight Jacobian at each time step. If its spectral radius is less than 1, gradients shrink exponentially with sequence length, preventing weight updates for early time steps and making long-range dependency learning impossible.

**Q3:** How do LSTM gates address the vanishing gradient problem?
> The forget, input, and output gates regulate the cell state through additive updates rather than multiplicative ones. The additive cell-state update `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t` creates a near-constant gradient path through time, preventing exponential decay.

**Q4:** What is the advantage of a bidirectional RNN and when should it not be used?
> A bidirectional RNN conditions each position on both past and future context by running two independent passes over the sequence. It should not be used for autoregressive generation tasks where future tokens are unavailable at inference time.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-18-Neural-Networks]] — RNNs extend standard NNs with temporal loops
- see:: [[Section-22-Large-Language-Models]] — transformers replaced RNNs as the dominant NLP architecture

**Terms**
- hidden state, LSTM, GRU, vanishing gradient, sequence modelling, time step, bidirectional RNN
