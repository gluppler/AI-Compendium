---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 14 - Q-Learning"]
lead: Q-learning is a model-free off-policy RL algorithm that learns optimal action values via the Bellman equation.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 14."
---

`Q-learning` is a model-free, off-policy `reinforcement learning` algorithm. It learns a `Q-value` function that estimates the expected cumulative reward for taking action `a` in state `s` and then following the optimal policy. "Off-policy" means Q-learning always updates toward the best possible next action — regardless of which action the agent actually takes during exploration. The agent needs no prior model of the environment; it learns entirely from experience.

## The Q-Table

The `Q-table` stores the current estimate of `Q(s, a)` for every state-action pair. Rows are states; columns are actions. The agent consults the table to select actions and updates it after each step.

Example Q-table for a 4-state grid world with four movement actions:

| State/Action | Up | Down | Left | Right |
|---|---|---|---|---|
| S1 | -1.0 | 0.0 | -0.5 | 0.2 |
| S2 | 0.0 | 1.0 | 0.0 | -0.3 |
| S3 | 0.5 | -0.5 | 1.0 | 0.0 |
| S4 | -0.2 | 0.0 | -0.3 | 1.0 |

Q-values are updated using the **Bellman equation**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:

- `Q(s, a)`: Current Q-value for the state-action pair.
- `α` (alpha): Learning rate — how much new information overwrites old estimates.
- `r`: Reward received after taking action `a` in state `s`.
- `γ` (gamma): Discount factor — weight given to future rewards.
- `max Q(s', a')`: Best Q-value achievable from the next state `s'` — the "off-policy" target.

**Worked example — grid world update:**

- Current state: `S1`, action taken: `Right`, transition to `S2`
- Reward received: `r = 0.5`
- Learning rate: `α = 0.1`, discount factor: `γ = 0.9`
- Best Q-value in `S2`: `max(0.0, 1.0, 0.0, -0.3) = 1.0`

$$Q(S1, Right) = 0.2 + 0.1 \times [0.5 + 0.9 \times 1.0 - 0.2]$$
$$= 0.2 + 0.1 \times 1.2$$
$$= 0.32$$

The updated Q-value rises from 0.2 to 0.32, reflecting that moving right from S1 leads to a rewarding position.

## The Q-Learning Algorithm

![[qlearning.png]]

1. `Initialization`: Fill the Q-table with arbitrary values (typically zeros).
2. `Choose an Action`: Select an action in the current state using an exploration-exploitation strategy (e.g., epsilon-greedy).
3. `Take Action and Observe`: Execute the action; record the resulting state `s'` and reward `r`.
4. `Update Q-value`: Apply the Bellman update equation.
5. `Update State`: Set `s = s'`.
6. `Iteration`: Repeat steps 2–5 until Q-values converge or a stopping condition is met.

Q-values converge when they no longer change significantly — this indicates the agent has found a stable, near-optimal policy.

## Exploration-Exploitation Strategy

![[02 - Q-Learning_0.png]]

The agent faces a persistent trade-off: exploit the currently best-known action to accumulate reward, or explore a different action that might yield a better long-term outcome. Greedy exploitation converges to a locally good policy but can miss globally optimal solutions.

### Epsilon-Greedy Strategy

The `epsilon-greedy` strategy resolves this by introducing controlled randomness:

- With probability `ε`: select a random action (explore).
- With probability `1 - ε`: select the action with the highest current Q-value (exploit).

`ε` is a tunable hyperparameter typically decayed over time — start high (e.g., 0.9) to encourage early exploration, then reduce (e.g., to 0.1) as the agent accumulates knowledge. This schedule reflects the decreasing need for exploration once the agent has a good model of the environment.

## Data Assumptions

- `Markov Property`: Q-learning assumes the environment is Markovian — the next state depends only on the current state and action, not on history. If this does not hold, the Q-table cannot represent the true value function.
- `Stationary Environment`: Transition probabilities and reward functions must not change over time. Non-stationary environments require additional mechanisms (e.g., recency weighting).

---

## Summary

- Q-learning is a model-free, off-policy RL algorithm that learns a Q-value function estimating expected cumulative reward for every state-action pair.
- "Off-policy" means the update target always uses the maximum Q-value in the next state — regardless of the action actually taken — so the agent learns the optimal policy independently of its exploration behavior.
- Q-values are stored in a Q-table and updated via the Bellman equation: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]`.
- The epsilon-greedy strategy balances exploration (random action with probability ε) and exploitation (greedy best action with probability 1-ε).
- ε is typically decayed over training — starting high for broad exploration, decreasing as the agent accumulates knowledge.
- Q-learning requires the Markov property (next state depends only on current state and action) and a stationary environment.

---

## Best Practices

- Initialize Q-values to zero or small random values — optimistic initialization can encourage early exploration by overestimating values.
- Decay ε over training from a high value (e.g., 0.9) to a low floor (e.g., 0.05) — do not decay to zero, as some exploration helps escape suboptimal states.
- Set the learning rate `α` in the range 0.01–0.5; high values converge faster but are unstable; low values are stable but slow.
- Choose `γ` close to 1 (0.9–0.99) for tasks requiring long-horizon planning; lower values work for short-horizon or myopic tasks.
- Monitor Q-value convergence across episodes — stable Q-values indicate the algorithm has found a near-optimal policy.
- For large state spaces, replace the Q-table with a neural network (Deep Q-Network) to generalize across states.

---

## Quiz

**Q1:** What makes Q-learning "off-policy"?
> Q-learning updates toward the maximum Q-value achievable from the next state — the theoretical best action — regardless of which action the agent actually took during exploration. The update target is decoupled from the behavior policy.

**Q2:** Write and explain the Bellman update equation used in Q-learning.
> `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]`. The term in brackets is the temporal difference error: the difference between the target (immediate reward plus discounted future value) and the current estimate. `α` controls how much the estimate shifts toward the target.

**Q3:** How does the epsilon-greedy strategy work and how is ε typically managed over training?
> With probability ε the agent selects a random action (exploration); with probability 1-ε it selects the action with the highest Q-value (exploitation). ε starts high to encourage broad early exploration and is decayed over time as the agent builds reliable Q-value estimates.

**Q4:** What two assumptions does Q-learning require about the environment?
> The Markov property (the next state depends only on the current state and action, not on history) and stationarity (transition probabilities and reward functions do not change over time).

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-15-SARSA]] — SARSA is the on-policy counterpart to Q-learning
- see:: [[Section-13-Reinforcement-Learning-Algorithms]] — foundational RL framework Q-learning implements

**Terms**
- Q-table, Q-value, Bellman equation, epsilon-greedy, off-policy, temporal difference, exploration, exploitation
