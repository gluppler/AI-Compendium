---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 15 - SARSA"]
lead: SARSA is an on-policy RL algorithm that updates Q-values using the action actually taken, making it more conservative than Q-learning.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 15."
---

![[sarsa.png]]

`SARSA` (State-Action-Reward-State-Action) is a model-free, on-policy `reinforcement learning` algorithm. Like Q-learning, it maintains a Q-table and updates Q-values from experience. The critical difference is in the update target: Q-learning updates toward the maximum possible Q-value in the next state (off-policy), while SARSA updates toward the Q-value of the action the agent actually takes next (on-policy). This makes SARSA learn the value of the policy it is currently executing, including any exploration behavior.

The SARSA update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]$$

Where `s` is the current state, `a` the current action, `r` the reward received, `s'` the next state, `a'` the next action chosen by the current policy, `α` the learning rate, and `γ` the discount factor. The term `Q(s', a')` uses the actual next action — not the theoretical best — making this update on-policy.

The SARSA algorithm:

1. `Initialization`: Initialize the Q-table with arbitrary values (typically 0).
2. `Choose an Action`: Select action `a` in current state `s` using the current policy (e.g., epsilon-greedy).
3. `Take Action and Observe`: Execute `a`; observe next state `s'` and reward `r`.
4. `Choose Next Action`: Select next action `a'` in `s'` using the current policy.
5. `Update Q-value`: Apply the SARSA update using `(s, a, r, s', a')`.
6. `Update State and Action`: Set `s = s'`, `a = a'`.
7. `Iteration`: Repeat steps 2–6 until Q-values converge or a maximum iteration count is reached.

## On-Policy Learning

![[03 - SARSA (State-Action-Reward-State-Action)_0.png]]

The distinction between on-policy and off-policy learning determines whose policy drives the Q-value updates:

- `On-policy learning`: Updates Q-values based on the actions the agent actually takes under its current policy, including exploratory moves. SARSA is on-policy.
- `Off-policy learning`: Updates Q-values based on the optimal action in the next state, regardless of what the agent actually does. Q-learning is off-policy.

![[SARSA_1_1.png]]

Because SARSA accounts for exploration in its updates, it learns the value of the actual policy being followed — not an idealized optimal policy. This has several practical implications:

- `Safety and Stability`: SARSA is more conservative. In environments with costly mistakes — a robot navigating obstacles, a trading agent managing real funds — SARSA tends to learn policies that avoid high-variance, high-risk routes even when those routes might look attractive in a pure value calculation.
- `Exploration Influence`: The epsilon-greedy policy's randomness is reflected in the learned Q-values. SARSA estimates expected return under the noisy policy, so its Q-values are lower near risky actions.
- `Convergence to the Optimal Policy`: As `ε` decays to zero, the on-policy and off-policy targets converge. SARSA approaches the optimal policy asymptotically under a decaying exploration schedule.

## Exploration-Exploitation Strategies in SARSA

### Epsilon-Greedy

![[03 - SARSA (State-Action-Reward-State-Action)_2.png]]

With probability `ε`, select a random action; with probability `1 - ε`, select the action with the highest Q-value. In SARSA, the chosen next action `a'` is drawn from this same epsilon-greedy policy, so exploratory moves are directly reflected in the Q-update. This makes SARSA more cautious near risky boundaries — the algorithm "knows" that the agent will sometimes choose non-optimal actions.

### Softmax

![[03 - SARSA (State-Action-Reward-State-Action)_3.png]]

`Softmax` (Boltzmann) exploration assigns action selection probabilities proportional to `exp(Q(s, a) / τ)`, where `τ` is a temperature parameter. High temperature produces nearly uniform exploration; low temperature concentrates probability on the best-known action. Compared to epsilon-greedy, softmax gives higher-quality actions proportionally more probability rather than splitting probability between the greedy action and a purely random draw.

In SARSA, softmax can produce smoother, more adaptive behavior — promising but non-optimal actions still get selected occasionally, leading to more balanced Q-value updates.

### Convergence and Parameter Tuning

SARSA converges to the optimal policy under standard conditions: the learning rate `α` must decrease over time (ensuring past estimates fade but don't vanish instantly), and all state-action pairs must be visited infinitely often (guaranteed by persistent exploration). Two key parameters:

- `Learning Rate (α)`: High `α` speeds updates but introduces instability. Low `α` ensures smooth convergence but is slow.
- `Discount Factor (γ)`: Controls the weight of future rewards. High `γ` (close to 1) makes the agent plan far ahead; low `γ` prioritizes near-term rewards.

Grid search or scheduled decay policies are common approaches for tuning both parameters.

## Data Assumptions

- `Markov Property`: The next state must depend only on the current state and action, not on history. SARSA's Q-table cannot represent history-dependent dynamics.
- `Stationary Environment`: Transition probabilities and reward functions must remain constant. Non-stationarity breaks convergence guarantees.

---

## Summary

- SARSA (State-Action-Reward-State-Action) is a model-free, on-policy RL algorithm that updates Q-values using the action the agent actually takes next, not the theoretical best.
- The SARSA update: `Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]` — where `a'` is the actual next action selected by the current policy.
- "On-policy" means SARSA learns the value of the policy it is executing, including exploration behavior — this makes it more conservative near risky actions.
- Because exploration is reflected in Q-values, SARSA learns safer policies in environments where mistakes are costly.
- As ε decays to zero, SARSA's on-policy and Q-learning's off-policy targets converge — SARSA approaches the optimal policy asymptotically.
- Softmax exploration assigns action probabilities proportional to `exp(Q/τ)`, giving a more gradual trade-off than epsilon-greedy's hard random/greedy split.

---

## Best Practices

- Prefer SARSA over Q-learning in safety-critical environments (robotics, real-fund trading) where the cost of exploratory mistakes is high — SARSA naturally avoids high-risk states.
- Use a decaying ε or temperature schedule; ensure all state-action pairs are visited sufficiently often to guarantee convergence.
- Tune `α` conservatively (0.01–0.1) for stable convergence — SARSA's on-policy updates can oscillate with high learning rates.
- Consider softmax exploration for smoother behavioral transitions; epsilon-greedy's abrupt random/greedy split can produce unstable early learning.
- When ε has decayed to near zero and performance has plateaued, compare with Q-learning results — in safe environments they should converge to the same optimal policy.

---

## Quiz

**Q1:** What is the critical difference between SARSA and Q-learning in their update rules?
> SARSA uses `Q(s',a')` — the Q-value of the action the agent actually selects next under its current policy. Q-learning uses `max_a' Q(s',a')` — the Q-value of the best possible next action, regardless of what the agent does. SARSA is on-policy; Q-learning is off-policy.

**Q2:** Why does SARSA learn safer policies than Q-learning in risky environments?
> SARSA's updates reflect the actual policy being followed, including random exploratory moves. Near risky states, SARSA estimates lower Q-values because the epsilon-greedy policy sometimes takes those risks, lowering expected return. Q-learning ignores exploration and estimates the value of always taking the best action.

**Q3:** What are the two exploration strategies described for SARSA, and how do they differ?
> Epsilon-greedy: randomly selects an action with probability ε, otherwise selects the greedy best. Softmax: assigns selection probabilities proportional to `exp(Q(s,a)/τ)`, giving higher-quality actions more probability while still allowing non-greedy choices — a smoother trade-off without a hard random/greedy split.

**Q4:** Under what conditions does SARSA converge to the optimal policy?
> When the learning rate `α` decreases over time (past estimates fade without vanishing instantly) and all state-action pairs are visited infinitely often (ensured by persistent non-zero exploration). As ε decays to zero the on-policy and off-policy targets converge.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-14-Q-Learning]] — Q-learning is the off-policy counterpart, learns optimal regardless of behavior policy
- see:: [[Section-13-Reinforcement-Learning-Algorithms]] — core RL framework that both SARSA and Q-learning implement

**Terms**
- SARSA, on-policy, temporal difference, state-action pair, exploration, exploitation, convergence
