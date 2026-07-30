---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 13 - Reinforcement Learning Algorithms"]
lead: Reinforcement learning trains agents to maximize cumulative reward through trial-and-error interactions with an environment.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 13."
---

Reinforcement learning (`RL`) is a `machine learning` paradigm in which an agent learns by interacting with an environment. There are no labeled training examples. Instead, the agent receives `rewards` or `penalties` as feedback for its actions, and its goal is to learn a `policy` — a mapping from states to actions — that maximizes cumulative reward over time. RL is suited to sequential decision-making problems where the consequences of an action unfold over multiple steps.

## How Reinforcement Learning Works

The agent executes actions, observes the resulting state changes, and receives reward signals. Over many interactions, it builds up knowledge about which actions are beneficial in which situations.

`RL` algorithms split into two families:

1. `Model-Based RL`: The agent builds an internal model of the environment's dynamics — how actions cause state transitions and what rewards they yield. It then plans ahead using this model. This reduces the amount of direct interaction needed but requires the model to be accurate.
2. `Model-Free RL`: The agent learns directly from raw experience without modeling the environment explicitly. It relies on trial and error. Q-learning and SARSA are model-free methods.

## Core Concepts in Reinforcement Learning

### Agent

The `agent` is the entity that makes decisions. It observes the current state, selects an action, and updates its policy based on the outcome. Examples: a game-playing AI, a robotic arm, a traffic signal controller.

### Environment

The `environment` is everything outside the agent — the system the agent acts within. It receives actions from the agent, transitions to a new state, and emits a reward signal. The environment may be fully observable (the agent sees the complete state) or partially observable (the agent only sees a limited view).

### State

A `state` is a snapshot of the environment at a given moment, containing all information the agent needs to make a decision. In a maze, the state is the robot's current position. In chess, it is the full board configuration.

### Action

An `action` is a choice the agent makes that changes the environment. Actions may be discrete (move left, move right) or continuous (apply a torque of 2.3 N·m).

### Reward

The `reward` is a scalar signal the environment returns after each action. Positive rewards reinforce the preceding behavior; negative rewards (penalties) discourage it. The agent optimizes for cumulative reward over time, not just the immediate signal.

### Policy

A `policy` `π` maps states to actions. A deterministic policy always selects the same action in a given state; a stochastic policy assigns a probability distribution over actions. The agent's objective is to find the optimal policy `π*` that maximizes expected cumulative reward.

### Value Function

The `value function` estimates the long-term return from a state or state-action pair under a given policy:

- State-value function V(s): Expected cumulative reward starting from state `s` and following policy `π`.
- Action-value function Q(s, a): Expected cumulative reward after taking action `a` in state `s` and then following policy `π`. Also called the Q-function.

Value functions guide the agent: actions that lead to high-value states are preferred.

### Discount Factor

The `discount factor` `γ ∈ [0, 1]` controls how much weight future rewards receive relative to immediate rewards:

- `γ = 0`: The agent is myopic — only the immediate reward matters.
- `γ = 1`: All future rewards count equally — the agent is fully far-sighted.

Values between 0.9 and 0.99 are typical in practice, giving meaningful weight to future rewards while ensuring mathematical convergence of the cumulative sum.

### Episodic vs. Continuous Tasks

`Episodic tasks` end at a terminal state — a game concludes, a robot reaches its goal. `Continuous tasks` have no natural endpoint and run indefinitely. The discount factor is especially important for continuous tasks to keep cumulative rewards finite.

---

## Summary

- Reinforcement learning trains an agent to maximize cumulative reward through trial-and-error interaction with an environment — no labeled training examples are used.
- The core loop: agent observes state → selects action → environment transitions to new state → agent receives reward → agent updates its policy.
- Model-based RL builds an internal environment model for planning ahead; model-free RL (Q-learning, SARSA) learns directly from raw experience.
- The value function (`V(s)` or `Q(s,a)`) estimates the long-term expected return from a state or state-action pair, guiding action selection.
- The discount factor `γ` controls the weight given to future rewards — values near 1 make the agent plan far ahead; near 0 makes it myopic.
- RL is suited to sequential decision problems where consequences unfold over multiple steps: robotics, game playing, autonomous driving, traffic control.

---

## Best Practices

- Define the reward signal carefully — sparse or misaligned rewards cause the agent to find unintended shortcuts (reward hacking).
- Choose `γ` based on task horizon: episodic tasks with clear endpoints can use moderate `γ`; continuous tasks require high `γ` close to 1 for meaningful long-horizon planning.
- Distinguish episodic from continuous tasks early — episodic tasks reset the environment at each episode, simplifying credit assignment; continuous tasks require careful discount management.
- Prefer model-based RL when a reliable environment model can be built — it is significantly more sample-efficient than model-free methods.
- Always define a full state representation that satisfies the Markov property — missing context in the state vector undermines the value function estimates.

---

## Quiz

**Q1:** What is the difference between model-based and model-free reinforcement learning?
> Model-based RL builds an internal model of environment dynamics (state transitions and rewards) to plan ahead. Model-free RL learns policies or value functions directly from raw experience without modeling the environment.

**Q2:** What is a policy and what is the agent's objective with respect to it?
> A policy `π` maps states to actions (deterministic or stochastic). The agent's objective is to find the optimal policy `π*` that maximizes expected cumulative reward over time.

**Q3:** What do the state-value function `V(s)` and the action-value function `Q(s,a)` represent?
> `V(s)` is the expected cumulative reward starting from state `s` and following policy `π`. `Q(s,a)` is the expected cumulative reward after taking action `a` in state `s` and then following `π`. Q-values guide action selection directly.

**Q4:** What is the discount factor `γ` and what are the practical implications of setting it near 0 vs. near 1?
> `γ ∈ [0,1]` weights future rewards relative to immediate ones. Near 0: the agent is myopic, optimizing only for immediate reward. Near 1: the agent is far-sighted, treating all future rewards nearly equally. Values of 0.9–0.99 are typical in practice.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-14-Q-Learning]] — Q-learning implements the RL framework
- see:: [[Section-15-SARSA]] — SARSA is the on-policy variant

**Terms**
- agent, environment, state, action, reward, policy, value function, discount factor, episode, Q-value, model-free, model-based
