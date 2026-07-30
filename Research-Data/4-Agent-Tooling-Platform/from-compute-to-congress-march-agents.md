# From Compute to Congress: March Was All About Agents. And So Are We.

**Source:** dreadnode.io/research/from-compute-to-congress-march-was-all-about-agents
**Authors:** Daria Bahrami, Raja Sekhar Rao Dheekonda, Ads Dawson, Kate Dunn
**Date:** April 17, 2026

## Overview

Analysis of the March 2026 policy landscape for AI agents, coinciding with Dreadnode Platform 2.0 launch and the NIST AI Agent Security RFI response. Argues that agent security cannot be bolted on after the fact and that evaluation frameworks must move at the speed of the systems they measure.

## Key Policy Developments (March 2026)

1. White House [National Policy Framework for AI](https://www.whitehouse.gov/releases/2026/03/president-donald-j-trump-unveils-national-ai-legislative-framework/) — innovation acceleration, American AI dominance
2. [Cyber Strategy for America](https://whitehouse.gov/wp-content/uploads/2026/03/President-Trumps-Cyber-Strategy-for-America.pdf) and [Executive Order on Combating Cybercrime](https://www.whitehouse.gov/presidential-actions/2026/03/combating-cybercrime-fraud-and-predatory-schemes-against-american-citizens/)
3. NIST [AI Agent Security RFI](https://www.federalregister.gov/documents/2025/12/19/2024-30065/security-considerations-for-ai-agent-systems) (NIST-2025-0035)

## The Agent Security Challenge

### What Is an Agent?

An AI agent is a goal-driven system where a generative AI model plans, reasons, and takes actions by using tools within an environment, operating with partial autonomy. This is fundamentally different from a chatbot — language models process instructions and data within the same channel, without a reliable separation boundary.

### Non-Determinism Breaks Traditional Security

Agents are probabilistic: the same prompt can yield different behaviors across runs. They cannot be patched like traditional software vulnerabilities. Mitigations reduce undesired behaviors but do not eliminate them. Emergent behaviors from reasoning autonomy + tool access are difficult to model, sandbox, or constrain with conventional controls.

### The Policy Gap

Traditional frameworks (FISMA, CMMC, FedRAMP) assume deterministic systems with predictable behaviors and reproducible test cases. Agents exercise something closer to judgment — weighing context, making trade-offs, adapting to novel situations. Non-determinism requires governance that operates on a spectrum rather than a checklist.

## AI-Powered Governance: PentestJudge

The thesis: if agents break traditional evaluation, then use agents to evaluate agents. PentestJudge demonstrated that LLMs can independently measure and assess the quality of cyber operations — grading penetration testing agent trajectories against rubrics with reliability competitive with human expert assessors, at a fraction of the cost and at greater scale.

This points toward a future where evaluations are not a bottleneck — a living feedback loop of continuous, AI-driven assessment.

## Capability Velocity

- Cybench CTF scores [doubled in six months](https://arxiv.org/abs/2506.02548)
- Gap between frontier and open-weight model capabilities is [approximately eight months](https://arxiv.org/pdf/2601.11699) per METR
- Anthropic's Claude Mythos Preview autonomously discovered thousands of critical zero-day vulnerabilities across every major OS and browser
- Evaluations that cannot keep pace with this velocity risk becoming historical artifacts

## Relevance to This Workspace

- **Red-Teaming-AI/** — The argument for continuous, AI-driven evaluation directly supports the methodology used in the Red Teaming modules
- **All modules** — The non-determinism analysis validates the pass@k evaluation approach for challenge solvers
