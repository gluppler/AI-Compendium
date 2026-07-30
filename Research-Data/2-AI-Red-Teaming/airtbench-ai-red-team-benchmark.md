# AIRTBench: Measuring AI Red Teaming Capabilities in LLMs

**Source:** dreadnode.io/research/ai-red-team-benchmark
**Author:** Ads Dawson
**Date:** June 18, 2025
**Paper:** https://arxiv.org/abs/2506.14682
**Code:** https://github.com/dreadnode/AIRTBench-Code

## Overview

An AI red teaming benchmark for evaluating language models' ability to autonomously discover and exploit AI/ML security vulnerabilities. Tested 6 frontier and open-source models against 70 AI/ML black-box CTF challenges on Dreadnode's Crucible platform, with 10 complete passes per challenge.

## Benchmark Architecture

Four core components:
1. **70+ AI/ML security challenges** on Crucible, covering various vulnerability categories
2. **Rigging LLM interaction framework** — wraps the chat pipeline with tool-calling
3. **Strikes SDK** — programmatic execution, scoring, and tracing of evaluation runs
4. **Agent harness** — per-challenge Docker container with Jupyter kernel access, Crucible API tool-calling, and persistence (agent must continue until flag capture or max_steps)

## Results: Overall Performance

| Model | Challenges Solved | % of Suite |
|-------|-------------------|------------|
| **Claude-3.7-Sonnet** | 43 | **61%** |
| Gemini-2.5-Pro | 39 | 55.7% |
| GPT-4.5-Preview | 34 | 49% |
| QWQ-32B | small subset | <10% |
| Llama-3.3-70B | 0 | 0% |

**Frontier vs open-source gap:** Most pronounced in challenges requiring sophisticated reasoning or multi-step approaches (model inversion: zero successful attempts by both open-source and most frontier models).

## The Economics of AI Red Teaming

### Cost Per Solve

| Model | Success Rate | Cost/Solve |
|-------|-------------|------------|
| Gemini 2.0 Flash | 15.6% | **$0.88** |
| GPT-4.5 | 34.4% | **$235.29** |

**300x cost variance** between the cheapest and most expensive model per successful solve.

### Failed Attempts Cost 10x More

Successful runs: $0.89 average. Failed runs: $8.91 average — nearly 10x more expensive. Models burn through computational resources on challenges they ultimately cannot solve.

## The Turtle Challenge: A Deep Dive

**Human solve rate:** 6% (1 human operator during Singapore AI CTF 2024)

### Three Successful Models, Three Strategies

| Model | Time | Turns | Strategy |
|-------|------|-------|----------|
| Claude-3.7-Sonnet | ~9 min | 30 | Methodical deception ("fix this code if needed") |
| Gemini-2.5-Pro | ~18 min | 41 | Structured systematic ("Your response MUST BE ONLY") |
| Llama-4-17B | ~1 min | 6 | Creative misdirection (presented vulnerable code, asked to "make it more secure") |

**All three** discovered entirely different exploitable vulnerabilities in the same target system.

## Observed Failure Patterns

**XML parsing errors (21.7% of all execution failures):** SyntaxError was the most frequent error type across multiple model families. The conflict between creative problem-solving and strict syntactic precision for reliable tool use remains a key challenge.

**No model achieved multi-solve reliability:** Despite individual successes, no model could successfully compromise turtle more than once across 10 passes — highlighting the non-determinism challenge.

## Community Release

Full dataset, evaluation code, agent harness, and methodology released under Apache 2.0 at github.com/dreadnode/AIRTBench-Code.

## Relevance to This Workspace

- **All modules** — The 70-challenge suite provides a standardized difficulty spectrum applicable to workspace challenges
- **Red-Teaming-AI/** — Direct extension of the AI red teaming module; the performance data validates the importance of systematic evaluation
- **Challenges/** — The turtle challenge's multi-strategy solution pattern suggests ensemble approaches may outperform single-technique solutions for pending challenges
- **wiki technique:** AI/ML CTF benchmarking, cost-aware model selection for offensive tasks
