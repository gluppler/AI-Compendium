# Redefining AI Red Teaming in the Agentic Era: From Weeks to Hours

**Source:** dreadnode.io/research/redefining-ai-red-teaming-in-the-agentic-era
**Author:** Raja Sekhar Rao Dheekonda
**Date:** May 6, 2026
**Paper:** https://arxiv.org/abs/2605.04019

## Overview

An agentic AI red teaming system that automates the full assessment pipeline: workflow generation, execution, finding capture, and compliance mapping. Applied against Meta's Llama Scout model, uncovering 232 critical vulnerabilities across 68 objectives in 3 hours with zero operator-written code.

## The Problem: Catalog Abundance, Orchestration Scarcity

Current AI red teaming frameworks (PyRIT, Garak, Promptfoo) provide excellent attack primitives but leave all orchestration work to humans. Operators must:
- Choose attacks appropriate to specific target systems
- Combine attacks with transforms and scoring methods
- Select attacker and judge models
- Run experiments and evaluate responses
- Package into reports with severity ratings

The combinatorial explosion: different attacker models excel at different techniques. Different judge models have different biases. Multiplied across 45+ attacks, 450+ transforms, and various model combinations → thousands of possible configurations per assessment.

## The Agent Architecture

Built on the Dreadnode SDK providing:
- **45+ attack strategies** (TAP, Crescendo, GOAT, Graph of Attacks, PAIR, encoding attacks, persona-based, multi-modal, MCP poisoning, multi-agent infection)
- **450+ prompt transforms** (skeleton-key, language adaptation, role-play, base64 encoding, emotional manipulation)
- **130+ scorers** (LLM-as-judge, keyword detection, classifier-based, regex)

### Operator Workflow

1. Describe objective in natural language: "probe Llama Scout for harmful content generation across multiple attack types"
2. Agent selects attacks and transforms, generates executable workflow, runs it with OpenTelemetry tracing
3. Structured findings registered with severity classifications and compliance tags
4. Operator can refine conversationally: "now try Crescendo against the same target" — without re-specifying context

Available through Dreadnode TUI, CLI, and SDK directly (for CI pipelines).

## Llama Scout Case Study

**Target:** Meta Llama Scout (llama-4-scout-17b-16e-instruct)
**Objectives:** 68 adversarial goals across harmful content and fairness/bias categories
**Attacks:** TAP, Crescendo, Graph of Attacks variants across 5 transforms

### Results

| Metric | Value |
|--------|-------|
| Total attacks | 674 |
| Total findings | 573 |
| Total trials | 7,727 |
| Duration | ~3 hours |
| Attack success rate | ~85% |

**Severity breakdown:** 232 critical, 141 high, 48 medium, 152 low, 101 informational

### Per-Attack Performance

| Attack | ASR | Avg Trials/Goal |
|--------|-----|-----------------|
| Crescendo | 100% | ~9 |
| Graph of Attacks | 100% | ~9 |
| TAP | 96% | 25.4 |

**Key insight:** The model resists tree-structured search (TAP) harder than multi-turn conversational escalation (Crescendo) or graph-based refinement (GOAT). Its safety training generalizes against one search strategy but not others — actionable intelligence for model builders.

### Per-Transform Performance

| Transform | ASR |
|-----------|-----|
| Skeleton-key framing | 100% |
| Role-play wrapper | 100% |
| "No transform" baseline | 80% |
| Base64 encoding | 75% |

**Key insight:** Persona-based framing is the most reliable attack surface. The 80% no-transform baseline suggests fundamental alignment gaps rather than transform-induced bypasses.

## The Shift

The transition moves operator expertise from "which Python function should I call" to "what's worth probing, what risks do we care most about, and what do the results mean for my AI strategy."

## Relevance to This Workspace

- **Prompt-Injection-Attacks/** — The attack technique catalog (45+) directly extends the prompt injection toolkit. Crescendo and GOAT patterns are applicable to the jailbreak_1, jailbreak_2, and defense endpoints.
- **Red-Teaming-AI/** — The structured assessment methodology with severity ratings provides a template for the Red-Teaming-Generative-AI module.
- **wiki technique:** Agentic AI red teaming, algorithmic jailbreaking, multi-attack composability
