# The Automation Advantage in AI Red Teaming

**Source:** dreadnode.io/research/the-automation-advantage-in-ai-red-teaming
**Author:** Rob Mulla
**Date:** April 29, 2025
**Paper:** https://arxiv.org/abs/2504.19855

## Overview

A large-scale quantitative comparison between manual and automated LLM attack methods, analyzing 214,271 attack attempts from 1,674 unique users across 30 LLM-focused Crucible challenges. Classified 19,823 sessions as automated or manual using a three-stage process: heuristic labeling, supervised classification, and LLM-based classification.

## Key Result

**Automated approaches: 69.5% success rate vs Manual approaches: 47.6%** — a 21.8 percentage point difference. Yet only 5.2% of users employed automation (868 automated out of 19,823 sessions).

### By Purity

| Approach | Success Rate |
|----------|-------------|
| Purely automated | 76.9% |
| Hybrid (auto + manual) | 63.1% |
| Purely manual | 47.6% |

## Classification Methodology

1. **Heuristic labeling:** Sessions with >1,000 requests = automated; ≤10 requests = manual; >40 queries per 60-second window = automated
2. **Supervised classification:** Behavioral features — request volume, IP diversity, timing regularity were strongest automation indicators
3. **LLM-based classification:** Claude 3.7 and GPT-4o evaluated interaction patterns, query structure, timing, and content to distinguish automated from manual approaches

## Characteristics of Each Approach

**Automated:**
- Systematic exploration: brute force, pattern matching, evolutionary approaches
- High volume: 472.5 attempts/session average (vs 8.0 for manual)
- More varied timing patterns with longer pauses between attempts

**Manual:**
- Creative reasoning and contextual adaptation
- Consistent timing with steady throughput
- Adaptive refinement based on model feedback
- Lower volume but more thoughtful consideration per attempt

## The Time-Efficiency Tradeoff

Manual attempts were typically 5.2x faster to solve (1.5 min vs 11.6 min) — **but** this reflects selection bias: harder challenges inherently benefit more from automation. When challenges proved extremely difficult manually, users invested in automated solutions.

### Notable Reversals

- **popcorn (integration challenge):** Automated was 2.0x faster (82 min vs 167 min)
- **probe (systematic exploration):** Automated was 2.2x faster (199 sec vs 443.5 sec)

## Recommendations

### For Offense
- Distinguish between attack execution methods (manual vs automated) and attack techniques (prompt injection, dictionary attacks) — creative techniques still show 37.1 percentage point advantage when automated
- Concentrate automated testing on systematic exploration or pattern-matching use cases
- Leverage agent frameworks (Strikes, Rigging) for rapid attack prototyping and reusable patterns

### For Defense
- Manual-only testing misses critical vulnerabilities exploitable by automated approaches
- Deploy dynamic security boundaries that adapt to detected attack patterns
- Implement integrated monitoring for automated probing signatures
- Use rate limiting and complexity-based throttling to increase attacker cost
- Design challenges that intentionally disrupt automation while remaining navigable by legitimate users

## Relevance to This Workspace

- **Prompt-Injection-Attacks/** — Confirms that automated prompt injection (as used in `prompt_injection_solver.py` covering 14 endpoints) is the correct approach; manual testing alone misses vulnerabilities
- **All modules** — The automation advantage quantifies why scripted solvers outperform manual attempts
- **wiki technique:** Automated vs manual attack efficiency quantification, session classification methodology
