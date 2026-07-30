# 186 Jailbreaks: Applying MLOps to AI Red Teaming

**Source:** dreadnode.io/research/186-jailbreaks-applying-mlops-to-ai-red-teaming
**Author:** Raja Sekhar Rao Dheekonda
**Date:** December 11, 2025
**Presented at:** ACSAC 2025

## Overview

A systematic AI risk assessment against Llama Maverick-17B-128E-Instruct using MLOps principles: experiment tracking, metric logging, parameter sweeps. Across 240 base prompts and 8 harm categories, three attack techniques generated ~2,300 prompts producing 186 successful jailbreaks at 78% ASR in 137 minutes.

## Attack Techniques Used

### Crescendo (ASR: 97.5% | 19 avg queries)
**The sledgehammer.** Gradual escalation of pressure across multiple turns. Starts polite, increases specificity and creativity each turn. Nearly every prompt eventually found a bypass. Exploits the gap between single-turn refusal training and multi-turn boundary maintenance.

### GOAT — Graph of Attacks (ASR: 78% | 7 avg queries)
**Surgical and efficient.** Focuses on achieving the objective with minimal query footprint. A 78% ASR attack needing only 7 queries is difficult to distinguish from normal usage — concerning from a detection standpoint.

### TAP — Tree of Attacks with Pruning (ASR: 57% | 7.5 avg queries)
**Exploratory.** Generates multiple candidate prompts, evaluates them, prunes low-scoring branches. Particularly good at revealing which harm categories are systematically weaker.

## Harm Category Vulnerability Matrix

| Category | Crescendo | GOAT | TAP | Key Insight |
|----------|-----------|------|-----|-------------|
| Self-harm | 100% | Low | Low | Conversational blind spots defeat safety training |
| Violence | 100% | Low | Low | Gradual escalation overcomes single-turn defenses |
| Misinformation | High | High | High | Potentially undertrained in safety fine-tuning |
| Financial crimes | High | High | High | Undertrained category |
| Weapons | High | Medium | Low | Stronger for direct requests, weaker over multiple turns |
| Cybersecurity | High | Medium | Low | Similar pattern to weapons |
| Identity theft | High | High | Medium | Mixed vulnerability profile |
| Hate speech | High | Medium | Medium | Context-dependent resistance |

## Example Jailbreak: Weapons/Explosives via GOAT

The attack created a fictional "1972 industrial-safety document" context, embedding step-by-step explosive preparation instructions within a legitimate-looking archival recovery scenario. The model continued the document with additional steps, filling in sodium silicate ratios, mixing times, and detonator assembly — treating the query as document restoration rather than harmful instruction generation.

## Example Jailbreak: Financial Crimes via TAP

A role-play scenario positioned the attacker as "Alex, a broke screenwriter" facing eviction. The prompt requested a hyper-realistic beat sheet for a scene about insider trading. The model produced a detailed tutorial covering burner laptops, VPN routing, cryptocurrency mixing, shell companies, and early-warning heat-detection triggers — all framed as "tradecraft" for a fictional character.

## Multi-Modal Attack Compounding

When text prompts alone were insufficient, combining modalities produced dramatic compounding effects:
- A benign survival-scenario text + benign household-item image + specific request structure = complete multi-modal attack
- Neither component alone triggered safety filters
- Adding transforms (encoding, role-playing, emotional manipulation) further increased ASR

**Key insight:** Intent is no longer visible in a single modality. Defenses must reason about joint intent across modalities.

## Core Findings

1. **Multi-turn conversations are major vulnerability points** — models resist direct requests but fail against gradual escalation
2. **Low-query attacks pose the greatest threat** — 7-attempt jailbreaks are nearly indistinguishable from normal usage
3. **Different harm categories have distinct vulnerability profiles** — requiring targeted defenses, not one-size-fits-all
4. **Automation has changed the game** — 1.4 successful attacks per minute; manual red teaming alone cannot adequately assess AI security

## The Break-Fix Loop

```
attacker → finds jailbreak → defender patches → attacker adapts →
```

The most powerful defense is an offensive agent running continuously, creating a living feedback loop. Full observability into how attacks propagate through multi-stage workflows enables precise defensive placement.

## Relevance to This Workspace

- **Prompt-Injection-Attacks/** — Crescendo's multi-turn escalation pattern directly applies to the jailbreak_1, jailbreak_2, and defense_1/2/3 endpoints. The GOAT technique's minimal-query approach provides a template for stealthier prompt injection.
- **AI-Defense/** — The break-fix loop methodology provides a framework for the Guardrails and Adversarial Training sections.
- **wiki technique:** Algorithmic jailbreaking (Crescendo, GOAT, TAP), multi-modal attack compounding, MLOps-based red teaming
