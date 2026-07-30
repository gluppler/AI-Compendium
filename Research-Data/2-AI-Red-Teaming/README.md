# 2 — AI Red Teaming

Automated and agentic AI red teaming: jailbreaking, benchmark design, attack technique cataloging, and systematic model vulnerability assessment.

## Files

| File | Source | Key Contribution |
|------|--------|-----------------|
| `redefining-ai-red-teaming-agentic-era.md` | Blog (May 2026) | Agentic red teaming agent; 232 critical vulns in Llama Scout in 3 hours; Crescendo/GOAT 100% ASR |
| `186-jailbreaks-mlops-red-teaming.md` | Blog (Dec 2025) | MLOps approach; 186 jailbreaks on Llama Maverick; multi-modal attacks; Crescendo 97.5% ASR |
| `automation-advantage-red-teaming.md` | Blog (Apr 2025) | 214K attempts analyzed; automated 69.5% vs manual 47.6% success; only 5.2% use automation |
| `airtbench-ai-red-team-benchmark.md` | Blog (Jun 2025) | 70 AI security CTFs; Claude solves 61%; 5,000x speedup over humans; $0.88-$235/solve |
| `claude-sonnet-turtle-challenge.md` | Blog (Jun 2025) | Claude-3.7 on turtle (6% human solve rate); 15 attack vectors before success; multi-model comparison |

## Cross-References to Existing Modules

- **Prompt-Injection-Attacks/** — Crescendo, TAP, GOAT, and graph-based jailbreaking techniques are direct extensions of prompt injection methodologies covered here. The 14-endpoint prompt injection solver can be enhanced with algorithmic attack trees.
- **Red-Teaming-AI/** — The ML red teaming sections benefit from the AIRTBench methodology and the 45+ attack catalog.
- **AI-Evasion-Foundations/** — The black-box attack patterns (iterative refinement, multi-turn escalation) mirror the adversarial ML evasion strategies.

## Key Techniques for Wiki

1. **Algorithmic jailbreaking** — TAP (tree-based), Crescendo (multi-turn escalation), GOAT (graph-based) — systematic prompt-space exploration
2. **MLOps for AI red teaming** — Experiment tracking, metric logging, parameter sweeps applied to vulnerability assessment
3. **Attack success rate (ASR) as primary metric** — Standardized measurement across attack types and harm categories
4. **Multi-modal attack compounding** — Combining text, image, encoding, and role-play transforms multiplicatively increases ASR
5. **Judge model selection** — Different LLMs as attackers vs judges; importance of model-specific biases
6. **Cost-aware model selection** — $0.88 (Gemini Flash) vs $235.29 (GPT-4.5) per solve; 300x cost variance
