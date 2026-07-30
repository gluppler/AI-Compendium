# 3 — Offensive Infrastructure

Autonomous malware, defensive countermeasures, reproducible lab environments, and closed-loop red/blue evaluation infrastructure.

## Files

| File | Source | Key Contribution |
|------|--------|-----------------|
| `mine-the-gap-dreadgoad-ares.md` | Blog (Apr 2026) | DreadGOAD (reproducible AD lab) + Ares (7 red agents, 3 blue agents); closed-loop evaluation |
| `llm-powered-amsi-provider.md` | Blog (Dec 2025) | Rust AMSI provider with Claude Sonnet; attacker-vs-defender loop generating ground-truth malicious datasets |
| `lolmil-living-off-land-models.md` | Blog (Oct 2025) | C2-less autonomous malware; Phi-3-mini + ONNX Runtime; Lua post-exploitation; privilege escalation |

## Cross-References to Existing Modules

- **AI-Defense/** — The LLM-powered AMSI provider is a direct implementation of AI-based defensive controls relevant to the Guardrails and Adversarial Training sections.
- **AI-in-InfoSec/** — The LOLMIL malware agent and the malicious script dataset generation connect to the Malware Classification module.
- **Challenges/** — DreadGOAD provides a reproducible AD environment suitable for the pending Doctrine-Studio and Prometheon challenges.

## Key Techniques for Wiki

1. **Closed-loop red/blue agent evaluation** — Attacker ground truth used to score defender reconstruction accuracy (Ares)
2. **LLM-powered runtime detection** — AMSI provider using Claude for script classification at execution time
3. **C2-less autonomous malware** — Local inference (Phi-3-mini) with Lua post-exploitation toolkit; no external API calls
4. **Living Off the Land Models (LOLMIL)** — Using pre-installed ONNX Runtime and NPU-optimized models on victim hosts
5. **Synthetic malicious dataset generation** — Paired attacker-defender interaction producing ground-truth labeled samples
6. **Reproducible AD lab provisioning** — Terraform + SSM + golden AMIs for deterministic multi-forest AD environments
