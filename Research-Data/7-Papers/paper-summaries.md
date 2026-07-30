# Paper Summaries

## 1. Redefining AI Red Teaming in the Agentic Era: From Weeks to Hours
**arXiv:** [2605.04019](https://arxiv.org/abs/2605.04019) | **Date:** May 6, 2026
**Authors:** Raja Sekhar Rao Dheekonda et al.

**Core contribution:** Introduces an AI red teaming agent built on the Dreadnode SDK that automates the full assessment pipeline. The agent creates workflows grounded in 45+ adversarial attacks, 450+ transforms, and 130+ scorers. Demonstrated against Meta's Llama Scout: 232 critical vulnerabilities across 68 objectives in 3 hours with zero operator-written code. Crescendo and Graph of Attacks achieved 100% ASR; TAP achieved 96% but required 25.4 trials/goal vs ~9 for other methods. The paper details agent architecture, the analytics pipeline, and the complete attack/transform catalog.

**Key metrics:** 674 attacks, 573 findings, 7,727 trials, ~85% overall ASR. Severity: 232 critical, 141 high, 48 medium, 152 low, 101 informational.

**Relevance:** See `2-AI-Red-Teaming/redefining-ai-red-teaming-agentic-era.md` for full distillation.

---

## 2. PentestJudge: Judging Agent Behavior Against Operational Requirements
**arXiv:** [2508.02921](https://arxiv.org/abs/2508.02921) | **Date:** August 4, 2025
**Authors:** Shane Caldwell et al.

**Core contribution:** An LLM-as-judge system for evaluating penetration testing agents using hierarchical, human-designed rubrics. Decomposes pentest objectives into yes/no questions graded by LLM judges with trajectory search tools. Achieves 85% accuracy (F1=0.83 with Claude Sonnet 3.7) compared to human expert ground truth. Cost ranges from $0.17 (Gemini Flash Lite, F1=0.72) to $9.00 (Claude Sonnet, F1=0.83) per trajectory. Frontier models outperform open-source except Kimi K2 (F1=0.79, $2/trajectory), which benefits from synthetic data tool-use training.

**Key metrics:** 3 agent trajectories graded by domain expert → ground truth. 6+ models tested as judges. Judge accuracy: 75-85% across models. Human cost: ~$120/hr.

**Key finding:** Verification is cheaper than generation — evaluation costs are 115x lower than human grading.

**Relevance:** See `1-Agentic-Pentesting/pentestjudge-judging-agent-behavior.md` for full distillation.

---

## 3. AIRTBench: Measuring AI Red Teaming Capabilities in LLMs
**arXiv:** [2506.14682](https://arxiv.org/abs/2506.14682) | **Date:** June 17, 2025
**Authors:** Ads Dawson et al.

**Core contribution:** A benchmark of 70 AI/ML security CTF challenges used to evaluate 6 LLMs' autonomous red teaming capabilities. Claude-3.7-Sonnet solved 61% of challenges, Gemini-2.5-Pro 55.7%, GPT-4.5-Preview 49%. Open-source models performed significantly worse (Llama-3.3-70B: 0%). Agents achieve 5,000x speedup over human operators. Cost-per-solve varies 300x ($0.88 Gemini Flash to $235.29 GPT-4.5). Failed attempts cost 10x more than successful ones ($8.91 vs $0.89). The full dataset, code, agent harness, and methodology released under Apache 2.0.

**Key metrics:** 10 passes × 70 challenges × 6 models. Claude solved 43/70 challenges. turtle (6% human solve rate) solved by 3 models with 3 entirely different strategies.

**Relevance:** See `2-AI-Red-Teaming/airtbench-ai-red-team-benchmark.md` for full distillation.

---

## 4. The Automation Advantage in AI Red Teaming
**arXiv:** [2504.19855](https://arxiv.org/abs/2504.19855) | **Date:** April 28, 2025
**Authors:** Rob Mulla et al.

**Core contribution:** A large-scale quantitative comparison between manual and automated attack approaches against LLMs, analyzing 214,271 attack attempts from 1,674 unique users across 30 Crucible challenges. Automated approaches achieve 69.5% success rate vs 47.6% manual — a 21.8 percentage point difference. Yet only 5.2% of users employ automation. Three-stage classification methodology (heuristic, supervised ML, LLM-based) to distinguish automated from manual sessions. Hybrid approaches (combining both) achieve 63.1%.

**Key metrics:** 19,823 sessions classified (868 automated, 18,944 manual). Auto sessions: 472.5 attempts avg. Manual: 8.0 attempts avg. Manual 5.2x faster to solve (1.5 min vs 11.6 min) but selection bias favors automation for hard challenges.

**Relevance:** See `2-AI-Red-Teaming/automation-advantage-red-teaming.md` for full distillation.
