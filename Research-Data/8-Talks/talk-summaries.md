# Talk Summaries

## 1. From Benchmarks to Breaches: Scaling Offensive Security
**Venue:** Offensive AI Con (OAIC) 2025 | **Date:** October 6, 2025
**Speaker:** Dreadnode team
**Link:** [YouTube](https://www.youtube.com/watch?v=YfvVBbQmPaY)

**Summary:** Presentation on scaling offensive security from research benchmarks to real-world breaches. Covers the progression from GOAD as a training/evaluation environment to production-grade autonomous penetration testing agents. Demonstrates how RL training using GOAD-based evaluations produced agents capable of real-world offensive operations. Preceded the Worlds and PentestJudge publications, establishing the evaluation-first methodology that would become Dreadnode's core research approach.

**Key themes:**
- Transitioning from benchmark performance to operational capability
- The role of reproducible environments (GOAD) in agent development
- Reinforcement learning applications in offensive security
- The Sim2Real gap between simulated and real attack scenarios

**Relevance to workspace:** Provides the foundational philosophy behind the agentic pentesting evaluation approach documented in 1-Agentic-Pentesting/.

---

## 2. Building with AI Rigging Workshop
**Venue:** Pivot Con 2025 | **Date:** May 7, 2025
**Presenter:** Martin Wendiggensen
**Link:** [GitHub](https://github.com/vmsv/pivot2025-llmworkshop/tree/main)

**Summary:** Hands-on workshop teaching participants to build LLM-powered agent systems using the Rigging framework. Covers chat pipelines, tool definition and calling, structured output parsing, and agent loop construction. The workshop materials provide a practical introduction to the same framework used throughout Dreadnode's research (AIRTBench, kerberoasting eval, evals blog).

**Key themes:**
- Rigging fundamentals: generators, chats, tools, pipelines
- Building agent harnesses in Python
- Tool-calling patterns and structured output
- Practical agent architecture

**Relevance to workspace:** The workshop materials can serve as a tutorial for building agent harnesses for workspace challenges. The Rigging patterns demonstrated are directly applicable to solver script architecture.

---

## 3. Ghosts on the Node
**Venue:** SOCON 2024 | **Date:** March 11, 2024
**Presenter:** Dreadnode team
**Link:** [PDF](https://github.com/dreadnode/conferences/blob/main/SOCON_2024/Ghosts%20on%20the%20Node.pdf)

**Summary:** Conference talk on AI security threats — one of Dreadnode's earliest public presentations. Covers the emerging AI attack surface and the security implications of deploying ML systems in production environments. Precedes most of the technical research published subsequently and likely influenced the development of the Crucible CTF platform.

**Key themes:**
- AI/ML attack surface overview
- Security implications of production ML deployments
- Adversarial ML fundamentals

**Relevance to workspace:** Foundational presentation covering the AI security domain that this workspace's modules address in depth. The attack categories likely map to the module structure (evasion, privacy, data attacks, prompt injection).
