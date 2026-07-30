# 4 — Agent Tooling & Platform

SDK, frameworks, orchestration primitives, and platform architecture for building and deploying security agents.

## Files

| File | Source | Key Contribution |
|------|--------|-----------------|
| `dreadnode-workers-agent-orchestration.md` | Blog (May 2026) | Workers primitive: 5 decorators, event-driven pipelines, ~500-line multi-agent source code analysis example |
| `from-compute-to-congress-march-agents.md` | Blog (Apr 2026) | Policy analysis of agent non-determinism; PentestJudge as AI-powered governance; NIST RFI response |

## Cross-References to Existing Modules

- **All modules** — The workers pattern provides infrastructure for automated evaluation pipelines applied to any challenge or module.
- **Red-Teaming-AI/** — The discussion of non-deterministic evaluation frameworks directly validates the pass@k approach.

## Key Techniques for Wiki

1. **Event-driven agent orchestration** — Worker decorators for startup, shutdown, event handling, cron scheduling, and supervised tasks
2. **Multi-agent pipeline coordination** — Fan-out/fan-in patterns using plain asyncio with semaphore-bounded concurrency
3. **AI-powered governance** — Using LLM judges (PentestJudge) to evaluate agentic systems at scale
