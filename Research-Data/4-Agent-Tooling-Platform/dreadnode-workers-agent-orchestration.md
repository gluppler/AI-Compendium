# Dreadnode Workers: The Connective Tissue for Flexible Agent Integration and Orchestration

**Source:** dreadnode.io/research/dreadnode-workers-agent-orchestration
**Author:** Vincent Abruzzo
**Date:** May 7, 2026

## Overview

Workers are long-running background processes that live inside Dreadnode capabilities alongside agents, tools, and prompts. They handle use cases that do not fit within a single chat session: event-driven pipelines, cron sweeps, external bridges, stateful long-running loops, and multi-agent orchestration.

## The Problem Workers Solve

Current agent platforms are rigid — every integration requires custom code. The workflow needed today may not match the workflow needed next month when the threat surface shifts. Workers provide a primitive that connects to existing tools (C2 frameworks, recon feeds, exploit pipelines, webhooks, internal APIs, custom scripts) with minimal ceremony.

## The Five Worker Decorators

| Decorator | When It Fires |
|-----------|---------------|
| `@worker.on_startup` | Once, before any handlers |
| `@worker.on_shutdown` | Once, on capability reload or runtime stop |
| `@worker.on_event("kind")` | Every time an event of that kind hits the message bus |
| `@worker.every(seconds=…)` / `every(cron=…)` | On a schedule |
| `@worker.task` | Supervised long-running coroutine, restart-on-crash |

State lives on `worker.state`, a plain dict shared across handlers. Use `asyncio.Lock` for concurrent mutation.

## Example: Source Code Analysis Pipeline

Released as a worked example — a single worker (~500 lines of Python) coordinates multiple agents across four stages:

1. **Attack-surface mapper** — Clones repo, maps codebase
2. **Fan-out** — Five specialized agents run in parallel against the map
3. **Final reviewer** — Stitches specialist reports together
4. **Validator** — Spawns a validator per high-severity finding

All coordinated with plain asyncio (semaphore-bounded concurrency, gather for results). Progress events stream onto the message bus — any subscriber (UI, Slack, webhook) can render them live.

## Use Cases Beyond Multi-Agent Orchestration

### External Bridges
```python
@worker.on_event("turn.completed")
async def to_slack(event, client):
    await httpx.post(SLACK_URL, json={"text": summarize(event.payload)})

@worker.on_event("capability.bridge.callback_received")
async def from_slack(event, client):
    session = await client.create_session(...)
    async for _ in client.stream_chat(session_id=session.session_id, ...):
        pass
```

### Cron Sweeps
```python
@worker.every(cron="0 9 * * 1")  # 9am every Monday
async def reeval_against_latest(client):
    await client.publish("eval.requested", {"model": "anthropic/claude-opus-4-7"})
```

### Stateful Loops
```python
@worker.task
async def reader(client):
    async for message in worker.state["ws"]:
        await process(message, client)
```

## Design Philosophy

Workers live inside the capability boundary — they inherit auth, event routing, session management, and tracing from the runtime. No separate service deployment, no new integration scaffolding. The gap between "I want to connect X to my agents" and "X is triggering agent work" should be measured in hours, not weeks.

## Relevance to This Workspace

- **All modules** — The worker pattern provides a template for building automated evaluation pipelines (e.g., re-running all solvers when a new model ships, or fanning out parallel challenge attempts)
- **wiki technique:** Event-driven agent orchestration, multi-agent pipeline coordination, background agent lifecycle management
