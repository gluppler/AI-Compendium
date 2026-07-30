# AI Agents from Scratch

A progressive, single-file agent system built on local LLMs.
No frameworks, no cloud APIs, no hidden reasoning.

## Requirements

- Python 3.10+
- 8GB+ RAM
- 5-10GB disk space for a GGUF model

## Setup

```bash
pip install llama-cpp-python
```

Download a GGUF model (e.g., Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) and place it
in the `models/` directory.

```bash
python setup_check.py
```

## Quick Start

```python
from agents import Agent

agent = Agent("models/llama-3-8b-instruct.gguf")
print(agent.simple_generate("What is an AI agent?"))
```

## Capabilities (12 Lessons)

| # | Capability | Method | Description |
|---|-----------|--------|-------------|
| 1 | Basic chat | `simple_generate()` | Text in, text out |
| 2 | System prompts | `generate_with_role()` | Role-based behavior |
| 3 | Structured output | `generate_structured()` | JSON schema enforcement |
| 4 | Decision making | `decide()` | Route between choices |
| 5 | Tools | `request_tool()` / `execute_tool_call()` | External capabilities |
| 6 | Agent loop | `run_loop()` | Multi-step observe-decide-act |
| 7 | Memory | `run_with_memory()` | Persistent context |
| 8 | Planning | `create_plan()` / `execute_plan()` | Step-by-step plans |
| 9 | Atomic actions | `create_atomic_action()` | Safe step execution |
| 10 | Atom of Thought | `create_aot_plan()` / `execute_aot_plan()` | Dependency graph execution |
| 11 | Evals | `AgentEval` | Regression testing |
| 12 | Telemetry | `Telemetry` | Runtime observability |

## Structure

- `agents.py` — Single-file agent system (LLM wrapper, prompts, state, memory, tools, planner, evals, telemetry)
- `complete_example.py` — Demo exercising all 12 lessons
- `setup_check.py` — Pre-flight verification
- `lessons/` — Step-by-step lesson markdown files
- `diagrams/` — Architecture diagrams

## Existing Agent Types

- `reflex_agent.py` — Simple condition-action rules
- `model_based_agent.py` — Internal world model
- `goal_based_agent.py` — Search and planning to reach goals
- `utility_based_agent.py` — Maximize expected utility
- `multi_agent_system.py` — Multiple interacting agents
- `learning_agent.py` — Learning from experience
