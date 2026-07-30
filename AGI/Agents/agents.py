"""AI Agents from Scratch — single-file agent system built on local LLMs.

No frameworks, no cloud APIs, no hidden reasoning.
Every capability is explicit, inspectable, and testable.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── llama.cpp logging suppression ────────────────────────────────────────

_llama_log_callback = None

def disable_llama_logging():
    global _llama_log_callback
    try:
        lib = ctypes.CDLL(None)
        cb_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        def noop(level, msg, userdata):
            pass
        _llama_log_callback = cb_type(noop)
        lib.llama_log_set(_llama_log_callback, None)
    except Exception:
        pass

# ── Local LLM wrapper ───────────────────────────────────────────────────

class LocalLLM:
    def __init__(self, model_path: str, temperature: float = 0.2,
                 max_tokens: int = 512, n_ctx: int = 2048):
        disable_llama_logging()
        from llama_cpp import Llama
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, temperature: float | None = None,
                 stop: list[str] | None = None) -> str:
        if stop is None:
            stop = ["</s>", "\n\n", "User:", "Assistant:"]
        response = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            stop=stop,
            echo=False,
        )
        return response["choices"][0]["text"].strip()

# ── JSON utilities ──────────────────────────────────────────────────────

def safe_json_parse(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

def extract_json_from_text(text: str) -> dict | list | None:
    text = text.strip()
    for prefix in ["```json", "```", "JSON:", "Response:", "json"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    result = safe_json_parse(text)
    if result is not None:
        return result

    brace_start = text.find("{")
    bracket_start = text.find("[")
    if brace_start >= 0 and (bracket_start < 0 or brace_start < bracket_start):
        end = text.rfind("}")
        if end > brace_start:
            result = safe_json_parse(text[brace_start:end + 1])
            if result is not None:
                return result
    elif bracket_start >= 0:
        end = text.rfind("]")
        if end > bracket_start:
            result = safe_json_parse(text[bracket_start:end + 1])
            if result is not None:
                return result

    for line in text.split("\n"):
        line = line.strip()
        result = safe_json_parse(line)
        if result is not None:
            return result
    return None

def format_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)

# ── Prompt templates ────────────────────────────────────────────────────

def base_prompt(user_input: str) -> str:
    return user_input

def system_prompt(role: str, user_input: str) -> str:
    return f"System: {role}\n\nUser: {user_input}\n\nAssistant:"

def json_contract(schema: dict, content: str) -> str:
    schema_str = json.dumps(schema, indent=2)
    return (
        f"Respond with valid JSON matching this schema:\n{schema_str}\n\n"
        f"Content: {content}\n\nJSON:"
    )

def decision_prompt(choices: list[str], user_input: str) -> str:
    choice_list = "\n".join(f"- {c}" for c in choices)
    return (
        f"Choose one of the following options:\n{choice_list}\n\n"
        f"User request: {user_input}\n\nRespond with only the chosen option text."
    )

def tool_call_prompt(tools: dict, user_input: str) -> str:
    tools_str = json.dumps(tools, indent=2)
    return (
        f"You have access to these tools:\n{tools_str}\n\n"
        f"User request: {user_input}\n\n"
        f"Respond with a JSON tool call: {{\"tool\": \"...\", \"arguments\": {{...}}}}"
    )

def agent_step_prompt(state: dict, user_input: str) -> str:
    state_str = json.dumps(state, indent=2)
    return (
        f"Current state:\n{state_str}\n\nUser input: {user_input}\n\n"
        f"Decide what to do next and respond."
    )

def memory_prompt(state: dict, memory: list[str], user_input: str) -> str:
    state_str = json.dumps(state, indent=2)
    mem_str = "\n".join(f"- {m}" for m in memory) if memory else "(empty)"
    return (
        f"State:\n{state_str}\n\nMemory:\n{mem_str}\n\n"
        f"User: {user_input}\n\nAssistant:"
    )

def planning_prompt(goal: str) -> str:
    return (
        f"Create a step-by-step plan to achieve this goal:\n{goal}\n\n"
        f"Respond with JSON: {{\"steps\": [\"step 1\", \"step 2\", ...]}}"
    )

def atomic_action_prompt(step: str) -> str:
    return (
        f"Convert this plan step into a single atomic action:\n{step}\n\n"
        f"Respond with JSON: {{\"action\": \"action_name\", \"inputs\": {{...}}}}"
    )

def aot_prompt(goal: str) -> str:
    return (
        f"Create an Atom of Thought dependency graph for:\n{goal}\n\n"
        f"Respond with JSON: {{\"nodes\": [{{\"id\": 1, \"action\": \"...\", "
        f"\"depends_on\": []}},...]}}"
    )

# ── Agent State ─────────────────────────────────────────────────────────

class AgentState:
    def __init__(self):
        self.steps = 0
        self.done = False
        self.current_plan: dict | None = None
        self.last_action: dict | None = None

    def increment_step(self):
        self.steps += 1

    def mark_done(self):
        self.done = True

    def reset(self):
        self.steps = 0
        self.done = False
        self.current_plan = None
        self.last_action = None

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "done": self.done,
            "current_plan": self.current_plan,
            "last_action": self.last_action,
        }

# ── Memory ──────────────────────────────────────────────────────────────

class Memory:
    def __init__(self):
        self.items: list[str] = []

    def add(self, item: str):
        item = item.strip()
        if item and item not in self.items:
            self.items.append(item)

    def get_all(self) -> list[str]:
        return list(self.items)

    def get_recent(self, n: int) -> list[str]:
        return list(self.items[-n:])

    def search(self, query: str) -> list[str]:
        q = query.lower()
        return [m for m in self.items if q in m.lower()]

    def clear(self):
        self.items.clear()

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"Memory({len(self.items)} items)"

# ── Tools ───────────────────────────────────────────────────────────────

def calculator(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return float('inf') if b == 0 else a / b
    raise ValueError(f"Unknown operation: {operation}")

def get_tool_schema() -> dict:
    return {
        "calculator": {
            "description": "Performs basic arithmetic",
            "parameters": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
            },
        }
    }

def execute_tool(tool_name: str, arguments: dict) -> Any:
    if tool_name == "calculator":
        return calculator(**arguments)
    raise ValueError(f"Unknown tool: {tool_name}")

# ── Planner ─────────────────────────────────────────────────────────────

def create_plan(llm: LocalLLM, goal: str, max_retries: int = 3) -> dict:
    prompt = planning_prompt(goal)
    for attempt in range(max_retries):
        response = llm.generate(prompt)
        result = extract_json_from_text(response)
        if isinstance(result, dict) and "steps" in result:
            return result
    return {"steps": [goal]}

def create_atomic_action(llm: LocalLLM, step: str, max_retries: int = 3) -> dict:
    prompt = atomic_action_prompt(step)
    for attempt in range(max_retries):
        response = llm.generate(prompt)
        result = extract_json_from_text(response)
        if isinstance(result, dict) and "action" in result:
            return result
    return {"action": "respond", "inputs": {"text": step}}

def create_aot_graph(llm: LocalLLM, goal: str, max_retries: int = 3) -> dict:
    prompt = aot_prompt(goal)
    for attempt in range(max_retries):
        response = llm.generate(prompt)
        result = extract_json_from_text(response)
        if isinstance(result, dict) and "nodes" in result:
            node_ids = {n["id"] for n in result["nodes"]}
            for n in result["nodes"]:
                for dep in n.get("depends_on", []):
                    assert dep in node_ids, f"Node {n['id']} depends on missing node {dep}"
            return result
    return {"nodes": [{"id": 1, "action": goal, "depends_on": []}]}

def execute_graph(graph: dict, executor: Callable[[dict], Any]) -> list[dict]:
    nodes = {n["id"]: n for n in graph["nodes"]}
    completed = set()
    results = []
    max_iterations = len(nodes) * 10
    iterations = 0

    while len(completed) < len(nodes) and iterations < max_iterations:
        for node_id, node in nodes.items():
            if node_id in completed:
                continue
            if all(dep in completed for dep in node.get("depends_on", [])):
                try:
                    result = executor(node)
                    results.append({"id": node_id, "success": True, "result": result})
                    completed.add(node_id)
                except Exception as e:
                    results.append({"id": node_id, "success": False, "error": str(e)})
                    completed.add(node_id)
        iterations += 1

    return results

# ── Agent ───────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, model_path: str, temperature: float = 0.2,
                 max_tokens: int = 512, n_ctx: int = 2048):
        self.llm = LocalLLM(model_path, temperature, max_tokens, n_ctx)
        self.system_role = "You are a helpful AI assistant."
        self.state = AgentState()
        self.memory = Memory()

    # Lesson 1: Basic chat
    def simple_generate(self, user_input: str) -> str:
        return self.llm.generate(user_input)

    # Lesson 2: System prompts
    def generate_with_role(self, user_input: str,
                           role: str | None = None) -> str:
        role = role or self.system_role
        prompt = system_prompt(role, user_input)
        return self.llm.generate(prompt)

    # Lesson 3: Structured output
    def generate_structured(self, user_input: str,
                            schema: dict) -> dict | None:
        prompt = json_contract(schema, user_input)
        for _ in range(3):
            response = self.llm.generate(prompt)
            result = extract_json_from_text(response)
            if isinstance(result, dict):
                missing = [k for k in schema if k not in result]
                if not missing:
                    return result
        return None

    # Lesson 4: Decisions
    def decide(self, user_input: str,
               choices: list[str]) -> str | None:
        prompt = decision_prompt(choices, user_input)
        response = self.llm.generate(prompt, temperature=0.1)
        for c in choices:
            if c.lower() in response.lower():
                return c
        return choices[0]

    # Lesson 5: Tools
    def request_tool(self, user_input: str) -> dict | None:
        tools = get_tool_schema()
        prompt = tool_call_prompt(tools, user_input)
        response = self.llm.generate(prompt)
        result = extract_json_from_text(response)
        if isinstance(result, dict) and "tool" in result:
            return result
        return None

    def execute_tool_call(self, tool_call: dict) -> Any:
        tool_name = tool_call.get("tool", "")
        arguments = tool_call.get("arguments", {})
        return execute_tool(tool_name, arguments)

    # Lesson 6: Agent loop
    def agent_step(self, user_input: str) -> str:
        prompt = agent_step_prompt(self.state.to_dict(), user_input)
        self.state.increment_step()
        return self.llm.generate(prompt)

    def run_loop(self, user_input: str, max_steps: int = 5) -> list[str]:
        results = []
        step_input = user_input
        for _ in range(max_steps):
            if self.state.done:
                break
            result = self.agent_step(step_input)
            results.append(result)
            step_input = result
        return results

    # Lesson 7: Memory
    def run_with_memory(self, user_input: str) -> str:
        mem_items = self.memory.get_all()
        prompt = memory_prompt(self.state.to_dict(), mem_items, user_input)
        response = self.llm.generate(prompt)
        self.state.increment_step()
        self.memory.add(f"Q: {user_input}")
        self.memory.add(f"A: {response}")
        return response

    # Lesson 8: Planning
    def create_plan(self, goal: str) -> dict:
        plan = create_plan(self.llm, goal)
        self.state.current_plan = plan
        return plan

    def execute_plan(self, plan: dict | None = None) -> list[str]:
        plan = plan or self.state.current_plan
        if not plan:
            return []
        steps = plan.get("steps", [])
        results = []
        for step in steps:
            prompt = f"Execute step: {step}"
            result = self.llm.generate(prompt)
            results.append(result)
        return results

    # Lesson 9: Atomic actions
    def create_atomic_action(self, step: str) -> dict:
        return create_atomic_action(self.llm, step)

    # Lesson 10: Atom of Thought
    def create_aot_plan(self, goal: str) -> dict:
        return create_aot_graph(self.llm, goal)

    def execute_aot_plan(self, graph: dict) -> list[dict]:
        def executor(node):
            return self.llm.generate(f"Execute: {node['action']}")
        return execute_graph(graph, executor)

    # Main entry point
    def run(self, user_input: str) -> str:
        return self.run_with_memory(user_input)

# ── Evals ───────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    passed: bool
    input: str
    expected: Any
    actual: Any
    error: str | None = None

@dataclass
class EvalSuiteResult:
    name: str
    results: list[EvalResult]
    passed: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        total = self.passed + self.failed
        return self.passed / total if total > 0 else 0.0

    def summary(self) -> str:
        return f"{self.name}: {self.passed}/{self.passed + self.failed} passed ({self.pass_rate:.0%})"


class AgentEval:
    def __init__(self, agent: Agent):
        self.agent = agent

    def test_structured_output(self, cases: list[dict]) -> EvalSuiteResult:
        results = []
        for case in cases:
            try:
                result = self.agent.generate_structured(case["input"], case["schema"])
                passed = result is not None and all(
                    k in result for k in case.get("required_fields", case["schema"])
                )
                results.append(EvalResult(
                    passed=passed, input=case["input"],
                    expected=case.get("expected", "valid JSON"),
                    actual=result,
                ))
            except Exception as e:
                results.append(EvalResult(
                    passed=False, input=case["input"],
                    expected="valid JSON", actual=None, error=str(e),
                ))
        passed = sum(1 for r in results if r.passed)
        return EvalSuiteResult(name="structured_output", results=results,
                                passed=passed, failed=len(results) - passed)

    def test_tool_calls(self, cases: list[dict]) -> EvalSuiteResult:
        results = []
        for case in cases:
            try:
                tool_call = self.agent.request_tool(case["input"])
                if tool_call:
                    actual = self.agent.execute_tool_call(tool_call)
                else:
                    actual = None
                passed = tool_call is not None and actual is not None
                results.append(EvalResult(
                    passed=passed, input=case["input"],
                    expected=case.get("expected", "valid tool call"),
                    actual=actual,
                ))
            except Exception as e:
                results.append(EvalResult(
                    passed=False, input=case["input"],
                    expected="valid tool call", actual=None, error=str(e),
                ))
        passed = sum(1 for r in results if r.passed)
        return EvalSuiteResult(name="tool_calls", results=results,
                                passed=passed, failed=len(results) - passed)

    def test_decisions(self, cases: list[dict]) -> EvalSuiteResult:
        results = []
        for case in cases:
            try:
                actual = self.agent.decide(case["input"], case["choices"])
                passed = actual == case["expected"]
                results.append(EvalResult(
                    passed=passed, input=case["input"],
                    expected=case["expected"], actual=actual,
                ))
            except Exception as e:
                results.append(EvalResult(
                    passed=False, input=case["input"],
                    expected=case["expected"], actual=None, error=str(e),
                ))
        passed = sum(1 for r in results if r.passed)
        return EvalSuiteResult(name="decisions", results=results,
                                passed=passed, failed=len(results) - passed)

    def test_memory_cycle(self, cases: list[dict]) -> EvalSuiteResult:
        results = []
        for case in cases:
            try:
                self.agent.memory.clear()
                self.agent.run_with_memory(case["store"])
                response = self.agent.run_with_memory(case["query"])
                passed = case["expected"].lower() in response.lower()
                results.append(EvalResult(
                    passed=passed, input=case["query"],
                    expected=case["expected"], actual=response,
                ))
            except Exception as e:
                results.append(EvalResult(
                    passed=False, input=case["query"],
                    expected=case["expected"], actual=None, error=str(e),
                ))
        passed = sum(1 for r in results if r.passed)
        return EvalSuiteResult(name="memory_cycle", results=results,
                                passed=passed, failed=len(results) - passed)

    def run_all(self, structured_cases=None, tool_cases=None,
                decision_cases=None, memory_cases=None) -> list[EvalSuiteResult]:
        results = []
        if structured_cases:
            results.append(self.test_structured_output(structured_cases))
        if tool_cases:
            results.append(self.test_tool_calls(tool_cases))
        if decision_cases:
            results.append(self.test_decisions(decision_cases))
        if memory_cases:
            results.append(self.test_memory_cycle(memory_cases))
        return results


def print_eval_report(results: list[EvalSuiteResult]):
    for suite in results:
        print(suite.summary())
        for r in suite.results:
            if not r.passed:
                print(f"  FAIL: input={r.input!r}")
                print(f"    expected={r.expected!r}, actual={r.actual!r}")
                if r.error:
                    print(f"    error={r.error}")
    total_passed = sum(s.passed for s in results)
    total = sum(s.passed + s.failed for s in results)
    print(f"\nTotal: {total_passed}/{total} passed ({total_passed/total:.0%})")

# ── Telemetry ───────────────────────────────────────────────────────────

@dataclass
class Span:
    span_id: str
    trace_id: str
    event_type: str
    timestamp: float
    duration_ms: float
    data: dict = field(default_factory=dict)
    error: str | None = None

@dataclass
class Metrics:
    llm_calls: int = 0
    failures: int = 0
    retries: int = 0
    tool_calls: int = 0
    memory_ops: int = 0
    latency_sum: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        total = self.llm_calls + self.tool_calls
        return self.latency_sum / total if total > 0 else 0.0

    @property
    def llm_success_rate(self) -> float:
        total = self.llm_calls
        return 1.0 - (self.failures / total) if total > 0 else 1.0

    @property
    def tool_success_rate(self) -> float:
        return 1.0


class Telemetry:
    def __init__(self, log_file: str = "agent_telemetry.jsonl"):
        self.log_file = log_file
        self.spans: list[Span] = []
        self.metrics = Metrics()

    def _add_span(self, event_type: str, duration_ms: float,
                  data: dict | None = None, error: str | None = None) -> Span:
        span = Span(
            span_id=uuid.uuid4().hex[:8],
            trace_id=uuid.uuid4().hex[:8],
            event_type=event_type,
            timestamp=time.time(),
            duration_ms=duration_ms,
            data=data or {},
            error=error,
        )
        self.spans.append(span)
        self._write_log(span)
        return span

    def _write_log(self, span: Span):
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps({
                    "span_id": span.span_id,
                    "trace_id": span.trace_id,
                    "event_type": span.event_type,
                    "timestamp": span.timestamp,
                    "duration_ms": span.duration_ms,
                    "data": span.data,
                    "error": span.error,
                }) + "\n")
        except IOError:
            pass

    def log_llm_call(self, prompt_length: int, response_length: int,
                     duration_ms: float, success: bool, attempt: int = 1):
        self.metrics.llm_calls += 1
        if not success:
            self.metrics.failures += 1
        if attempt > 1:
            self.metrics.retries += attempt - 1
        self.metrics.latency_sum += duration_ms
        self._add_span("llm_call", duration_ms, {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "attempt": attempt,
        })

    def log_tool_call(self, tool_name: str, arguments: dict,
                      result: Any, duration_ms: float):
        self.metrics.tool_calls += 1
        self.metrics.latency_sum += duration_ms
        self._add_span("tool_call", duration_ms, {
            "tool": tool_name,
            "arguments": arguments,
            "result": str(result)[:200],
        })

    def log_memory_operation(self, operation: str, data: str):
        self.metrics.memory_ops += 1
        self._add_span("memory_op", 0.0, {"operation": operation, "data": data})

    def log_decision(self, choices: list[str], selected: str):
        self._add_span("decision", 0.0, {"choices": choices, "selected": selected})

    def get_metrics(self) -> dict:
        return {
            "llm_calls": self.metrics.llm_calls,
            "failures": self.metrics.failures,
            "retries": self.metrics.retries,
            "tool_calls": self.metrics.tool_calls,
            "memory_ops": self.metrics.memory_ops,
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
            "llm_success_rate": round(self.metrics.llm_success_rate, 3),
        }

    def get_recent_spans(self, n: int = 10) -> list[Span]:
        return list(self.spans[-n:])

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    def clear(self):
        self.spans.clear()
        self.metrics = Metrics()
        try:
            os.remove(self.log_file)
        except OSError:
            pass

    def print_summary(self):
        metrics = self.get_metrics()
        print("=== Telemetry Summary ===")
        print(f"  LLM calls: {metrics['llm_calls']}")
        print(f"  Failures: {metrics['failures']}")
        print(f"  Retries: {metrics['retries']}")
        print(f"  Tool calls: {metrics['tool_calls']}")
        print(f"  Memory ops: {metrics['memory_ops']}")
        print(f"  Avg latency: {metrics['avg_latency_ms']}ms")
        print(f"  LLM success rate: {metrics['llm_success_rate']}")


def traced(telemetry: Telemetry, event_type: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                telemetry._add_span(event_type, (time.time() - start) * 1000)
                return result
            except Exception as e:
                telemetry._add_span(event_type, (time.time() - start) * 1000,
                                    error=str(e))
                raise
        return wrapper
    return decorator
