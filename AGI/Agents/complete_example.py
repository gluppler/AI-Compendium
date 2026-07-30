"""Demonstration script exercising all 12 agent capabilities."""

import sys
sys.path.insert(0, ".")

from agents import Agent, AgentEval, Telemetry, print_eval_report


def lesson_01_basic_chat(agent):
    print("\n=== Lesson 1: Basic Chat ===")
    result = agent.simple_generate("Explain what an AI agent is in one sentence.")
    print(f"Agent: {result}")


def lesson_02_with_role(agent):
    print("\n=== Lesson 2: System Prompts ===")
    result = agent.generate_with_role(
        "Explain what an AI agent is.",
        role="You are a teacher explaining to a 10-year-old.",
    )
    print(f"Agent: {result}")


def lesson_03_structured(agent):
    print("\n=== Lesson 3: Structured Output ===")
    schema = {"topic": "string", "difficulty": "string"}
    result = agent.generate_structured("Explain quantum computing", schema)
    print(f"Agent: {result}")


def lesson_04_decisions(agent):
    print("\n=== Lesson 4: Decisions ===")
    choices = ["answer_question", "summarize_text", "translate"]
    result = agent.decide("What is the capital of France?", choices)
    print(f"Decision: {result}")


def lesson_05_tools(agent):
    print("\n=== Lesson 5: Tools ===")
    tool_call = agent.request_tool("What is 42 * 7?")
    if tool_call:
        print(f"Tool call: {tool_call}")
        result = agent.execute_tool_call(tool_call)
        print(f"Result: {result}")
    else:
        print("No tool call generated")


def lesson_06_agent_loop(agent):
    print("\n=== Lesson 6: Agent Loop ===")
    results = agent.run_loop("Help me understand AI agents", max_steps=3)
    for i, r in enumerate(results):
        print(f"  Step {i + 1}: {r[:100]}...")


def lesson_07_memory(agent):
    print("\n=== Lesson 7: Memory ===")
    agent.memory.clear()
    agent.run_with_memory("My name is Alice.")
    result = agent.run_with_memory("What's my name?")
    print(f"Agent: {result}")


def lesson_08_planning(agent):
    print("\n=== Lesson 8: Planning ===")
    plan = agent.create_plan("Write a short blog post about AI")
    print(f"Plan: {plan}")


def lesson_09_atomic_actions(agent):
    print("\n=== Lesson 9: Atomic Actions ===")
    action = agent.create_atomic_action("Research the topic")
    print(f"Action: {action}")


def lesson_10_aot(agent):
    print("\n=== Lesson 10: Atom of Thought ===")
    graph = agent.create_aot_plan("Research and write an article")
    print(f"AoT Graph: {graph}")


def lesson_11_evals(agent):
    print("\n=== Lesson 11: Evals ===")
    telemetry = Telemetry()
    eval_agent = AgentEval(agent)
    results = eval_agent.run_all(
        decision_cases=[
            {"input": "Summarize this text", "choices": ["summarize_text", "translate", "answer_question"], "expected": "summarize_text"},
            {"input": "Translate hello to French", "choices": ["summarize_text", "translate", "answer_question"], "expected": "translate"},
        ],
    )
    print_eval_report(results)


def lesson_12_telemetry(agent):
    print("\n=== Lesson 12: Telemetry ===")
    tel = Telemetry()
    tel.log_llm_call(prompt_length=50, response_length=100, duration_ms=1500, success=True)
    tel.log_tool_call("calculator", {"a": 1, "b": 2, "operation": "add"}, 3.0, 10.0)
    tel.log_memory_operation("add", "test memory")
    tel.print_summary()


def main():
    print("=" * 60)
    print("AI Agents from Scratch — Complete Example")
    print("=" * 60)

    try:
        agent = Agent("models/llama-3-8b-instruct.gguf")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        print("Run with --offline to test without a model.")
        if "--offline" in sys.argv:
            print("Offline mode: testing module imports only.")
            return
        sys.exit(1)

    lesson_map = {
        1: lesson_01_basic_chat,
        2: lesson_02_with_role,
        3: lesson_03_structured,
        4: lesson_04_decisions,
        5: lesson_05_tools,
        6: lesson_06_agent_loop,
        7: lesson_07_memory,
        8: lesson_08_planning,
        9: lesson_09_atomic_actions,
        10: lesson_10_aot,
        11: lesson_11_evals,
        12: lesson_12_telemetry,
    }

    for num in range(1, 13):
        try:
            lesson_map[num](agent)
        except Exception as e:
            print(f"Lesson {num} failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
