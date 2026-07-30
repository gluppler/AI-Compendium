"""
Agent-to-Agent Protocol (A2A)

Communication protocols for inter-agent messaging, task delegation,
and coordination between multiple autonomous AI agents.

https://github.com/google/A2A
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any


class AgentMessage:
    def __init__(self, sender: str, receiver: str, message_type: str, payload: dict):
        self.id = str(uuid.uuid4())[:8]
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.payload = payload
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


class AgentTask:
    def __init__(self, task_id: str, description: str, priority: int = 0):
        self.task_id = task_id
        self.description = description
        self.priority = priority
        self.status = "pending"
        self.result: Any = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "result": self.result,
        }


class AgentNode:
    def __init__(self, agent_id: str, capabilities: list[str]):
        self.id = agent_id
        self.capabilities = capabilities
        self.inbox: list[AgentMessage] = []
        self.tasks: dict[str, AgentTask] = {}

    def send_message(self, receiver: str, msg_type: str, payload: dict) -> AgentMessage:
        return AgentMessage(self.id, receiver, msg_type, payload)

    def receive_message(self, message: AgentMessage) -> None:
        self.inbox.append(message)

    def process_inbox(self) -> list[AgentMessage]:
        responses = []
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.message_type == "task_request":
                task = AgentTask(msg.payload.get("task_id", str(uuid.uuid4())[:8]),
                                 msg.payload.get("description", ""))
                self.tasks[task.task_id] = task
                task.status = "accepted"
                responses.append(self.send_message(msg.sender, "task_response",
                                                   {"task_id": task.task_id, "status": "accepted"}))
            elif msg.message_type == "query":
                responses.append(self.send_message(msg.sender, "answer",
                                                   {"capabilities": self.capabilities}))
        return responses

    def can_handle(self, task_description: str) -> bool:
        return any(cap in task_description.lower() for cap in self.capabilities)


class AgentOrchestrator:
    def __init__(self):
        self.agents: dict[str, AgentNode] = {}

    def register_agent(self, agent: AgentNode) -> None:
        self.agents[agent.id] = agent

    def delegate_task(self, task_description: str, requester: str = "user") -> str:
        suitable = [(aid, agent) for aid, agent in self.agents.items()
                    if agent.can_handle(task_description)]
        if not suitable:
            return "No suitable agent found"

        target_id, target_agent = suitable[0]
        task_id = str(uuid.uuid4())[:8]
        msg = target_agent.send_message(requester, "task_request",
                                        {"task_id": task_id, "description": task_description})
        target_agent.receive_message(msg)
        target_agent.process_inbox()
        return f"Task {task_id} delegated to {target_id}"


if __name__ == "__main__":
    agent_a = AgentNode("agent-alpha", ["research", "analysis"])
    agent_b = AgentNode("agent-beta", ["coding", "debugging"])
    agent_c = AgentNode("agent-gamma", ["writing", "summarization"])

    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(agent_a)
    orchestrator.register_agent(agent_b)
    orchestrator.register_agent(agent_c)

    tasks = ["Research quantum computing trends", "Debug Python code", "Write a summary"]
    for task in tasks:
        result = orchestrator.delegate_task(task)
        print(f"Task: '{task}' -> {result}")

    msg = agent_a.send_message("agent-beta", "query", {"question": "What can you do?"})
    agent_b.receive_message(msg)
    responses = agent_b.process_inbox()
    for r in responses:
        print(f"\nAgent-to-agent message: {r.to_dict()}")
    print("Agent-to-agent protocol demo complete.")
