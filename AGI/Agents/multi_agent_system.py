"""
Multi-Agent System

Coordination and communication between multiple autonomous agents.
Includes implementations of common multi-agent patterns: cooperative,
competitive, and swarm-based coordination.

https://en.wikipedia.org/wiki/Multi-agent_system
"""

from __future__ import annotations

import random


class Agent:
    def __init__(self, agent_id: str, initial_resource: float = 10.0):
        self.id = agent_id
        self.resource = initial_resource
        self.knowledge: dict = {}

    def share(self) -> dict:
        return {"id": self.id, "resource": self.resource}

    def receive(self, message: dict) -> None:
        self.knowledge.update(message)


class SwarmAgent(Agent):
    def __init__(self, agent_id: str, x: float, y: float):
        super().__init__(agent_id)
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)

    def update_velocity(
        self, neighbors: list[SwarmAgent],
        cohesion_weight: float = 0.01,
        separation_dist: float = 2.0,
        alignment_weight: float = 0.05,
    ) -> None:
        cx, cy, count = 0.0, 0.0, 0
        for other in neighbors:
            if other.id == self.id:
                continue
            dx, dy = other.x - self.x, other.y - self.y
            dist = (dx**2 + dy**2) ** 0.5
            if dist < separation_dist and dist > 0:
                self.vx -= dx / dist
                self.vy -= dy / dist
            cx += other.x
            cy += other.y
            count += 1
        if count > 0:
            self.vx += (cx / count - self.x) * cohesion_weight
            self.vy += (cy / count - self.y) * cohesion_weight

    def move(self) -> None:
        self.x += self.vx
        self.y += self.vy


def contract_net_protocol(
    manager: Agent, tasks: list[str], workers: list[Agent]
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for task in tasks:
        bids = {}
        for worker in workers:
            if worker.resource >= 2:
                bids[worker.id] = worker.resource * random.uniform(0.8, 1.2)
        if bids:
            winner = min(bids, key=bids.get)
            assignments[task] = winner
            for w in workers:
                if w.id == winner:
                    w.resource -= 2
    return assignments


if __name__ == "__main__":
    print("=== Contract Net Protocol Demo ===")
    manager = Agent("manager")
    workers = [Agent(f"worker_{i}") for i in range(3)]
    tasks = ["build_A", "build_B", "build_C"]
    result = contract_net_protocol(manager, tasks, workers)
    print(f"Assignments: {result}")
    for w in workers:
        print(f"  {w.id}: remaining resource = {w.resource:.1f}")

    print("\n=== Swarm Boids Demo ===")
    boids = [SwarmAgent(f"boid_{i}", i * 3.0, i * 2.0) for i in range(4)]
    for _ in range(5):
        for b in boids:
            b.update_velocity(boids)
            b.move()
        positions = [(b.id, round(b.x, 2), round(b.y, 2)) for b in boids]
    print(f"Final positions: {positions}")
    print("Multi-agent system demo complete.")
