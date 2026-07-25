import time
import asyncio
from typing import Dict, Any, List, Optional

from jarvis_core.agents.protocol import AgentTask, AgentResult, MessageBus, AgentPriority
from jarvis_core.agents.base_agent import BaseAgent
from jarvis_core.agents.specialized_agents import (
    PlannerAgent,
    CodingAgent,
    ReviewAgent,
    TestingAgent,
    SecurityAgent,
    ResearchAgent,
    MemoryAgent,
    DocumentationAgent,
    DeploymentAgent
)

class AgentOrchestrator:
    """
    Hierarchical Agent Orchestrator managing agent team registration, task dispatching,
    dependency graph resolution, conflict management, metrics tracking, and automated repair.
    """

    def __init__(self, message_bus: Optional[MessageBus] = None):
        self.message_bus = message_bus or MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        
        # Register default specialized agent team
        self._register_default_team()
        self.execution_history: List[AgentResult] = []
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "repair_attempts": 0,
            "total_duration_ms": 0.0
        }

    def _register_default_team(self):
        default_agents = [
            PlannerAgent(self.message_bus),
            CodingAgent(self.message_bus),
            ReviewAgent(self.message_bus),
            TestingAgent(self.message_bus),
            SecurityAgent(self.message_bus),
            ResearchAgent(self.message_bus),
            MemoryAgent(self.message_bus),
            DocumentationAgent(self.message_bus),
            DeploymentAgent(self.message_bus)
        ]
        for agent in default_agents:
            self.agents[agent.name] = agent

    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent

    def select_agent_for_goal(self, goal: str) -> Optional[BaseAgent]:
        for agent in self.agents.values():
            if agent.can_handle(goal):
                return agent
        return self.agents.get("CodingAgent")

    async def execute_task(self, task: AgentTask) -> AgentResult:
        agent = self.agents.get(task.assigned_agent) or self.select_agent_for_goal(task.goal)
        if not agent:
            return AgentResult(
                task_id=task.task_id,
                agent_name="Orchestrator",
                status="failed",
                output=f"No agent found capable of handling goal: {task.goal}"
            )

        t0 = time.perf_counter()
        result = await agent.execute(task)
        verification = agent.verify(result)

        # Automated Repair Loop (Max 5 attempts)
        attempt = 0
        while not verification["passed"] and attempt < 5:
            attempt += 1
            self.metrics["repair_attempts"] += 1
            result = await agent.repair(result, attempt=attempt)
            verification = agent.verify(result)

        duration = (time.perf_counter() - t0) * 1000.0
        result.duration_ms = round(duration, 2)

        self.execution_history.append(result)
        self.metrics["total_tasks"] += 1
        if result.status == "completed":
            self.metrics["successful_tasks"] += 1
        else:
            self.metrics["failed_tasks"] += 1
        self.metrics["total_duration_ms"] += result.duration_ms

        return result

    async def run_goal(self, goal: str) -> Dict[str, Any]:
        """Runs multi-agent workflow: Planner decomposes goal, then dispatches ready subtasks."""
        planner = self.agents.get("PlannerAgent")
        plan = planner.plan(goal) if planner else {"subtasks": []}
        
        results = []
        for st in plan.get("subtasks", []):
            task = AgentTask(
                task_id=st["id"],
                goal=st["goal"],
                assigned_agent=st["agent"],
                dependencies=st.get("deps", [])
            )
            res = await self.execute_task(task)
            results.append(res)

        all_success = all(r.status == "completed" for r in results)
        return {
            "goal": goal,
            "status": "completed" if all_success else "failed",
            "subtasks_executed": len(results),
            "results": [r.__dict__ for r in results],
            "metrics": self.metrics
        }
