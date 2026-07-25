from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from jarvis_core.agents.protocol import AgentTask, AgentResult, AgentMessage, MessageBus

class BaseAgent(ABC):
    """Abstract interface for all specialized agents in the multi-agent collaboration framework."""

    def __init__(self, name: str, message_bus: Optional[MessageBus] = None):
        self.name = name
        self.message_bus = message_bus
        if self.message_bus:
            self.message_bus.subscribe(self.name, self._handle_message)

    def _handle_message(self, message: AgentMessage):
        """Callback handling incoming messages from the MessageBus."""
        pass

    @abstractmethod
    def can_handle(self, goal: str) -> bool:
        """Returns True if the agent is specialized to handle the goal."""
        pass

    @abstractmethod
    def plan(self, goal: str) -> Dict[str, Any]:
        """Generates task execution plan."""
        pass

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Executes assigned agent task asynchronously."""
        pass

    @abstractmethod
    def verify(self, result: AgentResult) -> Dict[str, Any]:
        """Verifies task result for correctness and quality."""
        pass

    @abstractmethod
    async def repair(self, result: AgentResult, attempt: int) -> AgentResult:
        """Performs critique and repair loop on failed task results."""
        pass

    def handoff(self, target_agent: str, task: AgentTask, context: Dict[str, Any]) -> AgentMessage:
        """Transfers task context and ownership to another specialized agent."""
        msg = AgentMessage(
            message_id=f"handoff_{task.task_id}_{self.name}_to_{target_agent}",
            sender=self.name,
            recipient=target_agent,
            message_type="handoff",
            payload={"task_id": task.task_id, "goal": task.goal, "context": context}
        )
        if self.message_bus:
            self.message_bus.publish(msg)
        return msg

    def report(self, result: AgentResult) -> Dict[str, Any]:
        """Generates structured status and metric report."""
        return {
            "agent_name": self.name,
            "task_id": result.task_id,
            "status": result.status,
            "duration_ms": result.duration_ms,
            "output_length": len(result.output),
            "artifacts": result.artifacts
        }
