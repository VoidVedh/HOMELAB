import uuid
import asyncio
from typing import Dict, Any, List, Optional

from bus.event_bus import EventBus
from bus.events import TaskSubmittedEvent, TaskCompletedEvent, MemoryStoredEvent
from plugins.plugin_registry import PluginRegistry
from jarvis_core.memory.memory_manager import ExperienceMemory
from ai_core.inference.engine import InferenceEngine

class ExecutivePlanner:
    """Executive Planner managing goal decomposition, agent execution loops, and event-driven logging."""

    def __init__(
        self,
        event_bus: EventBus,
        plugin_registry: PluginRegistry,
        memory: ExperienceMemory,
        inference_engine: Optional[InferenceEngine] = None
    ):
        self.event_bus = event_bus
        self.plugin_registry = plugin_registry
        self.memory = memory
        self.inference_engine = inference_engine

    async def execute_goal(self, goal: str) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())[:8]
        await self.event_bus.publish(TaskSubmittedEvent(sender="ExecutivePlanner", task_id=task_id, goal=goal))

        # 1. Goal Decomposition (Using inference_engine if available or standard heuristics)
        if self.inference_engine:
            prompt = f"<sys>Break down goal into 2 sub-tasks:</sys><usr>{goal}</usr><bot>"
            plan_response = self.inference_engine.generate(prompt=prompt, max_new_tokens=40)
        else:
            plan_response = f"Plan for task {task_id}: Step 1. Inspect environment, Step 2. Execute plan."

        # 2. Log Experience
        self.memory.log_experience(
            task_id=task_id,
            goal=goal,
            decision=f"Decomposed goal into execution steps: {plan_response[:60]}...",
            reason="Autonomous planning pipeline execution",
            files_modified=[],
            commands_executed=[]
        )

        result = {
            "task_id": task_id,
            "goal": goal,
            "status": "completed",
            "plan_output": plan_response
        }

        await self.event_bus.publish(TaskCompletedEvent(sender="ExecutivePlanner", task_id=task_id, result=result))
        return result
