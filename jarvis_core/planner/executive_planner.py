import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

from bus.event_bus import EventBus
from bus.events import TaskSubmittedEvent, TaskCompletedEvent, MemoryStoredEvent
from plugins.plugin_registry import PluginRegistry
from jarvis_core.memory.memory_manager import ExperienceMemory
from ai_core.inference.engine import InferenceEngine

@dataclass
class TaskNode:
    task_id: str
    action: str
    payload: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None

class TaskDAG:
    """Directed Acyclic Graph (DAG) for task dependency management and execution."""
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}

    def add_node(self, node: TaskNode):
        self.nodes[node.task_id] = node

    def get_ready_tasks(self) -> List[TaskNode]:
        ready = []
        for node in self.nodes.values():
            if node.status == "pending":
                # Check if all dependencies are completed
                deps_met = all(
                    self.nodes[dep_id].status == "completed"
                    for dep_id in node.dependencies
                    if dep_id in self.nodes
                )
                if deps_met:
                    ready.append(node)
        return ready

    def is_complete(self) -> bool:
        return all(node.status in ("completed", "failed") for node in self.nodes.values())

class ExecutivePlanner:
    """Executive Planner with explicit DAG task decomposition and dependency management."""

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

    def build_dag_for_goal(self, goal: str) -> TaskDAG:
        dag = TaskDAG()
        # Create a structured DAG with dependencies
        n1 = TaskNode(
            task_id="step_1_inspect",
            action="filesystem:list_dir",
            payload={"path": "."},
            dependencies=set()
        )
        n2 = TaskNode(
            task_id="step_2_execute",
            action="terminal:exec",
            payload={"command": "echo 'JARVIS DAG Step 2 Executed'"},
            dependencies={"step_1_inspect"}
        )
        n3 = TaskNode(
            task_id="step_3_log",
            action="filesystem:write_file",
            payload={"path": "logs/plan_execution.log", "content": f"Executed goal: {goal}\n"},
            dependencies={"step_2_execute"}
        )
        dag.add_node(n1)
        dag.add_node(n2)
        dag.add_node(n3)
        return dag

    async def execute_goal(self, goal: str) -> Dict[str, Any]:
        parent_task_id = str(uuid.uuid4())[:8]
        await self.event_bus.publish(TaskSubmittedEvent(sender="ExecutivePlanner", task_id=parent_task_id, goal=goal))

        dag = self.build_dag_for_goal(goal)
        executed_steps = []

        while not dag.is_complete():
            ready_nodes = dag.get_ready_tasks()
            if not ready_nodes:
                break

            for node in ready_nodes:
                node.status = "in_progress"
                
                # Execute action via plugin registry
                plugin_name, action_name = node.action.split(":", 1)
                plugin = self.plugin_registry.get(plugin_name)
                
                if plugin:
                    res = await plugin.run(action_name, node.payload)
                    node.result = res
                    node.status = "completed"
                else:
                    node.result = {"error": f"Plugin {plugin_name} not found"}
                    node.status = "failed"

                executed_steps.append({
                    "step_id": node.task_id,
                    "action": node.action,
                    "status": node.status,
                    "result": node.result
                })

        # Record in Experience Memory
        self.memory.log_experience(
            task_id=parent_task_id,
            goal=goal,
            decision=f"Executed {len(executed_steps)} DAG nodes with dependency constraints.",
            reason="Autonomous DAG Planning Execution",
            files_modified=["logs/plan_execution.log"],
            commands_executed=["echo 'JARVIS DAG Step 2 Executed'"]
        )

        result = {
            "parent_task_id": parent_task_id,
            "goal": goal,
            "dag_status": "completed" if dag.is_complete() else "partial",
            "executed_steps": executed_steps
        }

        await self.event_bus.publish(TaskCompletedEvent(sender="ExecutivePlanner", task_id=parent_task_id, result=result))
        return result
