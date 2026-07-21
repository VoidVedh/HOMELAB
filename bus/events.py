import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Event:
    event_type: str
    sender: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class TaskSubmittedEvent(Event):
    def __init__(self, sender: str, task_id: str, goal: str):
        super().__init__(
            event_type="task.submitted",
            sender=sender,
            payload={"task_id": task_id, "goal": goal}
        )

@dataclass
class TaskCompletedEvent(Event):
    def __init__(self, sender: str, task_id: str, result: Dict[str, Any]):
        super().__init__(
            event_type="task.completed",
            sender=sender,
            payload={"task_id": task_id, "result": result}
        )

@dataclass
class MemoryStoredEvent(Event):
    def __init__(self, sender: str, memory_id: str, memory_type: str, content: Dict[str, Any]):
        super().__init__(
            event_type="memory.stored",
            sender=sender,
            payload={"memory_id": memory_id, "memory_type": memory_type, "content": content}
        )

@dataclass
class PluginExecutionEvent(Event):
    def __init__(self, sender: str, plugin_name: str, action: str, result: Dict[str, Any]):
        super().__init__(
            event_type="plugin.execution",
            sender=sender,
            payload={"plugin_name": plugin_name, "action": action, "result": result}
        )
