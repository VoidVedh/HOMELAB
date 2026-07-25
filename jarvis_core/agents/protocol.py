import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

class AgentPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class MessageType(Enum):
    PUBLISH = "publish"
    BROADCAST = "broadcast"
    HANDOFF = "handoff"
    RETRY = "retry"
    CANCEL = "cancel"

@dataclass
class AgentMessage:
    message_id: str
    sender: str
    recipient: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentTask:
    task_id: str
    goal: str
    assigned_agent: str
    priority: AgentPriority = AgentPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # "pending", "running", "completed", "failed", "cancelled"
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class AgentResult:
    task_id: str
    agent_name: str
    status: str  # "completed", "failed"
    output: str
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

@dataclass
class ConversationState:
    conversation_id: str
    messages: List[AgentMessage] = field(default_factory=list)
    tasks: Dict[str, AgentTask] = field(default_factory=dict)
    results: Dict[str, AgentResult] = field(default_factory=dict)

class MessageBus:
    """In-memory asynchronous pub-sub and handoff message router for multi-agent coordination."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[AgentMessage], None]]] = {}
        self.history: List[AgentMessage] = []

    def subscribe(self, agent_name: str, callback: Callable[[AgentMessage], None]):
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)

    def publish(self, message: AgentMessage):
        self.history.append(message)
        if message.recipient in self.subscribers:
            for callback in self.subscribers[message.recipient]:
                callback(message)

    def broadcast(self, sender: str, payload: Dict[str, Any]):
        msg = AgentMessage(
            message_id=f"msg_{int(time.time()*1000)}",
            sender=sender,
            recipient="*",
            message_type=MessageType.BROADCAST,
            payload=payload
        )
        self.history.append(msg)
        for recipient, callbacks in self.subscribers.items():
            if recipient != sender:
                for callback in callbacks:
                    callback(msg)
