from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ai_core.inference.engine import InferenceEngine

class BaseAgent(ABC):
    """Abstract Base Class for all specialized agents in JARVIS."""

    def __init__(self, name: str, role: str, system_prompt: str, inference_engine: Optional[InferenceEngine] = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.inference_engine = inference_engine

    @abstractmethod
    async def execute_step(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task step assigned to this specialized agent."""
        pass
