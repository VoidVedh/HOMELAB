from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BasePlugin(ABC):
    """Unified Base Interface for all JARVIS tools/plugins."""

    @abstractmethod
    def name(self) -> str:
        """Unique plugin tool identifier."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of tool capabilities."""
        pass

    @abstractmethod
    def permissions(self) -> List[str]:
        """List of security permissions required."""
        pass

    def validate(self, payload: Dict[str, Any]) -> bool:
        """Validate payload parameters prior to execution."""
        return isinstance(payload, dict)

    @abstractmethod
    async def run(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool action with given payload."""
        pass

    async def execute(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validated tool execution wrapper."""
        if not self.validate(payload):
            return {"error": f"Invalid payload schema for tool '{self.name()}'"}
        return await self.run(action, payload)
