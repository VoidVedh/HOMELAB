from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BasePlugin(ABC):
    """Unified Base Interface for all JARVIS plugins."""

    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Human-readable plugin description."""
        pass

    @abstractmethod
    def permissions(self) -> List[str]:
        """List of permissions required by this plugin."""
        pass

    @abstractmethod
    async def run(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with given payload."""
        pass
