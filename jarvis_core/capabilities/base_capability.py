from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseCapability(ABC):
    """Abstract Base Interface for JARVIS OS Capabilities."""

    @abstractmethod
    def can_handle(self, goal: str) -> bool:
        """Determines if this capability can handle the specified user goal."""
        pass

    @abstractmethod
    def plan(self, goal: str) -> Dict[str, Any]:
        """Generate structured execution plan for the goal."""
        pass

    @abstractmethod
    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan using available tools and agents."""
        pass

    @abstractmethod
    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify execution results against goal criteria."""
        pass

    @abstractmethod
    async def repair(self, result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Attempt automated repair if verification failed (up to max 3 attempts)."""
        pass
