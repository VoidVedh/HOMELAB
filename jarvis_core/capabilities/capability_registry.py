from typing import List, Optional
from jarvis_core.capabilities.base_capability import BaseCapability
from jarvis_core.capabilities.code_capability import CodeCapability

class CapabilityRegistry:
    """Registry matching user goals to capable execution capabilities."""

    def __init__(self):
        self._capabilities: List[BaseCapability] = []
        # Register default code capability
        self.register(CodeCapability())

    def register(self, capability: BaseCapability):
        self._capabilities.append(capability)

    def select_capability(self, goal: str) -> Optional[BaseCapability]:
        for cap in self._capabilities:
            if cap.can_handle(goal):
                return cap
        return self._capabilities[0] if self._capabilities else None
