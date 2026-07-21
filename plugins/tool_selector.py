from typing import Dict, Any, List, Optional, Tuple
from plugins.plugin_registry import PluginRegistry

class ToolSelector:
    """Selects tools and validates action arguments against defined schemas."""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry

    def validate_and_select(self, tool_call: Dict[str, Any]) -> Tuple[Optional[Any], str, Dict[str, Any]]:
        plugin_name = tool_call.get("plugin")
        action_name = tool_call.get("action")
        payload = tool_call.get("payload", {})

        plugin = self.registry.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_name}' is not registered.")

        # Basic payload validation
        if not isinstance(payload, dict):
            raise TypeError("Tool payload must be a dictionary.")

        return plugin, action_name, payload
