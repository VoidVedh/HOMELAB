from typing import Dict, List, Optional
from plugins.base_plugin import BasePlugin

class PluginRegistry:
    """Registry for discovering, retrieving, and executing modular plugins."""

    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}

    def register(self, plugin: BasePlugin):
        self._plugins[plugin.name()] = plugin

    def get(self, name: str) -> Optional[BasePlugin]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": p.name(),
                "description": p.description(),
                "permissions": p.permissions()
            }
            for p in self._plugins.values()
        ]
