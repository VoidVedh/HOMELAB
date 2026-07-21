import pytest
from bus.event_bus import EventBus
from plugins.plugin_registry import PluginRegistry
from plugins.filesystem_plugin.filesystem_plugin import FilesystemPlugin
from jarvis_core.memory.memory_manager import ExperienceMemory
from jarvis_core.planner.executive_planner import ExecutivePlanner

@pytest.mark.asyncio
async def test_executive_planner_run(tmp_path):
    db_path = str(tmp_path / "memory.db")
    memory = ExperienceMemory(db_path=db_path)
    bus = EventBus()
    registry = PluginRegistry()
    registry.register(FilesystemPlugin())

    planner = ExecutivePlanner(event_bus=bus, plugin_registry=registry, memory=memory)
    res = await planner.execute_goal("Build a mini dataset loader")

    assert res["status"] == "completed"
    assert "task_id" in res

    experiences = memory.get_experiences()
    assert len(experiences) == 1
    assert "Build a mini dataset loader" in experiences[0]["goal"]
