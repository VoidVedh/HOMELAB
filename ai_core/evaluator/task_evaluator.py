import os
import asyncio
from typing import Dict, Any, List
from bus.event_bus import EventBus
from plugins.plugin_registry import PluginRegistry
from plugins.filesystem_plugin.filesystem_plugin import FilesystemPlugin
from plugins.terminal_plugin.terminal_plugin import TerminalPlugin
from jarvis_core.memory.memory_manager import ExperienceMemory
from jarvis_core.planner.executive_planner import ExecutivePlanner

class TaskEvaluator:
    """Evaluates AI Operating System task execution success across practical benchmark tasks."""

    def __init__(self, tmp_dir: str = "tmp_eval"):
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    async def evaluate_task_success(self) -> Dict[str, Any]:
        bus = EventBus()
        registry = PluginRegistry()
        registry.register(FilesystemPlugin())
        registry.register(TerminalPlugin())
        memory = ExperienceMemory(db_path=os.path.join(self.tmp_dir, "eval_memory.db"))
        planner = ExecutivePlanner(event_bus=bus, plugin_registry=registry, memory=memory)

        tasks = [
            {"name": "Create File", "goal": "Create a new text file"},
            {"name": "Modify Code", "goal": "Update configuration file"},
            {"name": "Fix Bug", "goal": "Run diagnostic check"}
        ]

        results = []
        passed_count = 0

        for task in tasks:
            res = await planner.execute_goal(task["goal"])
            passed = (res.get("dag_status") == "completed")
            if passed:
                passed_count += 1
            results.append({
                "task_name": task["name"],
                "passed": passed,
                "parent_task_id": res.get("parent_task_id")
            })

        return {
            "total_tasks": len(tasks),
            "passed_tasks": passed_count,
            "success_rate": round(passed_count / len(tasks), 2),
            "details": results
        }
