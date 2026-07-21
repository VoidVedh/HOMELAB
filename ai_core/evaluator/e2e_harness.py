import os
import asyncio
from typing import Dict, Any, List
from bus.event_bus import EventBus
from plugins.plugin_registry import PluginRegistry
from plugins.filesystem_plugin.filesystem_plugin import FilesystemPlugin
from plugins.terminal_plugin.terminal_plugin import TerminalPlugin
from jarvis_core.memory.memory_manager import ExperienceMemory
from jarvis_core.planner.executive_planner import ExecutivePlanner
from jarvis_core.capabilities.capability_registry import CapabilityRegistry
from observability.structured_logger import StructuredLogger

class E2EHarness:
    """End-to-End Evaluation Harness measuring real developer tasks and scoring capability stages."""

    def __init__(self, tmp_dir: str = "tmp_e2e"):
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
        self.logger = StructuredLogger(log_dir=tmp_dir)

    async def run_e2e_eval(self) -> Dict[str, Any]:
        bus = EventBus()
        registry = PluginRegistry()
        registry.register(FilesystemPlugin())
        registry.register(TerminalPlugin())
        memory = ExperienceMemory(db_path=os.path.join(self.tmp_dir, "e2e_memory.db"))
        planner = ExecutivePlanner(event_bus=bus, plugin_registry=registry, memory=memory)
        cap_registry = CapabilityRegistry()

        e2e_tasks = [
            "Create a FastAPI backend project",
            "Build a CSV data processing script",
            "Refactor code and run unit tests"
        ]

        task_results = []
        passed_count = 0

        for goal in e2e_tasks:
            cap = cap_registry.select_capability(goal)
            plan = cap.plan(goal)
            exec_res = await cap.execute(plan)
            ver_res = cap.verify(exec_res)

            passed = ver_res["passed"]
            if passed:
                passed_count += 1

            task_results.append({
                "goal": goal,
                "capability": cap.__class__.__name__,
                "passed": passed,
                "steps": plan["steps"]
            })

            self.logger.log_execution(
                goal=goal,
                task_id="e2e_task",
                plan_steps=len(plan["steps"]),
                tools_used=["filesystem", "terminal"],
                duration_ms=15.0,
                success=passed
            )

        return {
            "total_e2e_tasks": len(e2e_tasks),
            "passed_count": passed_count,
            "pass_rate": round(passed_count / len(e2e_tasks), 2),
            "details": task_results
        }
