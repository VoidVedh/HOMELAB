import time
from typing import Dict, Any, List

class StateManager:
    """Live tracker for OS state: Current Goal, Current Task, Completed Tasks, Files Changed, Failures."""

    def __init__(self):
        self.current_goal: str = ""
        self.current_task: str = ""
        self.completed_tasks: List[str] = []
        self.pending_tasks: List[str] = []
        self.files_changed: List[str] = []
        self.failures: List[Dict[str, Any]] = []

    def set_goal(self, goal: str):
        self.current_goal = goal
        self.completed_tasks.clear()
        self.pending_tasks.clear()
        self.failures.clear()

    def update_task(self, task_id: str, status: str):
        self.current_task = task_id
        if status == "completed" and task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)

    def record_file_change(self, file_path: str):
        if file_path not in self.files_changed:
            self.files_changed.append(file_path)

    def record_failure(self, task_id: str, error: str):
        self.failures.append({"task_id": task_id, "error": error, "timestamp": time.time()})

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "current_goal": self.current_goal,
            "current_task": self.current_task,
            "completed_count": len(self.completed_tasks),
            "pending_count": len(self.pending_tasks),
            "files_changed_count": len(self.files_changed),
            "failure_count": len(self.failures)
        }
