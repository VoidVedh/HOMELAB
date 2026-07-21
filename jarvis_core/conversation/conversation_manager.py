import time
from typing import Dict, Any, List, Optional
from jarvis_core.memory.memory_manager import ExperienceMemory

class ConversationManager:
    """Manages multi-turn conversation sessions, turn history, and memory persistence."""

    def __init__(self, memory: Optional[ExperienceMemory] = None):
        self.memory = memory or ExperienceMemory()
        self._active_sessions: Dict[str, List[Dict[str, Any]]] = {}

    def add_turn(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = []

        turn = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        self._active_sessions[session_id].append(turn)

        # Persist turn in Experience Memory
        self.memory.add_conversation(session_id=session_id, role=role, content=content)
        return turn

    def get_history(self, session_id: str, max_turns: int = 10) -> List[Dict[str, Any]]:
        history = self._active_sessions.get(session_id, [])
        return history[-max_turns:]

    def format_history_for_prompt(self, session_id: str, max_turns: int = 6) -> str:
        turns = self.get_history(session_id, max_turns=max_turns)
        if not turns:
            return ""

        formatted = []
        for t in turns:
            role = t["role"].capitalize()
            content = t["content"].strip()
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)
