from typing import Dict, Any, List, Optional

class ContextManager:
    """Allocates token/character budgets dynamically across prompt sections."""

    def __init__(self, max_prompt_chars: int = 4000):
        self.max_prompt_chars = max_prompt_chars

    def allocate_budgets(self) -> Dict[str, int]:
        # Budget allocation ratios: System 15%, RAG 40%, State/Memory 25%, User 20%
        return {
            "system_max": int(self.max_prompt_chars * 0.15),
            "rag_max": int(self.max_prompt_chars * 0.40),
            "state_max": int(self.max_prompt_chars * 0.25),
            "user_max": int(self.max_prompt_chars * 0.20),
        }
