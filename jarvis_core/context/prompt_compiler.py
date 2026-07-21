from typing import Dict, Any, Optional
from jarvis_core.context.context_manager import ContextManager

class PromptCompiler:
    """Compiles structured prompts: [SYSTEM] -> [CONVERSATION] -> [RAG DOCS] -> [STATE] -> [USER]."""

    def __init__(self, context_manager: Optional[ContextManager] = None):
        self.context_manager = context_manager or ContextManager()

    def compile(
        self,
        system_instruction: str,
        user_query: str,
        rag_context: str = "",
        os_state: Optional[Dict[str, Any]] = None,
        conversation_history: str = ""
    ) -> str:
        budgets = self.context_manager.allocate_budgets()

        # Truncate RAG context if needed
        rag_snippet = rag_context[:budgets["rag_max"]] if rag_context else "None"
        
        # State snippet
        state_str = str(os_state)[:budgets["state_max"]] if os_state else "None"

        compiled = (
            f"<sys>{system_instruction[:budgets['system_max']]}</sys>\n"
            f"[RELEVANT_KNOWLEDGE]\n{rag_snippet}\n"
            f"[SYSTEM_STATE]\n{state_str}\n"
            f"[CONVERSATION]\n{conversation_history}\n"
            f"<usr>{user_query[:budgets['user_max']]}</usr>\n"
            f"<bot>"
        )
        return compiled
