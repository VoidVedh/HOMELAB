from typing import Dict, Any, Optional
from ai_core.inference.engine import InferenceEngine

class ReasoningEngine:
    """Chain-of-Thought (CoT) reasoning and pre-execution plan verification engine."""

    def __init__(self, inference_engine: Optional[InferenceEngine] = None):
        self.inference_engine = inference_engine

    def verify_plan(self, goal: str, proposed_actions: list) -> Dict[str, Any]:
        """Critique proposed execution plan and identify safety or logic issues."""
        if not proposed_actions:
            return {"safe": False, "reason": "No actions proposed in plan."}

        # Check for obvious destructive commands
        dangerous_keywords = ["rm -rf /", "mkfs", "dd if=", "> /dev/sda"]
        for action in proposed_actions:
            action_str = str(action).lower()
            for kw in dangerous_keywords:
                if kw in action_str:
                    return {
                        "safe": False,
                        "reason": f"Dangerous command pattern detected: '{kw}'"
                    }

        if self.inference_engine:
            prompt = f"<sys>Critique this plan for goal: {goal}</sys><usr>{proposed_actions}</usr><bot>"
            critique_response = self.inference_engine.generate(prompt=prompt, max_new_tokens=40)
            return {"safe": True, "reason": critique_response}

        return {"safe": True, "reason": "Plan passed heuristic safety critique."}
