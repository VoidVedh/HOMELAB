from typing import Dict, Any, Optional
from jarvis_core.agent_manager.base_agent import BaseAgent
from ai_core.inference.engine import InferenceEngine

class CodingAgent(BaseAgent):
    """Specialized Agent for software architecture, code generation, refactoring, and file patching."""

    def __init__(self, inference_engine: Optional[InferenceEngine] = None):
        super().__init__(
            name="CodingAgent",
            role="Software Engineer & Architect",
            system_prompt="You are an expert software engineer specializing in clean Python architecture.",
            inference_engine=inference_engine
        )

    async def execute_step(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = task_input.get("prompt", "")
        if self.inference_engine:
            compiled_prompt = f"<sys>{self.system_prompt}</sys><usr>{prompt_text}</usr><bot>"
            response = self.inference_engine.generate(prompt=compiled_prompt, max_new_tokens=64)
        else:
            response = f"[CodingAgent Output] Processed code request for: {prompt_text}"

        return {
            "agent": self.name,
            "status": "completed",
            "code_output": response
        }
