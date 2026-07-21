from typing import Dict, Any, List, Optional
from jarvis_core.agent_manager.base_agent import BaseAgent
from jarvis_core.agent_manager.coding_agent import CodingAgent

class AgentManager:
    """Agent Registry and Orchestrator routing tasks to specialized agents."""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        # Register default coding agent
        self.register_agent(CodingAgent())

    def register_agent(self, agent: BaseAgent):
        self._agents[agent.name] = agent

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    async def dispatch_task(self, agent_name: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        if not agent:
            return {"error": f"Agent '{agent_name}' not registered."}
        return await agent.execute_step(task_input)
