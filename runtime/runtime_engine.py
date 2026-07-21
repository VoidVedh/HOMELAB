import asyncio
from typing import Dict, Any, Optional

from bus.event_bus import EventBus
from plugins.plugin_registry import PluginRegistry
from plugins.filesystem_plugin.filesystem_plugin import FilesystemPlugin
from plugins.terminal_plugin.terminal_plugin import TerminalPlugin
from jarvis_core.memory.memory_manager import ExperienceMemory
from jarvis_core.knowledge.indexer import KnowledgeIndex
from jarvis_core.knowledge.file_watcher import FileWatcher
from jarvis_core.context.context_manager import ContextManager
from jarvis_core.context.prompt_compiler import PromptCompiler
from jarvis_core.state.state_manager import StateManager
from jarvis_core.planner.executive_planner import ExecutivePlanner
from jarvis_core.agent_manager.agent_manager import AgentManager
from jarvis_core.rag.retriever import Retriever
from jarvis_core.rag.context_builder import ContextBuilder

class RuntimeEngine:
    """Production Runtime Worker orchestrating OS services, EventBus, Planner, and ai_core inference."""

    def __init__(self, db_path: str = "memory.db", workspace_dir: str = "."):
        self.event_bus = EventBus()
        self.memory = ExperienceMemory(db_path=db_path)
        self.knowledge_index = KnowledgeIndex()
        self.file_watcher = FileWatcher(knowledge_index=self.knowledge_index)
        self.file_watcher.scan_directory(workspace_dir)

        self.plugin_registry = PluginRegistry()
        self.plugin_registry.register(FilesystemPlugin())
        self.plugin_registry.register(TerminalPlugin())

        self.context_manager = ContextManager()
        self.prompt_compiler = PromptCompiler(context_manager=self.context_manager)
        self.state_manager = StateManager()

        self.retriever = Retriever(knowledge_index=self.knowledge_index)
        self.context_builder = ContextBuilder()

        self.agent_manager = AgentManager()
        self.planner = ExecutivePlanner(
            event_bus=self.event_bus,
            plugin_registry=self.plugin_registry,
            memory=self.memory
        )

    async def handle_user_goal(self, goal: str) -> Dict[str, Any]:
        self.state_manager.set_goal(goal)
        
        # 1. RAG retrieval over workspace
        ranked_chunks = self.retriever.retrieve(query=goal, top_k=3)
        rag_context = self.context_builder.build_context_snippet(ranked_chunks)

        # 2. Compile prompt
        compiled_prompt = self.prompt_compiler.compile(
            system_instruction="You are JARVIS AI OS.",
            user_query=goal,
            rag_context=rag_context,
            os_state=self.state_manager.get_state_summary()
        )

        # 3. Execute goal via Planner DAG
        result = await self.planner.execute_goal(goal)
        return {
            "result": result,
            "rag_chunks_found": len(ranked_chunks),
            "compiled_prompt_sample": compiled_prompt[:150] + "..."
        }
