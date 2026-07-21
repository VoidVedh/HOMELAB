import pytest
from jarvis_core.context.context_manager import ContextManager
from jarvis_core.context.prompt_compiler import PromptCompiler
from jarvis_core.state.state_manager import StateManager
from plugins.tool_selector import ToolSelector
from plugins.plugin_registry import PluginRegistry
from plugins.filesystem_plugin.filesystem_plugin import FilesystemPlugin

def test_context_manager_and_prompt_compiler():
    cm = ContextManager(max_prompt_chars=2000)
    compiler = PromptCompiler(context_manager=cm)

    compiled = compiler.compile(
        system_instruction="You are JARVIS AI OS.",
        user_query="Create task plan",
        rag_context="RAG Chunk 1",
        os_state={"goal": "Build OS"}
    )
    assert "<sys>You are JARVIS AI OS.</sys>" in compiled
    assert "[RELEVANT_KNOWLEDGE]" in compiled
    assert "<usr>Create task plan</usr>" in compiled

def test_state_manager_and_tool_selector():
    sm = StateManager()
    sm.set_goal("Test Goal")
    sm.update_task("task_1", "completed")
    summary = sm.get_state_summary()
    assert summary["current_goal"] == "Test Goal"
    assert summary["completed_count"] == 1

    registry = PluginRegistry()
    registry.register(FilesystemPlugin())
    selector = ToolSelector(registry=registry)

    plugin, action, payload = selector.validate_and_select({
        "plugin": "filesystem",
        "action": "list_dir",
        "payload": {"path": "."}
    })
    assert plugin.name() == "filesystem"
    assert action == "list_dir"
