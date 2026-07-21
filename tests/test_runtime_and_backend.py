import pytest
import asyncio
from runtime.runtime_engine import RuntimeEngine
from jarvis_core.conversation.conversation_manager import ConversationManager
from jarvis_core.reasoning.reasoning_engine import ReasoningEngine
from jarvis_core.agent_manager.agent_manager import AgentManager

@pytest.mark.asyncio
async def test_runtime_engine_handle_goal(tmp_path):
    db_path = str(tmp_path / "runtime_memory.db")
    engine = RuntimeEngine(db_path=db_path, workspace_dir=str(tmp_path))
    res = await engine.handle_user_goal("Execute environment scan")
    
    assert "result" in res
    assert res["result"]["dag_status"] == "completed"

def test_conversation_and_reasoning(tmp_path):
    conv = ConversationManager()
    conv.add_turn("session_1", "user", "Hello JARVIS")
    conv.add_turn("session_1", "assistant", "Hello! How can I assist you?")
    
    history = conv.get_history("session_1")
    assert len(history) == 2
    
    reasoning = ReasoningEngine()
    safety_check = reasoning.verify_plan("Test Goal", ["filesystem:read_file"])
    assert safety_check["safe"] is True

@pytest.mark.asyncio
async def test_agent_manager_dispatch():
    mgr = AgentManager()
    res = await mgr.dispatch_task("CodingAgent", {"prompt": "Write hello world script"})
    assert res["status"] == "completed"
    assert res["agent"] == "CodingAgent"
