import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from runtime.runtime_engine import RuntimeEngine

app = FastAPI(
    title="JARVIS AI Operating System API",
    description="Production REST backend interfacing runtime engine, executive planner, and ai_core inference.",
    version="1.0.0"
)

runtime = RuntimeEngine()

class GoalRequest(BaseModel):
    goal: str
    session_id: Optional[str] = "default_session"

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "JARVIS AI OS", "version": "1.0.0"}

@app.get("/api/v1/system/status")
def system_status():
    return {
        "plugins": runtime.plugin_registry.list_plugins(),
        "state_summary": runtime.state_manager.get_state_summary(),
        "event_bus_history_len": len(runtime.event_bus.get_history())
    }

@app.post("/api/v1/plan")
async def execute_plan(request: GoalRequest):
    if not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")
    res = await runtime.handle_user_goal(request.goal)
    return res

@app.get("/api/v1/memory")
def get_memory(limit: int = 10):
    experiences = runtime.memory.get_experiences(limit=limit)
    return {"count": len(experiences), "experiences": experiences}
