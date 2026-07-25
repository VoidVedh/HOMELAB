import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

from runtime.runtime_engine import RuntimeEngine
from ai_core.registry.model_registry import ModelRegistry
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.inference.engine import InferenceEngine
from ai_core.inference.streamer import TokenStreamer
from ai_core.evaluator.regression_framework import ModelRegressionFramework

app = FastAPI(
    title="JARVIS AI Operating System API",
    description="Production REST & SSE Streaming backend interfacing runtime engine, executive planner, and ai_core inference.",
    version="0.5.0"
)

runtime = RuntimeEngine()

# Initialize AI Core Inference Engine
model_path = "checkpoints/jarvis0/jarvis0_best.pt"
vocab_path = "checkpoints/bpe_vocab.json"

if os.path.exists(vocab_path):
    tokenizer = BPETokenizer.load(vocab_path)
else:
    tokenizer = BPETokenizer()
    tokenizer.train("JARVIS AI OS API", vocab_size=4096)

if os.path.exists(model_path):
    model = ModelRegistry.load("jarvis0", config_dir="config", checkpoint_path=model_path)
else:
    model = ModelRegistry.load("jarvis0", config_dir="config")

inference_engine = InferenceEngine(model=model, tokenizer=tokenizer)

class GoalRequest(BaseModel):
    goal: str
    session_id: Optional[str] = "default_session"

class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 1.1

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "JARVIS AI OS", "version": "0.5.0"}

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

@app.post("/api/v1/infer")
def generate_completion(req: CompletionRequest):
    output = inference_engine.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        repetition_penalty=req.repetition_penalty,
        use_kv_cache=True
    )
    return {
        "output": output,
        "metrics": inference_engine.last_metrics
    }

@app.post("/api/v1/infer/stream")
def generate_completion_stream(req: CompletionRequest):
    def token_generator():
        streamer = TokenStreamer()
        for tok_text in inference_engine.generate_stream(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            use_kv_cache=True,
            streamer=streamer
        ):
            yield f"data: {json.dumps({'token': tok_text})}\n\n"
        yield f"data: {json.dumps({'metrics': inference_engine.last_metrics})}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.get("/api/v1/benchmarks")
def get_benchmark_report():
    regression = ModelRegressionFramework(history_dir="observability/benchmarks")
    past_runs = regression.load_historical_benchmark_runs()
    latest_run = past_runs[-1] if past_runs else None
    comp = regression.compare_with_previous_run(latest_run) if latest_run else {"status": "NO_RUNS"}
    return {
        "total_benchmark_runs": len(past_runs),
        "latest_run": latest_run,
        "regression_status": comp
    }

@app.get("/api/v1/memory")
def get_memory(limit: int = 10):
    experiences = runtime.memory.get_experiences(limit=limit)
    return {"count": len(experiences), "experiences": experiences}

# Agent Framework REST API Extensions
from jarvis_core.agents.orchestrator import AgentOrchestrator
from jarvis_core.agents.protocol import AgentTask, AgentPriority

agent_orchestrator = AgentOrchestrator()

class AgentGoalRequest(BaseModel):
    goal: str

class AgentTaskRequest(BaseModel):
    task_id: str
    goal: str
    assigned_agent: Optional[str] = "CodingAgent"
    payload: Optional[Dict[str, Any]] = None

@app.post("/api/v1/agent/run")
async def run_agent_goal(req: AgentGoalRequest):
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")
    res = await agent_orchestrator.run_goal(req.goal)
    return res

@app.post("/api/v1/agent/task")
async def execute_agent_task(req: AgentTaskRequest):
    task = AgentTask(
        task_id=req.task_id,
        goal=req.goal,
        assigned_agent=req.assigned_agent or "CodingAgent",
        payload=req.payload or {}
    )
    res = await agent_orchestrator.execute_task(task)
    return res.__dict__

@app.get("/api/v1/agent/status")
def get_agent_status():
    return {
        "agents_registered": list(agent_orchestrator.agents.keys()),
        "active_metrics": agent_orchestrator.metrics
    }

@app.get("/api/v1/agent/history")
def get_agent_history(limit: int = 10):
    history = [r.__dict__ for r in agent_orchestrator.execution_history[-limit:]]
    return {"count": len(history), "history": history}

@app.get("/api/v1/agent/metrics")
def get_agent_metrics():
    return agent_orchestrator.metrics
