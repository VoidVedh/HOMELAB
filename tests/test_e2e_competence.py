import pytest
import asyncio
from jarvis_core.capabilities.capability_registry import CapabilityRegistry
from jarvis_core.capabilities.code_capability import CodeCapability
from jarvis_core.capabilities.file_capability import FileCapability
from jarvis_core.capabilities.terminal_capability import TerminalCapability
from jarvis_core.planner.repair_engine import RepairEngine
from observability.structured_logger import StructuredLogger
from ai_core.evaluator.e2e_harness import E2EHarness

@pytest.mark.asyncio
async def test_capability_pipeline():
    registry = CapabilityRegistry()
    
    code_cap = registry.select_capability("Create a Python REST service")
    assert isinstance(code_cap, CodeCapability)

    file_cap = registry.select_capability("List local directory files")
    assert isinstance(file_cap, FileCapability)

    term_cap = registry.select_capability("Run bash terminal command")
    assert isinstance(term_cap, TerminalCapability)

    plan = file_cap.plan("List local directory files")
    exec_res = await file_cap.execute(plan)
    ver_res = file_cap.verify(exec_res)
    assert ver_res["passed"] is True

@pytest.mark.asyncio
async def test_repair_engine():
    engine = RepairEngine(max_attempts=3)
    
    async def mock_exec():
        return {"status": "success"}

    res = await engine.attempt_repair("task_001", "Syntax error on line 5", 1, mock_exec)
    assert res["repaired"] is True
    assert res["attempt"] == 1

def test_structured_logger(tmp_path):
    logger = StructuredLogger(log_dir=str(tmp_path))
    record = logger.log_execution(
        goal="Test Goal",
        task_id="t1",
        plan_steps=2,
        tools_used=["filesystem"],
        duration_ms=10.5,
        success=True
    )
    assert record["success"] is True

@pytest.mark.asyncio
async def test_e2e_harness(tmp_path):
    harness = E2EHarness(tmp_dir=str(tmp_path))
    results = await harness.run_e2e_eval()
    assert results["pass_rate"] == 1.0
    assert results["passed_count"] == 3
