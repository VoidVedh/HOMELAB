import pytest
import asyncio
from jarvis_core.sandbox.policy import ExecutionPolicy
from jarvis_core.sandbox.result import ExecutionResult
from jarvis_core.sandbox.python_sandbox import PythonSandbox
from jarvis_core.sandbox.bash_sandbox import BashSandbox

@pytest.mark.asyncio
async def test_python_sandbox_successful_execution():
    sandbox = PythonSandbox()
    code = "print('Hello from Python Sandbox')\n"
    res = await sandbox.execute(code)

    assert res.is_success == True
    assert res.exit_code == 0
    assert "Hello from Python Sandbox" in res.stdout
    assert res.duration_ms > 0.0

@pytest.mark.asyncio
async def test_bash_sandbox_successful_execution():
    sandbox = BashSandbox()
    cmd = "echo 'JARVIS Sandbox OK'"
    res = await sandbox.execute(cmd)

    assert res.is_success == True
    assert res.exit_code == 0
    assert "JARVIS Sandbox OK" in res.stdout

@pytest.mark.asyncio
async def test_sandbox_timeout_enforcement():
    sandbox = BashSandbox()
    policy = ExecutionPolicy(timeout=0.5, allowed_commands=["sleep", "bash"])
    cmd = "sleep 3"
    res = await sandbox.execute(cmd, policy=policy)

    assert res.timed_out == True
    assert res.exit_code == -1
    assert res.is_success == False

@pytest.mark.asyncio
async def test_sandbox_command_disallowed_security_check():
    sandbox = BashSandbox()
    policy = ExecutionPolicy(allowed_commands=["python3"])
    cmd = "bash -c 'echo test'"
    res = await sandbox.execute(cmd, policy=policy)

    assert res.is_success == False
    assert "disallowed" in res.stderr.lower() or res.exception is not None
