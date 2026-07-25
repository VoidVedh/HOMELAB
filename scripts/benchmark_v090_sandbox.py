#!/usr/bin/env python3
"""
HOMELAB Milestone v0.9.0 Benchmark Runner
Measures execution latency, sandbox overhead, repair loop success rate, 
and JSON grammar constrained sampling accuracy.
"""

import time
import asyncio
import torch

from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.sampler.grammar_sampler import JSONGrammar, GrammarConstrainedSampler
from jarvis_core.sandbox.python_sandbox import PythonSandbox
from jarvis_core.sandbox.bash_sandbox import BashSandbox
from jarvis_core.planner.repair_engine import RepairEngine
from observability.sandbox_telemetry import SandboxTelemetryLogger

async def run_benchmarks():
    print("============================================================")
    print("⚡ HOMELAB Milestone v0.9.0 Sandbox & Grammar Benchmark")
    print("============================================================")

    # 1. Grammar-Constrained Sampling Validity Test
    tokenizer = BPETokenizer()
    tokenizer.train('{"action": "terminal:exec", "payload": {"command": "echo hi"}}\n', vocab_size=500)
    sampler = GrammarConstrainedSampler(tokenizer=tokenizer, grammar=JSONGrammar())

    current_text = '{"action":'
    logits = torch.randn(1, len(tokenizer.vocab))
    valid_count = 0
    num_samples = 100

    for _ in range(num_samples):
        tok_id = sampler.sample(logits=logits, current_text=current_text, temperature=0.8)
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        if JSONGrammar().is_valid_partial(current_text + tok_str):
            valid_count += 1

    json_validity_rate = (valid_count / num_samples) * 100.0
    print(f"\n--- 1. Grammar-Constrained Sampling ---")
    print(f"JSON Partial Validity Rate: {json_validity_rate:.2f}% ({valid_count}/{num_samples})")

    # 2. Python Sandbox Execution Latency & Overhead
    py_sandbox = PythonSandbox()
    t0 = time.perf_counter()
    py_res = await py_sandbox.execute("print(sum(range(10000)))\n")
    py_elapsed = (time.perf_counter() - t0) * 1000.0

    print(f"\n--- 2. Python Sandbox Execution ---")
    print(f"Stdout: {py_res.stdout.strip()}")
    print(f"Process Duration: {py_res.duration_ms:.2f} ms")
    print(f"Total Call Overhead: {py_elapsed - py_res.duration_ms:.2f} ms")

    # 3. Bash Sandbox Execution Latency
    bash_sandbox = BashSandbox()
    t0 = time.perf_counter()
    bash_res = await bash_sandbox.execute("echo 'JARVIS Sandbox Speed Test'")
    bash_elapsed = (time.perf_counter() - t0) * 1000.0

    print(f"\n--- 3. Bash Sandbox Execution ---")
    print(f"Stdout: {bash_res.stdout.strip()}")
    print(f"Process Duration: {bash_res.duration_ms:.2f} ms")
    print(f"Total Call Overhead: {bash_elapsed - bash_res.duration_ms:.2f} ms")

    # 4. Repair Engine Integration Test
    telemetry = SandboxTelemetryLogger()
    repair_engine = RepairEngine(telemetry_logger=telemetry)

    attempt_counter = 0
    async def faulty_callback():
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            raise RuntimeError("Initial syntax error in generated script")
        return {"status": "completed", "exit_code": 0, "stdout": "Self-repaired output"}

    repair_res = await repair_engine.attempt_repair(
        task_id="task_bench_01",
        error_msg="Initial syntax error",
        current_attempt=1,
        execute_fn=faulty_callback
    )

    print(f"\n--- 4. Automated Repair Loop ---")
    print(f"Repair Status: Repaired = {repair_res['repaired']}, Attempts = {attempt_counter}")
    print(f"Failure Classification: {repair_res['critique']['failure_type']}")

    print("\n============================================================")
    print("✅ Milestone v0.9.0 Benchmark Completed Successfully!")
    print("============================================================")

def main():
    asyncio.run(run_benchmarks())

if __name__ == "__main__":
    main()
