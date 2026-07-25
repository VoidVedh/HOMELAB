#!/usr/bin/env python3
"""
HOMELAB Milestone v0.12.0 Benchmark Runner
Measures multi-agent task latency, message bus pub-sub throughput,
specialized agent execution overhead, and automated repair loop rates.
"""

import time
import asyncio

from jarvis_core.agents.protocol import MessageBus, AgentMessage, MessageType, AgentTask
from jarvis_core.agents.orchestrator import AgentOrchestrator

async def run_agent_benchmarks():
    print("============================================================")
    print("⚡ HOMELAB Milestone v0.12.0 Multi-Agent Framework Benchmark")
    print("============================================================")

    # 1. Message Bus Pub-Sub Throughput
    bus = MessageBus()
    count = 0
    def sub_cb(msg: AgentMessage):
        nonlocal count
        count += 1

    bus.subscribe("TargetAgent", sub_cb)
    num_messages = 10000

    t0 = time.perf_counter()
    for i in range(num_messages):
        msg = AgentMessage(
            message_id=f"m_{i}",
            sender="SenderAgent",
            recipient="TargetAgent",
            message_type=MessageType.PUBLISH,
            payload={"index": i}
        )
        bus.publish(msg)

    bus_duration_sec = time.perf_counter() - t0
    msg_per_sec = num_messages / max(bus_duration_sec, 1e-6)

    print(f"\n--- 1. Message Bus Pub-Sub Performance ---")
    print(f"Messages Routed: {count:,} / {num_messages:,}")
    print(f"Routing Duration: {bus_duration_sec*1000.0:.2f} ms")
    print(f"Throughput: {msg_per_sec:,.2f} messages/sec")

    # 2. Multi-Agent Team Orchestration Workflow Latency
    orchestrator = AgentOrchestrator(message_bus=bus)
    
    t0 = time.perf_counter()
    res = await orchestrator.run_goal("Build, test, review, and audit new neural network module")
    workflow_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n--- 2. Multi-Agent Workflow Execution ---")
    print(f"Goal Status: {res['status'].upper()}")
    print(f"Subtasks Executed: {res['subtasks_executed']}")
    print(f"Total Workflow Latency: {workflow_ms:.2f} ms")

    # 3. Agent Execution Breakdown
    print(f"\n--- 3. Subtask Agent Metrics Breakdown ---")
    for st in res["results"]:
        print(f"  • Task [{st['task_id']}] -> Agent '{st['agent_name']}': Status = {st['status']}, Duration = {st['duration_ms']:.2f} ms")

    print(f"\n--- 4. Orchestrator Summary Metrics ---")
    print(f"Total Executed Tasks: {orchestrator.metrics['total_tasks']}")
    print(f"Successful Tasks: {orchestrator.metrics['successful_tasks']}")
    print(f"Repair Attempts: {orchestrator.metrics['repair_attempts']}")

    print("\n============================================================")
    print("✅ Milestone v0.12.0 Benchmark Completed Successfully!")
    print("============================================================")

def main():
    asyncio.run(run_agent_benchmarks())

if __name__ == "__main__":
    main()
