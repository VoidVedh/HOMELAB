import pytest
import asyncio

from jarvis_core.agents.protocol import MessageBus, AgentMessage, MessageType, AgentTask
from jarvis_core.agents.specialized_agents import (
    PlannerAgent,
    CodingAgent,
    ReviewAgent,
    TestingAgent,
    SecurityAgent,
    ResearchAgent,
    MemoryAgent,
    DocumentationAgent,
    DeploymentAgent
)
from jarvis_core.agents.orchestrator import AgentOrchestrator

def test_message_bus_pub_sub():
    bus = MessageBus()
    received_msgs = []

    def callback(msg: AgentMessage):
        received_msgs.append(msg)

    bus.subscribe("CodingAgent", callback)
    
    test_msg = AgentMessage(
        message_id="msg_01",
        sender="PlannerAgent",
        recipient="CodingAgent",
        message_type=MessageType.PUBLISH,
        payload={"task": "implement"}
    )
    bus.publish(test_msg)

    assert len(received_msgs) == 1
    assert received_msgs[0].sender == "PlannerAgent"

@pytest.mark.asyncio
async def test_specialized_agents_execution():
    bus = MessageBus()
    planner = PlannerAgent(bus)
    coding = CodingAgent(bus)
    review = ReviewAgent(bus)
    security = SecurityAgent(bus)

    assert planner.can_handle("plan architecture") == True
    assert coding.can_handle("code script") == True
    assert review.can_handle("review code") == True
    assert security.can_handle("security audit") == True

    task_coding = AgentTask(
        task_id="t1",
        goal="code function",
        assigned_agent="CodingAgent",
        payload={"code": "print('Unit test code')\n"}
    )
    res_coding = await coding.execute(task_coding)
    assert res_coding.status == "completed"
    assert "Unit test code" in res_coding.output

    task_sec = AgentTask(
        task_id="t2",
        goal="security scan",
        assigned_agent="SecurityAgent",
        payload={"target": "clean python string"}
    )
    res_sec = await security.execute(task_sec)
    assert res_sec.status == "completed"

@pytest.mark.asyncio
async def test_agent_orchestrator_workflow():
    orchestrator = AgentOrchestrator()
    res = await orchestrator.run_goal("Build and verify python module")

    assert res["status"] == "completed"
    assert res["subtasks_executed"] > 0
    assert orchestrator.metrics["total_tasks"] > 0
    assert orchestrator.metrics["successful_tasks"] > 0
