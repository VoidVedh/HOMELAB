import pytest
import asyncio
from bus.event_bus import EventBus
from bus.events import TaskSubmittedEvent

@pytest.mark.asyncio
async def test_event_bus_publish_subscribe():
    bus = EventBus()
    received_events = []

    async def handler(event):
        received_events.append(event)

    bus.subscribe("task.submitted", handler)
    
    event = TaskSubmittedEvent(sender="test", task_id="123", goal="Test Goal")
    await bus.publish(event)

    assert len(received_events) == 1
    assert received_events[0].payload["task_id"] == "123"
    assert len(bus.get_history()) == 1
