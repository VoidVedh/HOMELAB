import asyncio
from typing import Dict, List, Callable, Awaitable
from bus.events import Event

EventHandler = Callable[[Event], Awaitable[None]]

class EventBus:
    """Asynchronous Pub/Sub Event Bus for decoupled system module communication."""
    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._history: List[Event] = []

    def subscribe(self, event_type: str, handler: EventHandler):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Event):
        self._history.append(event)
        handlers = self._subscribers.get(event.event_type, [])
        # Execute handlers concurrently
        if handlers:
            await asyncio.gather(*(handler(event) for handler in handlers), return_exceptions=True)

    def get_history(self, event_type: str = None) -> List[Event]:
        if not event_type:
            return list(self._history)
        return [e for e in self._history if e.event_type == event_type]
