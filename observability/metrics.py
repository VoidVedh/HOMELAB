import time
from typing import Dict, Any

class TelemetryCollector:
    """Collects system latency, token generation speed, and memory telemetry."""
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}

    def increment(self, metric: str, value: int = 1):
        self._counters[metric] = self._counters.get(metric, 0) + value

    def set_gauge(self, metric: str, value: float):
        self._gauges[metric] = value

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges)
        }
