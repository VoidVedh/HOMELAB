import math
import time
from typing import Dict, Any

class MetricTracker:
    """Tracks research metrics: loss, perplexity, throughput (tokens/sec), and latency."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_losses = []
        self.total_tokens = 0
        self.start_time = time.time()

    def log_step(self, loss: float, num_tokens: int):
        self.step_losses.append(loss)
        self.total_tokens += num_tokens

    def get_summary(self) -> Dict[str, Any]:
        if not self.step_losses:
            return {"loss": 0.0, "perplexity": 0.0, "tokens_per_sec": 0.0}

        avg_loss = sum(self.step_losses) / len(self.step_losses)
        perplexity = math.exp(min(avg_loss, 20.0))  # Cap for numerical stability
        elapsed = max(time.time() - self.start_time, 1e-6)
        tok_per_sec = self.total_tokens / elapsed

        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 4),
            "tokens_per_sec": round(tok_per_sec, 2),
            "total_tokens": self.total_tokens,
            "elapsed_sec": round(elapsed, 2)
        }
