import time
import torch

def get_device() -> torch.device:
    """Returns the optimal available PyTorch device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Timer:
    """Utility timer context manager for profiling inference & training latency."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval_ms = (self.end - self.start) * 1000.0
