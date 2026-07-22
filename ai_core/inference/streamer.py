from abc import ABC, abstractmethod
from typing import Callable, Optional, List

class BaseStreamer(ABC):
    """Abstract base interface for token generation streaming."""
    
    @abstractmethod
    def put(self, token_id: int, token_text: str):
        """Processes a newly generated token."""
        pass

    @abstractmethod
    def end(self):
        """Flushes the stream upon completion."""
        pass

class TokenStreamer(BaseStreamer):
    """Token streamer yielding strings and invoking optional callback handlers."""
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.tokens: List[str] = []

    def put(self, token_id: int, token_text: str):
        self.tokens.append(token_text)
        if self.callback:
            self.callback(token_text)

    def end(self):
        pass

    def get_full_text(self) -> str:
        return "".join(self.tokens)
