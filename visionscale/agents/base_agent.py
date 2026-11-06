"""Base class for deterministic Python-based VisionScale agents."""

from __future__ import annotations
import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("visionscale.agents")


@dataclass
class AgentContext:
    """Shared pipeline context passed through the agent pipeline."""
    data: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def log(self, sender: str, message: str) -> None:
        entry = f"[{sender}] {message}"
        self.logs.append(entry)
        logger.info(entry)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class BaseAgent(abc.ABC):
    """Abstract deterministic agent."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._current_context: Optional[AgentContext] = None

    def log(self, *args: Any) -> None:
        """Flexible logger supporting self.log(msg), self.log(sender, msg), or self.log(context, msg)."""
        if len(args) == 1:
            msg = str(args[0])
            sender = self.name
        elif len(args) >= 2:
            if isinstance(args[0], AgentContext):
                sender = self.name
                msg = str(args[1])
                args[0].log(sender, msg)
                return
            sender = str(args[0])
            msg = str(args[1])
        else:
            return

        if self._current_context is not None:
            self._current_context.log(sender, msg)
        else:
            logger.info(f"[{sender}] {msg}")

    def __call__(self, context: AgentContext) -> AgentContext:
        self._current_context = context
        try:
            return self.run(context)
        finally:
            self._current_context = None

    @abc.abstractmethod
    def run(self, context: AgentContext) -> AgentContext:
        """Execute deterministic reasoning or processing step."""
        pass
