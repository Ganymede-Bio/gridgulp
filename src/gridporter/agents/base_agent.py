"""Base Agent class for GridPorter agents."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from gridporter.config import Config


class BaseAgent(ABC):
    """
    Abstract base class for all GridPorter agents.

    Agents are responsible for making strategic decisions and coordinating
    tools to accomplish complex tasks. They maintain state, handle errors,
    and implement retry logic.
    """

    def __init__(self, config: Config):
        """Initialize the agent with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state: dict[str, Any] = {}
        self.error_count = 0
        self.max_retries = 3

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute the agent's main task.

        This method should be implemented by each agent to perform its
        specific responsibilities.
        """
        pass

    async def execute_with_retry(self, *args, **kwargs) -> Any:
        """Execute with automatic retry on failure."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return await self.execute(*args, **kwargs)
            except Exception as e:
                self.error_count += 1
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        # All retries failed
        self.logger.error(f"All {self.max_retries} attempts failed")
        raise last_error

    def update_state(self, key: str, value: Any) -> None:
        """Update agent state."""
        self.state[key] = value
        self.logger.debug(f"State updated: {key} = {value}")

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state."""
        return self.state.get(key, default)

    def reset_state(self) -> None:
        """Reset agent state."""
        self.state.clear()
        self.error_count = 0
        self.logger.debug("Agent state reset")

    def should_use_fallback(self) -> bool:
        """Determine if fallback strategy should be used."""
        return self.error_count >= 2

    @property
    def agent_type(self) -> str:
        """Return the agent type for logging and metrics."""
        return self.__class__.__name__.replace("Agent", "").lower()
