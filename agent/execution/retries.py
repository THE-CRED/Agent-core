"""
Retry handling for transient errors.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from agent.types.config import RetryConfig

T = TypeVar("T")


class RetryHandler:
    """Handles retry logic for operations."""

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def execute(
        self,
        operation: Callable[[], T],
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> T:
        """
        Execute operation with retries.

        Args:
            operation: The operation to execute
            on_retry: Optional callback called before each retry

        Returns:
            Result of the operation

        Raises:
            The last exception if all retries fail
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e

                if not self.config.should_retry(e, attempt):
                    raise

                if attempt < self.config.max_retries:
                    delay = self.config.get_delay(attempt, e)
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    time.sleep(delay)

        # Should not reach here, but satisfy type checker
        raise last_error  # type: ignore

    async def execute_async(
        self,
        operation: Callable[[], Awaitable[T]],
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> T:
        """
        Execute async operation with retries.

        Args:
            operation: The async operation to execute
            on_retry: Optional callback called before each retry

        Returns:
            Result of the operation

        Raises:
            The last exception if all retries fail
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_error = e

                if not self.config.should_retry(e, attempt):
                    raise

                if attempt < self.config.max_retries:
                    delay = self.config.get_delay(attempt, e)
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    await asyncio.sleep(delay)

        raise last_error  # type: ignore


# Re-export for backwards compatibility
__all__ = ["RetryConfig", "RetryHandler"]
