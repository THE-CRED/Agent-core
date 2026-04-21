"""
Base provider interface.

All provider adapters must implement this interface.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agent.messages import AgentRequest
from agent.response import AgentResponse
from agent.stream import StreamEvent
from agent.types.config import ProviderCapabilities


class BaseProvider(ABC):
    """
    Base class for all provider adapters.

    Each provider must implement the core methods to handle
    request conversion, response normalization, and streaming.
    """

    name: str = "base"
    capabilities: ProviderCapabilities = ProviderCapabilities()

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs

    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResponse:
        """
        Execute a synchronous request.

        Args:
            request: Normalized agent request

        Returns:
            Normalized agent response
        """
        ...

    @abstractmethod
    async def run_async(self, request: AgentRequest) -> AgentResponse:
        """
        Execute an asynchronous request.

        Args:
            request: Normalized agent request

        Returns:
            Normalized agent response
        """
        ...

    @abstractmethod
    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        """
        Execute a streaming request.

        Args:
            request: Normalized agent request

        Yields:
            Normalized stream events
        """
        ...

    @abstractmethod
    def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        """
        Execute an asynchronous streaming request.

        Subclasses implement this as an async generator (``async def`` with ``yield``).
        Callers should iterate with ``async for``, not ``await``.

        Args:
            request: Normalized agent request

        Yields:
            Normalized stream events
        """
        ...

    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return self.capabilities.tools

    def supports_structured_output(self) -> bool:
        """Check if provider supports structured output."""
        return self.capabilities.structured_output

    def supports_vision(self) -> bool:
        """Check if provider supports vision/images."""
        return self.capabilities.vision

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return self.capabilities.streaming

    def supports_json_mode(self) -> bool:
        """Check if provider supports JSON mode."""
        return self.capabilities.json_mode

    def supports_native_schema(self) -> bool:
        """Check if provider supports native schema-enforced output."""
        return self.capabilities.native_schema_output

    def validate_config(self) -> list[str]:
        """
        Validate provider configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not self.api_key:
            errors.append(f"API key required for {self.name} provider")
        return errors
