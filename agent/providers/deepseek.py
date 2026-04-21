"""
DeepSeek provider adapter.

Uses OpenAI-compatible API.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any  # noqa: F811

from agent.messages import AgentRequest
from agent.providers.base import BaseProvider
from agent.providers.registry import ProviderRegistry
from agent.response import AgentResponse
from agent.stream import StreamEvent
from agent.types.config import ProviderCapabilities

# DeepSeek uses OpenAI-compatible API
try:
    from agent.providers.openai import HAS_OPENAI, OpenAIProvider
except ImportError:
    HAS_OPENAI = False
    OpenAIProvider: Any = None


class DeepSeekProvider(BaseProvider):
    """
    DeepSeek provider adapter.

    Uses the OpenAI-compatible API with DeepSeek's endpoint.
    """

    name = "deepseek"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=False,  # DeepSeek doesn't support vision yet
        system_messages=True,
        batch=False,
        native_schema_output=False,
    )

    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package not installed (required for DeepSeek). "
                "Install with: pip install agent-core-py[deepseek]"
            )

        super().__init__(
            api_key=api_key,
            base_url=base_url or self.DEEPSEEK_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Use OpenAI provider with DeepSeek endpoint
        assert OpenAIProvider is not None
        self._openai_provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or self.DEEPSEEK_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Override provider name in responses
        self._openai_provider.name = self.name

    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a synchronous request."""
        response = self._openai_provider.run(request)
        response.provider = self.name
        return response

    async def run_async(self, request: AgentRequest) -> AgentResponse:
        """Execute an asynchronous request."""
        response = await self._openai_provider.run_async(request)
        response.provider = self.name
        return response

    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        """Execute a streaming request."""
        yield from self._openai_provider.stream(request)

    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        """Execute an async streaming request."""
        async for event in self._openai_provider.stream_async(request):
            yield event


# Register the provider
ProviderRegistry.register("deepseek", DeepSeekProvider)
