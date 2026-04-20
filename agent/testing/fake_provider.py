"""
Fake provider for testing.

Provides deterministic responses for unit testing.
"""

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from agent.messages import AgentRequest
from agent.providers.base import BaseProvider, ProviderCapabilities
from agent.providers.registry import ProviderRegistry
from agent.response import AgentResponse, Usage
from agent.stream import StreamEvent
from agent.tools import ToolCall


@dataclass
class FakeResponse:
    """Configuration for a fake response."""

    text: str = "This is a fake response."
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    stop_reason: str = "stop"
    latency_ms: float = 100.0
    error: Exception | None = None

    @classmethod
    def with_text(cls, text: str) -> "FakeResponse":
        """Create a response with specific text."""
        return cls(text=text)

    @classmethod
    def with_tool_call(
        cls,
        name: str,
        arguments: dict[str, Any],
        id: str = "call_123",
    ) -> "FakeResponse":
        """Create a response with a tool call."""
        return cls(
            text="",
            tool_calls=[ToolCall(id=id, name=name, arguments=arguments)],
            stop_reason="tool_calls",
        )

    @classmethod
    def with_error(cls, error: Exception) -> "FakeResponse":
        """Create a response that raises an error."""
        return cls(error=error)


class FakeProvider(BaseProvider):
    """
    Fake provider for testing.

    Allows configuring responses for deterministic testing.

    Example:
        ```python
        provider = FakeProvider()
        provider.set_response(FakeResponse(text="Hello!"))

        agent = Agent(provider="fake", model="fake")
        response = agent.run("Hi")
        assert response.text == "Hello!"
        ```
    """

    name = "fake"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=True,
        system_messages=True,
        batch=True,
        native_schema_output=True,
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        self._responses: list[FakeResponse] = []
        self._response_index = 0
        self._requests: list[AgentRequest] = []
        self._response_fn: Callable[[AgentRequest], FakeResponse] | None = None
        self._default_response = FakeResponse()

    def set_response(self, response: FakeResponse) -> None:
        """Set a single response to return."""
        self._responses = [response]
        self._response_index = 0

    def set_responses(self, responses: list[FakeResponse]) -> None:
        """Set multiple responses to return in sequence."""
        self._responses = responses
        self._response_index = 0

    def set_response_fn(
        self,
        fn: Callable[[AgentRequest], FakeResponse],
    ) -> None:
        """Set a function to generate responses dynamically."""
        self._response_fn = fn

    def get_requests(self) -> list[AgentRequest]:
        """Get all requests that were made."""
        return self._requests

    def get_last_request(self) -> AgentRequest | None:
        """Get the most recent request."""
        return self._requests[-1] if self._requests else None

    def clear(self) -> None:
        """Clear all responses and recorded requests."""
        self._responses = []
        self._response_index = 0
        self._requests = []
        self._response_fn = None

    def _get_next_response(self, request: AgentRequest) -> FakeResponse:
        """Get the next response to return."""
        self._requests.append(request)

        # Use response function if set
        if self._response_fn:
            return self._response_fn(request)

        # Use configured responses
        if self._responses:
            response = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)
            return response

        # Use default response
        return self._default_response

    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a synchronous request."""
        fake_response = self._get_next_response(request)

        if fake_response.error:
            raise fake_response.error

        return AgentResponse(
            text=fake_response.text,
            content=[{"type": "text", "text": fake_response.text}] if fake_response.text else [],
            provider=self.name,
            model="fake-model",
            usage=fake_response.usage or Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            stop_reason=fake_response.stop_reason,
            tool_calls=fake_response.tool_calls,
            raw={"fake": True},
            latency_ms=fake_response.latency_ms,
        )

    async def run_async(self, request: AgentRequest) -> AgentResponse:
        """Execute an asynchronous request."""
        return self.run(request)

    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        """Execute a streaming request."""
        fake_response = self._get_next_response(request)

        if fake_response.error:
            raise fake_response.error

        yield StreamEvent.message_start_event()

        # Stream text in chunks
        if fake_response.text:
            words = fake_response.text.split()
            for i, word in enumerate(words):
                text = word if i == 0 else " " + word
                yield StreamEvent.text_delta(text)

        # Emit tool calls
        for tc in fake_response.tool_calls:
            yield StreamEvent.tool_call_start(tc)

        # Usage and end
        yield StreamEvent.usage_event(
            fake_response.usage or Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            )
        )
        yield StreamEvent.message_end()

    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        """Execute an async streaming request."""
        for event in self.stream(request):
            yield event


# Register the fake provider
ProviderRegistry.register("fake", FakeProvider, aliases=["test", "mock"])
