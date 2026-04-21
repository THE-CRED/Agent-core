"""
Test fixtures for Agent testing.
"""

from typing import Any

from agent.agent import Agent
from agent.response import AgentResponse, Usage
from agent.testing.fake_provider import FakeProvider, FakeResponse
from agent.tools import ToolCall


def create_test_agent(
    responses: list[FakeResponse] | None = None,
    **kwargs: Any,
) -> tuple[Agent, FakeProvider]:
    """
    Create an agent with a fake provider for testing.

    Args:
        responses: Optional list of fake responses
        **kwargs: Additional agent configuration

    Returns:
        Tuple of (Agent, FakeProvider) for testing

    Example:
        ```python
        agent, provider = create_test_agent([
            FakeResponse(text="Hello!"),
            FakeResponse(text="Goodbye!"),
        ])

        response1 = agent.run("Hi")
        assert response1.text == "Hello!"

        response2 = agent.run("Bye")
        assert response2.text == "Goodbye!"
        ```
    """
    # Import here to avoid circular import
    from agent.testing.fake_provider import FakeProvider

    # Create agent with fake provider
    agent = Agent(
        provider="fake",
        model="fake-model",
        **kwargs,
    )

    # Get the provider instance
    provider = agent._provider
    if not isinstance(provider, FakeProvider):
        raise RuntimeError("Expected FakeProvider")

    # Set responses if provided
    if responses:
        provider.set_responses(responses)

    return agent, provider


def create_test_response(
    text: str = "Test response",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
    provider: str = "fake",
    model: str = "fake-model",
    stop_reason: str = "stop",
    latency_ms: float = 100.0,
) -> AgentResponse:
    """
    Create a test AgentResponse.

    Args:
        text: Response text
        tool_calls: Optional tool calls
        usage: Optional usage info
        provider: Provider name
        model: Model name
        stop_reason: Stop reason
        latency_ms: Latency in milliseconds

    Returns:
        AgentResponse for testing

    Example:
        ```python
        response = create_test_response(
            text="Hello, world!",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert response.text == "Hello, world!"
        assert response.usage.total_tokens == 15
        ```
    """
    return AgentResponse(
        text=text,
        content=[{"type": "text", "text": text}] if text else [],
        provider=provider,
        model=model,
        usage=usage
        or Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
        stop_reason=stop_reason,
        tool_calls=tool_calls or [],
        raw={"test": True},
        latency_ms=latency_ms,
    )


class AgentTestCase:
    """
    Base test case class for Agent testing.

    Provides helper methods for common testing patterns.

    Example:
        ```python
        class TestMyFeature(AgentTestCase):
            def test_greeting(self):
                self.set_response("Hello!")
                response = self.agent.run("Hi")
                self.assert_response_text(response, "Hello!")
                self.assert_request_contains("Hi")
        ```
    """

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.agent, self.provider = create_test_agent()

    def set_response(self, text: str) -> None:
        """Set a simple text response."""
        self.provider.set_response(FakeResponse(text=text))

    def set_responses(self, texts: list[str]) -> None:
        """Set multiple text responses."""
        self.provider.set_responses([FakeResponse(text=t) for t in texts])

    def set_tool_response(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_id: str = "call_123",
    ) -> None:
        """Set a tool call response."""
        self.provider.set_response(FakeResponse.with_tool_call(tool_name, arguments, tool_id))

    def set_error(self, error: Exception) -> None:
        """Set an error response."""
        self.provider.set_response(FakeResponse.with_error(error))

    def get_last_request(self) -> Any:
        """Get the last request made to the provider."""
        return self.provider.get_last_request()

    def assert_response_text(self, response: AgentResponse, expected: str) -> None:
        """Assert response text matches expected."""
        assert response.text == expected, f"Expected '{expected}', got '{response.text}'"

    def assert_request_contains(self, text: str) -> None:
        """Assert the last request input contains text."""
        request = self.get_last_request()
        assert request is not None, "No request was made"
        assert text in (request.input or ""), f"Request input doesn't contain '{text}'"

    def assert_tool_called(self, tool_name: str) -> None:
        """Assert a tool was called (in tool loop scenarios)."""
        request = self.get_last_request()
        assert request is not None, "No request was made"
        # Check if any messages contain tool results for the named tool
        for msg in request.messages:
            if msg.role == "tool" and msg.name == tool_name:
                return
        raise AssertionError(f"Tool '{tool_name}' was not called")
