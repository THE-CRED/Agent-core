"""Tests for the Agent class."""

from agent import AgentResponse
from agent.testing import FakeResponse, create_test_agent


class TestAgent:
    """Test Agent class functionality."""

    def test_create_agent(self):
        """Test basic agent creation."""
        agent, _ = create_test_agent()
        assert agent.provider == "fake"
        assert agent.model == "fake-model"

    def test_run_returns_response(self):
        """Test that run() returns an AgentResponse."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hello, world!"))

        response = agent.run("Hi there")

        assert isinstance(response, AgentResponse)
        assert response.text == "Hello, world!"

    def test_run_records_request(self):
        """Test that requests are recorded."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Response"))

        agent.run("Test input")

        request = provider.get_last_request()
        assert request is not None
        assert request.input == "Test input"

    def test_system_prompt(self):
        """Test system prompt is passed correctly."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="OK"))

        agent.run("Hi", system="You are a helpful assistant")

        request = provider.get_last_request()
        assert request is not None
        assert request.system == "You are a helpful assistant"

    def test_default_system_prompt(self):
        """Test default system prompt from config."""
        agent, provider = create_test_agent(default_system="Default system")
        provider.set_response(FakeResponse(text="OK"))

        agent.run("Hi")

        request = provider.get_last_request()
        assert request is not None
        assert request.system == "Default system"

    def test_temperature_override(self):
        """Test temperature can be overridden."""
        agent, provider = create_test_agent(temperature=0.5)
        provider.set_response(FakeResponse(text="OK"))

        agent.run("Hi", temperature=0.9)

        request = provider.get_last_request()
        assert request is not None
        assert request.temperature == 0.9

    def test_multiple_responses(self):
        """Test multiple responses in sequence."""
        agent, provider = create_test_agent()
        provider.set_responses(
            [
                FakeResponse(text="First"),
                FakeResponse(text="Second"),
                FakeResponse(text="Third"),
            ]
        )

        r1 = agent.run("1")
        r2 = agent.run("2")
        r3 = agent.run("3")

        assert r1.text == "First"
        assert r2.text == "Second"
        assert r3.text == "Third"

    def test_with_config_creates_new_agent(self):
        """Test with_config creates a new agent."""
        agent, _ = create_test_agent()
        new_agent = agent.with_config(temperature=0.8)

        assert new_agent is not agent
        assert new_agent.config.temperature == 0.8

    def test_response_includes_usage(self):
        """Test response includes usage information."""
        agent, provider = create_test_agent()
        from agent.response import Usage

        provider.set_response(
            FakeResponse(
                text="Hi",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        )

        response = agent.run("Hello")

        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15


class TestAgentStreaming:
    """Test Agent streaming functionality."""

    def test_stream_returns_events(self):
        """Test streaming returns events."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hello world"))

        events = list(agent.stream("Hi"))

        assert len(events) > 0
        # Should have text deltas
        text_events = [e for e in events if e.type == "text_delta"]
        assert len(text_events) > 0

    def test_stream_accumulates_text(self):
        """Test stream response accumulates text."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hello world"))

        stream = agent.stream("Hi")
        stream.collect()

        assert "Hello" in stream.text
        assert "world" in stream.text


class TestAgentJson:
    """Test Agent structured output functionality."""

    def test_json_returns_response(self):
        """Test json() returns response with output."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text='{"name": "test", "value": 42}'))

        response = agent.json("Get data", schema=TestSchema)

        assert response.output is not None
        assert response.output.name == "test"
        assert response.output.value == 42
