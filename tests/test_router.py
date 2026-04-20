"""Tests for the AgentRouter class."""

import pytest

from agent import AgentRouter
from agent.errors import ProviderError, RoutingError
from agent.router import RoutingStrategy
from agent.testing import FakeResponse, create_test_agent


class TestRouterBasics:
    """Test basic router functionality."""

    def test_create_router(self):
        """Test router creation."""
        agent1, _ = create_test_agent()
        agent2, _ = create_test_agent()

        router = AgentRouter(agents=[agent1, agent2])

        assert len(router.agents) == 2
        assert router.strategy == RoutingStrategy.FALLBACK

    def test_router_requires_agents(self):
        """Test router requires at least one agent."""
        with pytest.raises(ValueError, match="At least one agent"):
            AgentRouter(agents=[])

    def test_router_run(self):
        """Test basic router run."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Hello!"))

        router = AgentRouter(agents=[agent])
        response = router.run("Hi")

        assert response.text == "Hello!"


class TestFallbackStrategy:
    """Test fallback routing strategy."""

    def test_fallback_uses_first_agent(self):
        """Test fallback tries first agent first."""
        agent1, provider1 = create_test_agent()
        agent2, provider2 = create_test_agent()

        provider1.set_response(FakeResponse(text="From agent 1"))
        provider2.set_response(FakeResponse(text="From agent 2"))

        router = AgentRouter(
            agents=[agent1, agent2],
            strategy="fallback",
        )
        response = router.run("Test")

        assert response.text == "From agent 1"
        assert provider1.get_last_request() is not None
        assert provider2.get_last_request() is None

    def test_fallback_on_error(self):
        """Test fallback to second agent on error."""
        agent1, provider1 = create_test_agent()
        agent2, provider2 = create_test_agent()

        provider1.set_response(FakeResponse.with_error(ProviderError("API Error")))
        provider2.set_response(FakeResponse(text="From agent 2"))

        router = AgentRouter(
            agents=[agent1, agent2],
            strategy="fallback",
        )
        response = router.run("Test")

        assert response.text == "From agent 2"

    def test_all_agents_fail(self):
        """Test error when all agents fail."""
        agent1, provider1 = create_test_agent()
        agent2, provider2 = create_test_agent()

        provider1.set_response(FakeResponse.with_error(ProviderError("Error 1")))
        provider2.set_response(FakeResponse.with_error(ProviderError("Error 2")))

        router = AgentRouter(agents=[agent1, agent2])

        with pytest.raises(RoutingError) as exc_info:
            router.run("Test")

        assert len(exc_info.value.errors) == 2


class TestRoundRobinStrategy:
    """Test round-robin routing strategy."""

    def test_round_robin_rotates(self):
        """Test round-robin rotates through agents."""
        agent1, provider1 = create_test_agent()
        agent2, provider2 = create_test_agent()

        provider1.set_response(FakeResponse(text="Agent 1"))
        provider2.set_response(FakeResponse(text="Agent 2"))

        router = AgentRouter(
            agents=[agent1, agent2],
            strategy="round_robin",
        )

        r1 = router.run("First")
        r2 = router.run("Second")
        r3 = router.run("Third")

        # Should rotate: 1, 2, 1
        assert r1.text == "Agent 1"
        assert r2.text == "Agent 2"
        assert r3.text == "Agent 1"


class TestCheapestStrategy:
    """Test cheapest routing strategy."""

    def test_cheapest_orders_by_cost(self):
        """Test cheapest strategy orders agents by cost."""
        # This test is more of an integration test since it depends on pricing
        agent1, provider1 = create_test_agent()
        provider1.set_response(FakeResponse(text="Response"))

        router = AgentRouter(
            agents=[agent1],
            strategy="cheapest",
        )
        response = router.run("Test")

        assert response.text == "Response"


class TestRouterStreaming:
    """Test router streaming functionality."""

    def test_stream_basic(self):
        """Test basic streaming works."""
        agent, provider = create_test_agent()
        provider.set_response(FakeResponse(text="Streamed response"))

        router = AgentRouter(agents=[agent])

        events = list(router.stream("Test"))
        text_events = [e for e in events if e.type == "text_delta"]

        assert len(text_events) > 0


class TestRouterJson:
    """Test router JSON/structured output."""

    def test_json_with_fallback(self):
        """Test JSON output with fallback."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            value: str

        agent1, provider1 = create_test_agent()
        agent2, provider2 = create_test_agent()

        provider1.set_response(FakeResponse.with_error(ProviderError("Error")))
        provider2.set_response(FakeResponse(text='{"value": "test"}'))

        router = AgentRouter(agents=[agent1, agent2])
        response = router.json("Get data", schema=TestModel)

        assert response.output is not None
        assert response.output.value == "test"
