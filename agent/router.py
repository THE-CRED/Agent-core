"""
Agent router for multi-agent routing and fallback.
"""

import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agent.agent import Agent
from agent.config import PRICING, resolve_model
from agent.errors import AgentError, RoutingError
from agent.messages import AgentRequest, Message
from agent.response import AgentResponse
from agent.stream import AsyncStreamResponse, StreamResponse
from agent.types.router import RouteResult, RoutingStrategy


class AgentRouter:
    """
    Routes requests across multiple agents with fallback support.

    The router provides strategies for:
    - Automatic failover when providers fail
    - Load balancing across providers
    - Capability-based routing
    - Cost optimization

    Example:
        ```python
        router = AgentRouter(
            agents=[
                Agent(provider="anthropic", model="claude-sonnet"),
                Agent(provider="openai", model="gpt-4o"),
            ],
            strategy="fallback",
        )

        # Automatically falls back if first provider fails
        response = router.run("Hello!")
        ```
    """

    def __init__(
        self,
        agents: list[Agent],
        strategy: RoutingStrategy | str = RoutingStrategy.FALLBACK,
        custom_router: Callable[[AgentRequest, list[Agent]], RouteResult] | None = None,
    ):
        """
        Initialize the router.

        Args:
            agents: List of agents to route between
            strategy: Routing strategy to use
            custom_router: Custom routing function (required if strategy is CUSTOM)
        """
        if not agents:
            raise ValueError("At least one agent is required")

        self.agents = agents
        self.strategy = RoutingStrategy(strategy) if isinstance(strategy, str) else strategy
        self.custom_router = custom_router
        self._round_robin_index = 0

        if self.strategy == RoutingStrategy.CUSTOM and not custom_router:
            raise ValueError("custom_router required when strategy is CUSTOM")

    def run(
        self,
        input: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Execute a request using the routing strategy.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            stop: Stop sequences
            metadata: Request metadata

        Returns:
            AgentResponse from the selected agent

        Raises:
            RoutingError: If all agents fail
        """
        request = AgentRequest(
            input=input,
            messages=messages or [],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            metadata=metadata or {},
        )

        if self.strategy == RoutingStrategy.FALLBACK:
            return self._run_fallback(request)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._run_round_robin(request)
        elif self.strategy == RoutingStrategy.FASTEST:
            return self._run_fastest_sync(request)
        elif self.strategy == RoutingStrategy.CHEAPEST:
            return self._run_cheapest(request)
        elif self.strategy == RoutingStrategy.CAPABILITY:
            return self._run_capability(request)
        elif self.strategy == RoutingStrategy.CUSTOM:
            return self._run_custom(request)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def run_async(
        self,
        input: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Execute a request asynchronously using the routing strategy.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            stop: Stop sequences
            metadata: Request metadata

        Returns:
            AgentResponse from the selected agent
        """
        request = AgentRequest(
            input=input,
            messages=messages or [],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            metadata=metadata or {},
        )

        if self.strategy == RoutingStrategy.FALLBACK:
            return await self._run_fallback_async(request)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._run_round_robin_async(request)
        elif self.strategy == RoutingStrategy.FASTEST:
            return await self._run_fastest_async(request)
        elif self.strategy == RoutingStrategy.CHEAPEST:
            return await self._run_cheapest_async(request)
        elif self.strategy == RoutingStrategy.CAPABILITY:
            return await self._run_capability_async(request)
        elif self.strategy == RoutingStrategy.CUSTOM:
            return await self._run_custom_async(request)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def stream(
        self,
        input: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StreamResponse:
        """
        Stream a response using the routing strategy.

        For fallback strategy, only tries next agent if initial connection fails.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            metadata: Request metadata

        Returns:
            StreamResponse from the selected agent
        """
        errors: list[Exception] = []

        for agent in self._get_ordered_agents():
            try:
                return agent.stream(
                    input=input,
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata=metadata,
                )
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    async def stream_async(
        self,
        input: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncStreamResponse:
        """
        Stream a response asynchronously using the routing strategy.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            metadata: Request metadata

        Returns:
            AsyncStreamResponse from the selected agent
        """
        errors: list[Exception] = []

        for agent in self._get_ordered_agents():
            try:
                return await agent.stream_async(
                    input=input,
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata=metadata,
                )
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    def json(
        self,
        input: str | None = None,
        *,
        schema: type[BaseModel] | dict[str, Any],
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Execute a structured output request with routing.

        Args:
            input: User input text
            schema: Output schema
            messages: Optional message history
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens
            metadata: Request metadata

        Returns:
            AgentResponse with parsed output
        """
        errors: list[Exception] = []

        for agent in self._get_ordered_agents():
            try:
                return agent.json(
                    input=input,
                    schema=schema,
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata=metadata,
                )
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed for structured output",
            errors=errors,
        )

    # Strategy implementations

    def _run_fallback(self, request: AgentRequest) -> AgentResponse:
        """Try each agent in order until one succeeds."""
        errors: list[Exception] = []

        for agent in self.agents:
            try:
                return agent._runtime.run(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    async def _run_fallback_async(self, request: AgentRequest) -> AgentResponse:
        """Try each agent in order until one succeeds (async)."""
        errors: list[Exception] = []

        for agent in self.agents:
            try:
                return await agent._runtime.run_async(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    def _run_round_robin(self, request: AgentRequest) -> AgentResponse:
        """Rotate through agents."""
        errors: list[Exception] = []
        start_index = self._round_robin_index

        for i in range(len(self.agents)):
            index = (start_index + i) % len(self.agents)
            agent = self.agents[index]

            try:
                self._round_robin_index = (index + 1) % len(self.agents)
                return agent._runtime.run(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    async def _run_round_robin_async(self, request: AgentRequest) -> AgentResponse:
        """Rotate through agents (async)."""
        errors: list[Exception] = []
        start_index = self._round_robin_index

        for i in range(len(self.agents)):
            index = (start_index + i) % len(self.agents)
            agent = self.agents[index]

            try:
                self._round_robin_index = (index + 1) % len(self.agents)
                return await agent._runtime.run_async(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    def _run_fastest_sync(self, request: AgentRequest) -> AgentResponse:
        """Race agents synchronously (uses threads)."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(agent._runtime.run, request): agent
                for agent in self.agents
            }

            errors: list[Exception] = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    return future.result()
                except AgentError as e:
                    errors.append(e)
                    continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    async def _run_fastest_async(self, request: AgentRequest) -> AgentResponse:
        """Race agents, return first successful response."""
        tasks = [
            asyncio.create_task(agent._runtime.run_async(request))
            for agent in self.agents
        ]

        errors: list[Exception] = []
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Get first successful result
        for task in done:
            try:
                result = task.result()
                # Cancel pending tasks
                for p in pending:
                    p.cancel()
                return result
            except AgentError as e:
                errors.append(e)

        # Wait for remaining if first failed
        if pending:
            done, _ = await asyncio.wait(pending)
            for task in done:
                try:
                    return task.result()
                except AgentError as e:
                    errors.append(e)

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    def _run_cheapest(self, request: AgentRequest) -> AgentResponse:
        """Use the cheapest available agent."""
        # Sort agents by estimated cost
        def get_cost(agent: Agent) -> float:
            model = resolve_model(agent.model)
            pricing = PRICING.get(model, {})
            return pricing.get("input", float("inf")) + pricing.get("output", float("inf"))

        sorted_agents = sorted(self.agents, key=get_cost)

        errors: list[Exception] = []
        for agent in sorted_agents:
            try:
                return agent._runtime.run(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    async def _run_cheapest_async(self, request: AgentRequest) -> AgentResponse:
        """Use the cheapest available agent (async)."""
        def get_cost(agent: Agent) -> float:
            model = resolve_model(agent.model)
            pricing = PRICING.get(model, {})
            return pricing.get("input", float("inf")) + pricing.get("output", float("inf"))

        sorted_agents = sorted(self.agents, key=get_cost)

        errors: list[Exception] = []
        for agent in sorted_agents:
            try:
                return await agent._runtime.run_async(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All {len(self.agents)} agents failed",
            errors=errors,
        )

    def _run_capability(self, request: AgentRequest) -> AgentResponse:
        """Route based on required capabilities."""
        # Determine required capabilities from request
        needs_tools = bool(request.tools)
        needs_schema = bool(request.schema)

        # Filter to capable agents
        capable_agents = []
        for agent in self.agents:
            provider = agent._provider
            if needs_tools and not provider.supports_tools():
                continue
            if needs_schema and not provider.supports_structured_output():
                continue
            capable_agents.append(agent)

        if not capable_agents:
            raise RoutingError(
                "No agents support the required capabilities",
                errors=[],
            )

        # Use fallback among capable agents
        errors: list[Exception] = []
        for agent in capable_agents:
            try:
                return agent._runtime.run(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All capable agents ({len(capable_agents)}) failed",
            errors=errors,
        )

    async def _run_capability_async(self, request: AgentRequest) -> AgentResponse:
        """Route based on required capabilities (async)."""
        needs_tools = bool(request.tools)
        needs_schema = bool(request.schema)

        capable_agents = []
        for agent in self.agents:
            provider = agent._provider
            if needs_tools and not provider.supports_tools():
                continue
            if needs_schema and not provider.supports_structured_output():
                continue
            capable_agents.append(agent)

        if not capable_agents:
            raise RoutingError(
                "No agents support the required capabilities",
                errors=[],
            )

        errors: list[Exception] = []
        for agent in capable_agents:
            try:
                return await agent._runtime.run_async(request)
            except AgentError as e:
                errors.append(e)
                continue

        raise RoutingError(
            f"All capable agents ({len(capable_agents)}) failed",
            errors=errors,
        )

    def _run_custom(self, request: AgentRequest) -> AgentResponse:
        """Use custom routing function."""
        if not self.custom_router:
            raise ValueError("custom_router not configured")

        result = self.custom_router(request, self.agents)
        return result.agent._runtime.run(request)

    async def _run_custom_async(self, request: AgentRequest) -> AgentResponse:
        """Use custom routing function (async)."""
        if not self.custom_router:
            raise ValueError("custom_router not configured")

        result = self.custom_router(request, self.agents)
        return await result.agent._runtime.run_async(request)

    def _get_ordered_agents(self) -> list[Agent]:
        """Get agents in order based on strategy."""
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            # Rotate starting point
            start = self._round_robin_index
            self._round_robin_index = (start + 1) % len(self.agents)
            return self.agents[start:] + self.agents[:start]

        if self.strategy == RoutingStrategy.CHEAPEST:
            def get_cost(agent: Agent) -> float:
                model = resolve_model(agent.model)
                pricing = PRICING.get(model, {})
                return pricing.get("input", float("inf")) + pricing.get("output", float("inf"))

            return sorted(self.agents, key=get_cost)

        return self.agents

    def __repr__(self) -> str:
        return f"AgentRouter(agents={len(self.agents)}, strategy={self.strategy.value})"
