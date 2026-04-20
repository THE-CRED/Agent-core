"""
Main Agent class.

The primary interface for interacting with LLM providers.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agent.execution.runtime import ExecutionRuntime
from agent.messages import AgentRequest, Message
from agent.middleware import Middleware, MiddlewareChain
from agent.providers.base import BaseProvider
from agent.providers.registry import get_provider
from agent.response import AgentResponse
from agent.stream import AsyncStreamResponse, StreamResponse
from agent.tools import Tool
from agent.types.config import AgentConfig, RetryConfig

if TYPE_CHECKING:
    from agent.session import Session


class Agent:
    """
    A provider-agnostic LLM agent.

    The Agent class is the main entry point for interacting with LLM providers.
    It provides a unified interface for text generation, streaming, structured
    outputs, and tool calling across multiple providers.

    Example:
        ```python
        from agent import Agent

        agent = Agent(provider="openai", model="gpt-4o")
        response = agent.run("Hello, world!")
        print(response.text)
        ```
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[Tool] | None = None,
        middleware: list[Middleware] | None = None,
        default_system: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize an Agent.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4o", "claude-sonnet")
            api_key: API key (defaults to environment variable)
            base_url: Custom base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient errors
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            tools: List of tools available to the agent
            middleware: List of middleware to apply
            default_system: Default system prompt
            **kwargs: Additional provider-specific options
        """
        # Build configuration
        self.config = AgentConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            default_system=default_system,
            extra=kwargs,
        )

        # Initialize provider
        self._provider: BaseProvider = get_provider(
            provider,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            **kwargs,
        )

        # Store tools
        self._tools = tools or []

        # Build middleware chain
        self._middleware = MiddlewareChain(middleware or [])

        # Create execution runtime
        self._runtime = ExecutionRuntime(
            provider=self._provider,
            config=self.config,
            tools=self._tools,
            middleware=self._middleware,
            retry_config=RetryConfig(max_retries=max_retries),
        )

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.config.provider

    @property
    def model(self) -> str:
        """Get the model name."""
        return self.config.model

    @property
    def tools(self) -> list[Tool]:
        """Get registered tools."""
        return self._tools

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
        Execute a synchronous request.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            stop: Stop sequences
            metadata: Request metadata

        Returns:
            AgentResponse with the model's response
        """
        request = self._build_request(
            input=input,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            metadata=metadata,
        )
        return self._runtime.run(request)

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
        Execute an asynchronous request.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            stop: Stop sequences
            metadata: Request metadata

        Returns:
            AgentResponse with the model's response
        """
        request = self._build_request(
            input=input,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            metadata=metadata,
        )
        return await self._runtime.run_async(request)

    def stream(
        self,
        input: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StreamResponse:
        """
        Execute a streaming request.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            stop: Stop sequences
            metadata: Request metadata

        Returns:
            StreamResponse iterator yielding events
        """
        request = self._build_request(
            input=input,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            metadata=metadata,
        )
        return self._runtime.stream(request)

    async def stream_async(
        self,
        input: str | None = None,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncStreamResponse:
        """
        Execute an async streaming request.

        Args:
            input: User input text
            messages: Optional message history
            system: System prompt (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            stop: Stop sequences
            metadata: Request metadata

        Returns:
            AsyncStreamResponse iterator yielding events
        """
        request = self._build_request(
            input=input,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            metadata=metadata,
        )
        return await self._runtime.stream_async(request)

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
        Execute a request expecting structured JSON output.

        Args:
            input: User input text
            schema: Pydantic model or JSON schema for output
            messages: Optional message history
            system: System prompt (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            metadata: Request metadata

        Returns:
            AgentResponse with parsed output in response.output
        """
        request = self._build_request(
            input=input,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        return self._runtime.run(request, schema=schema)

    async def json_async(
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
        Execute an async request expecting structured JSON output.

        Args:
            input: User input text
            schema: Pydantic model or JSON schema for output
            messages: Optional message history
            system: System prompt (overrides default)
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens (overrides default)
            metadata: Request metadata

        Returns:
            AgentResponse with parsed output in response.output
        """
        request = self._build_request(
            input=input,
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        return await self._runtime.run_async(request, schema=schema)

    def session(
        self,
        session_id: str | None = None,
        system: str | None = None,
    ) -> "Session":
        """
        Create a new session for multi-turn conversation.

        Args:
            session_id: Optional session identifier
            system: System prompt for this session

        Returns:
            Session instance
        """
        from agent.session import Session

        return Session(
            agent=self,
            session_id=session_id,
            system=system or self.config.default_system,
        )

    def with_config(self, **kwargs: Any) -> "Agent":
        """
        Create a new Agent with modified configuration.

        Args:
            **kwargs: Configuration overrides

        Returns:
            New Agent instance with modified config
        """
        new_config = self.config.with_overrides(**kwargs)
        return Agent(
            provider=new_config.provider,
            model=new_config.model,
            api_key=new_config.api_key,
            base_url=new_config.base_url,
            timeout=new_config.timeout,
            max_retries=new_config.max_retries,
            temperature=new_config.temperature,
            max_tokens=new_config.max_tokens,
            top_p=new_config.top_p,
            tools=self._tools,
            middleware=self._middleware.middlewares,
            default_system=new_config.default_system,
            **new_config.extra,
        )

    def _build_request(
        self,
        input: str | None = None,
        messages: list[Message] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRequest:
        """Build a normalized request from parameters."""
        return AgentRequest(
            input=input,
            messages=messages or [],
            system=system or self.config.default_system,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=top_p or self.config.top_p,
            stop=stop,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"Agent(provider={self.provider!r}, model={self.model!r})"
