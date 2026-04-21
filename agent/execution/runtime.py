"""
Main execution runtime.

Orchestrates the complete request lifecycle including middleware,
retries, tool loops, and structured output handling.
"""

import time
from typing import Any

from pydantic import BaseModel

from agent.errors import UnsupportedFeatureError
from agent.execution.retries import RetryHandler
from agent.execution.structured_output import StructuredOutputHandler
from agent.execution.tool_loop import ToolLoop
from agent.messages import AgentRequest
from agent.middleware import MiddlewareChain
from agent.providers.base import BaseProvider
from agent.response import AgentResponse
from agent.stream import AsyncStreamResponse, StreamResponse
from agent.tools import Tool
from agent.types.config import AgentConfig, RetryConfig, ToolLoopConfig, estimate_cost


class ExecutionRuntime:
    """
    Core execution runtime for Agent.

    Handles the complete lifecycle of a request:
    1. Middleware preprocessing
    2. Request preparation
    3. Retry handling
    4. Tool loop orchestration
    5. Structured output parsing
    6. Middleware postprocessing
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: AgentConfig,
        tools: list[Tool] | None = None,
        middleware: MiddlewareChain | None = None,
        retry_config: RetryConfig | None = None,
        tool_loop_config: ToolLoopConfig | None = None,
    ):
        self.provider = provider
        self.config = config
        self.tools = tools or []
        self.middleware = middleware or MiddlewareChain()
        self.retry_handler = RetryHandler(
            retry_config or RetryConfig(max_retries=config.max_retries)
        )
        self.tool_loop = ToolLoop(self.tools, tool_loop_config) if self.tools else None

    def run(
        self,
        request: AgentRequest,
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Execute a request synchronously.

        Args:
            request: The normalized request
            schema: Optional output schema

        Returns:
            Normalized response
        """
        start_time = time.time()

        # Run middleware before hooks
        request = self.middleware.run_before(request)

        try:
            # Handle structured output
            output_handler = None
            if schema:
                output_handler = StructuredOutputHandler(schema)
                if not self.provider.supports_native_schema():
                    # Add schema instructions to system prompt
                    schema_prompt = output_handler.get_system_prompt_addition()
                    if request.system:
                        request.system = f"{request.system}\n\n{schema_prompt}"
                    else:
                        request.system = schema_prompt
                else:
                    request.schema = output_handler.get_json_schema()

            # Add tools to request
            if self.tools:
                if not self.provider.supports_tools():
                    raise UnsupportedFeatureError(
                        f"Provider '{self.provider.name}' does not support tools",
                        feature="tools",
                        provider=self.provider.name,
                    )
                request.tools = [t.spec for t in self.tools]

            # Execute with retries
            tool_loop = self.tool_loop
            if tool_loop and self.tools:
                # Run tool loop
                response = self.retry_handler.execute(
                    lambda: tool_loop.run_loop(request, self.provider.run)
                )
            else:
                # Simple execution
                response = self.retry_handler.execute(lambda: self.provider.run(request))

            # Parse structured output
            if output_handler and response.text:
                try:
                    response.output = output_handler.parse_response(response.text)
                except Exception:
                    # Attach error info but don't fail
                    response.output = None

            # Calculate latency and cost
            response.latency_ms = (time.time() - start_time) * 1000
            if response.usage:
                response.cost_estimate = estimate_cost(
                    self.config.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            # Run middleware after hooks
            response = self.middleware.run_after(request, response)

            return response

        except Exception as e:
            # Run middleware error hooks
            handled_error = self.middleware.run_on_error(request, e)
            if handled_error is None:
                # Error was suppressed
                return AgentResponse(
                    text="",
                    provider=self.provider.name,
                    model=self.config.model,
                )
            raise handled_error from e

    async def run_async(
        self,
        request: AgentRequest,
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Execute a request asynchronously.

        Args:
            request: The normalized request
            schema: Optional output schema

        Returns:
            Normalized response
        """
        start_time = time.time()

        request = self.middleware.run_before(request)

        try:
            output_handler = None
            if schema:
                output_handler = StructuredOutputHandler(schema)
                if not self.provider.supports_native_schema():
                    schema_prompt = output_handler.get_system_prompt_addition()
                    if request.system:
                        request.system = f"{request.system}\n\n{schema_prompt}"
                    else:
                        request.system = schema_prompt
                else:
                    request.schema = output_handler.get_json_schema()

            if self.tools:
                if not self.provider.supports_tools():
                    raise UnsupportedFeatureError(
                        f"Provider '{self.provider.name}' does not support tools",
                        feature="tools",
                        provider=self.provider.name,
                    )
                request.tools = [t.spec for t in self.tools]

            tool_loop = self.tool_loop
            if tool_loop and self.tools:
                response = await self.retry_handler.execute_async(
                    lambda: tool_loop.run_loop_async(request, self.provider.run_async)
                )
            else:
                response = await self.retry_handler.execute_async(
                    lambda: self.provider.run_async(request)
                )

            if output_handler and response.text:
                try:
                    response.output = output_handler.parse_response(response.text)
                except Exception:
                    response.output = None

            response.latency_ms = (time.time() - start_time) * 1000
            if response.usage:
                response.cost_estimate = estimate_cost(
                    self.config.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            response = self.middleware.run_after(request, response)
            return response

        except Exception as e:
            handled_error = self.middleware.run_on_error(request, e)
            if handled_error is None:
                return AgentResponse(
                    text="",
                    provider=self.provider.name,
                    model=self.config.model,
                )
            raise handled_error from e

    def stream(self, request: AgentRequest) -> StreamResponse:
        """
        Execute a streaming request.

        Args:
            request: The normalized request

        Returns:
            Stream response iterator
        """
        request = self.middleware.run_before(request)

        if not self.provider.supports_streaming():
            raise UnsupportedFeatureError(
                f"Provider '{self.provider.name}' does not support streaming",
                feature="streaming",
                provider=self.provider.name,
            )

        if self.tools:
            request.tools = [t.spec for t in self.tools]

        events = self.provider.stream(request)
        return StreamResponse(
            _events=events,
            provider=self.provider.name,
            model=self.config.model,
        )

    async def stream_async(self, request: AgentRequest) -> AsyncStreamResponse:
        """
        Execute an async streaming request.

        Args:
            request: The normalized request

        Returns:
            Async stream response iterator
        """
        request = self.middleware.run_before(request)

        if not self.provider.supports_streaming():
            raise UnsupportedFeatureError(
                f"Provider '{self.provider.name}' does not support streaming",
                feature="streaming",
                provider=self.provider.name,
            )

        if self.tools:
            request.tools = [t.spec for t in self.tools]

        events = self.provider.stream_async(request)
        return AsyncStreamResponse(
            events=events,
            provider=self.provider.name,
            model=self.config.model,
        )
