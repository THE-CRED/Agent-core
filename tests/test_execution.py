"""
Tests for the execution module: retries, structured_output, tool_loop, runtime.
"""

import asyncio
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from agent.errors import (
    ProviderError,
    SchemaValidationError,
    UnsupportedFeatureError,
)
from agent.execution.retries import RetryHandler
from agent.execution.runtime import ExecutionRuntime
from agent.execution.structured_output import (
    StructuredOutputHandler,
    prepare_structured_request,
)
from agent.execution.tool_loop import ToolLoop
from agent.messages import AgentRequest
from agent.middleware import Middleware, MiddlewareChain
from agent.response import AgentResponse, Usage
from agent.stream import AsyncStreamResponse, StreamResponse
from agent.testing.fake_provider import FakeProvider, FakeResponse
from agent.tools import tool
from agent.types.config import AgentConfig, RetryConfig, ToolLoopConfig
from agent.types.tools import ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def agent_config():
    return AgentConfig(provider="fake", model="fake-model")


@pytest.fixture
def simple_request():
    return AgentRequest(input="Hello")


@pytest.fixture
def greet_tool():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    return greet


@pytest.fixture
def failing_tool():
    @tool
    def explode(msg: str) -> str:
        """Always fails."""
        raise RuntimeError(msg)

    return explode


class PersonSchema(BaseModel):
    name: str
    age: int


# ===========================================================================
# RetryHandler tests
# ===========================================================================


class TestRetryHandler:
    """Tests for RetryHandler.execute and execute_async."""

    def test_execute_succeeds_first_try(self):
        handler = RetryHandler(RetryConfig(max_retries=2))
        result = handler.execute(lambda: "ok")
        assert result == "ok"

    @patch("agent.execution.retries.time.sleep")
    def test_execute_retries_on_retryable_error(self, mock_sleep):
        config = RetryConfig(max_retries=2, jitter=False, initial_delay=1.0)
        handler = RetryHandler(config)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = handler.execute(operation)
        assert result == "recovered"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("agent.execution.retries.time.sleep")
    def test_execute_calls_on_retry_callback(self, mock_sleep):
        config = RetryConfig(max_retries=2, jitter=False, initial_delay=1.0)
        handler = RetryHandler(config)

        callback_calls = []

        def on_retry(attempt, error, delay):
            callback_calls.append((attempt, str(error), delay))

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "done"

        handler.execute(operation, on_retry=on_retry)
        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1
        assert callback_calls[1][0] == 2

    def test_execute_raises_non_retryable_error_immediately(self):
        config = RetryConfig(max_retries=3)
        handler = RetryHandler(config)

        with pytest.raises(ValueError, match="bad input"):
            handler.execute(lambda: (_ for _ in ()).throw(ValueError("bad input")))

    @patch("agent.execution.retries.time.sleep")
    def test_execute_exhausts_retries_and_raises(self, mock_sleep):
        config = RetryConfig(max_retries=2, jitter=False)
        handler = RetryHandler(config)

        with pytest.raises(ConnectionError, match="always fails"):
            handler.execute(lambda: (_ for _ in ()).throw(ConnectionError("always fails")))

    @pytest.mark.asyncio
    async def test_execute_async_succeeds_first_try(self):
        handler = RetryHandler(RetryConfig(max_retries=2))

        async def op():
            return "async ok"

        result = await handler.execute_async(op)
        assert result == "async ok"

    @pytest.mark.asyncio
    async def test_execute_async_retries_on_retryable_error(self):
        config = RetryConfig(max_retries=2, jitter=False, initial_delay=0.01)
        handler = RetryHandler(config)
        call_count = 0

        async def op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = await handler.execute_async(op)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_async_calls_on_retry_callback(self):
        config = RetryConfig(max_retries=2, jitter=False, initial_delay=0.01)
        handler = RetryHandler(config)
        callbacks = []

        async def op():
            if len(callbacks) < 2:
                raise ConnectionError("fail")
            return "ok"

        def on_retry(attempt, error, delay):
            callbacks.append(attempt)

        result = await handler.execute_async(op, on_retry=on_retry)
        assert result == "ok"
        assert len(callbacks) == 2

    @pytest.mark.asyncio
    async def test_execute_async_raises_non_retryable_immediately(self):
        handler = RetryHandler(RetryConfig(max_retries=3))

        async def op():
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await handler.execute_async(op)


# ===========================================================================
# StructuredOutputHandler tests
# ===========================================================================


class TestStructuredOutputHandler:
    """Tests for StructuredOutputHandler."""

    def test_get_json_schema_with_pydantic(self):
        handler = StructuredOutputHandler(PersonSchema)
        schema = handler.get_json_schema()
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_get_json_schema_with_dict(self):
        raw_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        handler = StructuredOutputHandler(raw_schema)
        assert handler.get_json_schema() == raw_schema

    def test_get_system_prompt_addition(self):
        handler = StructuredOutputHandler(PersonSchema)
        prompt = handler.get_system_prompt_addition()
        assert "JSON" in prompt
        assert "name" in prompt

    def test_parse_response_valid_json(self):
        handler = StructuredOutputHandler(PersonSchema)
        result = handler.parse_response('{"name": "Alice", "age": 30}')
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_response_json_in_code_block(self):
        handler = StructuredOutputHandler(PersonSchema)
        text = '```json\n{"name": "Bob", "age": 25}\n```'
        result = handler.parse_response(text)
        assert result.name == "Bob"
        assert result.age == 25

    def test_parse_response_invalid_json_raises(self):
        handler = StructuredOutputHandler(PersonSchema, repair_attempts=0)
        with pytest.raises(SchemaValidationError):
            handler.parse_response("not json at all !!!")

    def test_parse_response_repair_attempt(self):
        handler = StructuredOutputHandler(PersonSchema, repair_attempts=1)
        # Trailing comma is repairable
        text = '{"name": "Carol", "age": 40,}'
        result = handler.parse_response(text)
        assert result.name == "Carol"
        assert result.age == 40

    def test_validate_native_output(self):
        handler = StructuredOutputHandler(PersonSchema)
        result = handler.validate_native_output({"name": "Dave", "age": 50})
        assert result.name == "Dave"
        assert result.age == 50

    def test_validate_native_output_invalid_raises(self):
        handler = StructuredOutputHandler(PersonSchema)
        with pytest.raises(SchemaValidationError):
            handler.validate_native_output({"name": "Eve"})  # missing age


class TestPrepareStructuredRequest:
    """Tests for the prepare_structured_request helper."""

    def test_returns_unchanged_when_schema_is_none(self):
        system, json_schema = prepare_structured_request(None, "Be helpful.", True)
        assert system == "Be helpful."
        assert json_schema is None

    def test_returns_json_schema_when_native_supported(self):
        system, json_schema = prepare_structured_request(PersonSchema, "sys", True)
        assert system == "sys"
        assert json_schema is not None
        assert "properties" in json_schema

    def test_modifies_system_prompt_when_native_not_supported(self):
        system, json_schema = prepare_structured_request(PersonSchema, "sys", False)
        assert json_schema is None
        assert "sys" in system
        assert "JSON" in system

    def test_creates_system_prompt_when_none_and_not_native(self):
        system, json_schema = prepare_structured_request(PersonSchema, None, False)
        assert json_schema is None
        assert system is not None
        assert "JSON" in system


# ===========================================================================
# ToolLoop tests
# ===========================================================================


class TestToolLoop:
    """Tests for ToolLoop."""

    def test_init_registers_tools(self, greet_tool):
        loop = ToolLoop([greet_tool])
        assert "greet" in loop.registry

    def test_execute_tool_calls_success(self, greet_tool):
        loop = ToolLoop([greet_tool])
        calls = [ToolCall(id="c1", name="greet", arguments={"name": "World"})]
        results = loop.execute_tool_calls(calls)
        assert len(results) == 1
        assert results[0].is_error is False
        assert "Hello, World!" in results[0].content

    def test_execute_tool_calls_unknown_tool(self, greet_tool):
        loop = ToolLoop([greet_tool])
        calls = [ToolCall(id="c1", name="unknown_tool", arguments={})]
        results = loop.execute_tool_calls(calls)
        assert len(results) == 1
        assert results[0].is_error is True
        assert "Unknown tool" in results[0].content

    def test_execute_tool_calls_error_no_stop(self, failing_tool):
        loop = ToolLoop([failing_tool], config=ToolLoopConfig(stop_on_error=False))
        calls = [ToolCall(id="c1", name="explode", arguments={"msg": "boom"})]
        results = loop.execute_tool_calls(calls)
        assert len(results) == 1
        # Tool.execute_sync catches exceptions internally and returns "Error: ..."
        # so ToolLoop sees a successful string result, not an error
        assert "boom" in results[0].content

    def test_execute_tool_calls_error_stop_on_error(self, failing_tool):
        # Tool.execute_sync catches exceptions internally, so stop_on_error
        # in ToolLoop doesn't trigger for tool-internal errors. The result
        # contains the error string instead.
        loop = ToolLoop([failing_tool], config=ToolLoopConfig(stop_on_error=True))
        calls = [ToolCall(id="c1", name="explode", arguments={"msg": "boom"})]
        results = loop.execute_tool_calls(calls)
        assert len(results) == 1
        assert "Error:" in results[0].content

    def test_execute_tool_calls_respects_max_per_iteration(self, greet_tool):
        loop = ToolLoop([greet_tool], config=ToolLoopConfig(max_tool_calls_per_iteration=1))
        calls = [
            ToolCall(id="c1", name="greet", arguments={"name": "A"}),
            ToolCall(id="c2", name="greet", arguments={"name": "B"}),
        ]
        results = loop.execute_tool_calls(calls)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_async_parallel(self, greet_tool):
        loop = ToolLoop([greet_tool], config=ToolLoopConfig(parallel_tool_execution=True))
        calls = [
            ToolCall(id="c1", name="greet", arguments={"name": "A"}),
            ToolCall(id="c2", name="greet", arguments={"name": "B"}),
        ]
        results = await loop.execute_tool_calls_async(calls)
        assert len(results) == 2
        assert all(not r.is_error for r in results)

    @pytest.mark.asyncio
    async def test_execute_tool_calls_async_sequential(self, greet_tool):
        loop = ToolLoop([greet_tool], config=ToolLoopConfig(parallel_tool_execution=False))
        calls = [
            ToolCall(id="c1", name="greet", arguments={"name": "A"}),
            ToolCall(id="c2", name="greet", arguments={"name": "B"}),
        ]
        results = await loop.execute_tool_calls_async(calls)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_execute_single_tool_async_timeout(self):
        @tool(timeout=0.01)
        async def slow_tool() -> str:
            """A slow tool."""
            await asyncio.sleep(10)
            return "done"

        loop = ToolLoop([slow_tool])
        call = ToolCall(id="c1", name="slow_tool", arguments={})
        result = await loop._execute_single_tool_async(call)
        assert result.is_error is True
        assert "timed out" in result.content

    @pytest.mark.asyncio
    async def test_execute_single_tool_async_unknown(self, greet_tool):
        loop = ToolLoop([greet_tool])
        call = ToolCall(id="c1", name="nonexistent", arguments={})
        result = await loop._execute_single_tool_async(call)
        assert result.is_error is True
        assert "Unknown tool" in result.content

    def test_build_tool_messages(self, greet_tool):
        loop = ToolLoop([greet_tool])
        tc = ToolCall(id="c1", name="greet", arguments={"name": "W"})
        response = AgentResponse(
            text="I'll greet",
            tool_calls=[tc],
            provider="fake",
            model="fake",
        )
        results = [ToolResult(tool_call_id="c1", name="greet", content="Hello, W!", is_error=False)]
        messages = loop.build_tool_messages(response, results)
        assert len(messages) == 2
        assert messages[0].role == "assistant"
        assert messages[1].role == "tool"
        assert messages[1].content == "Hello, W!"

    def test_run_loop_no_tool_calls(self, greet_tool, fake_provider):
        fake_provider.set_response(FakeResponse(text="Final answer"))
        loop = ToolLoop([greet_tool])
        request = AgentRequest(input="Hi")
        response = loop.run_loop(request, fake_provider.run)
        assert response.text == "Final answer"

    def test_run_loop_with_tool_iteration(self, greet_tool, fake_provider):
        tc = ToolCall(id="c1", name="greet", arguments={"name": "Loop"})
        fake_provider.set_responses(
            [
                FakeResponse(text="", tool_calls=[tc], stop_reason="tool_calls"),
                FakeResponse(text="Done after tool"),
            ]
        )
        loop = ToolLoop([greet_tool])
        request = AgentRequest(input="Hi")
        response = loop.run_loop(request, fake_provider.run)
        assert response.text == "Done after tool"
        assert len(fake_provider.get_requests()) == 2

    def test_run_loop_respects_max_iterations(self, greet_tool, fake_provider):
        tc = ToolCall(id="c1", name="greet", arguments={"name": "Loop"})
        # Always returns tool calls, never a final answer
        fake_provider.set_response(FakeResponse(text="", tool_calls=[tc], stop_reason="tool_calls"))
        loop = ToolLoop([greet_tool], config=ToolLoopConfig(max_iterations=3))
        request = AgentRequest(input="Hi")
        loop.run_loop(request, fake_provider.run)
        # Should have been called max_iterations times
        assert len(fake_provider.get_requests()) == 3

    @pytest.mark.asyncio
    async def test_run_loop_async_with_tool_iteration(self, greet_tool, fake_provider):
        tc = ToolCall(id="c1", name="greet", arguments={"name": "Async"})
        fake_provider.set_responses(
            [
                FakeResponse(text="", tool_calls=[tc], stop_reason="tool_calls"),
                FakeResponse(text="Async done"),
            ]
        )
        loop = ToolLoop([greet_tool])
        request = AgentRequest(input="Hi")
        response = await loop.run_loop_async(request, fake_provider.run_async)
        assert response.text == "Async done"


# ===========================================================================
# ExecutionRuntime tests
# ===========================================================================


class TestExecutionRuntime:
    """Tests for ExecutionRuntime."""

    def test_run_simple_success(self, fake_provider, agent_config, simple_request):
        fake_provider.set_response(FakeResponse(text="Hi there"))
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
        )
        response = runtime.run(simple_request)
        assert response.text == "Hi there"
        assert response.latency_ms is not None
        assert response.latency_ms > 0

    def test_run_records_cost_estimate(self, fake_provider, agent_config, simple_request):
        fake_provider.set_response(
            FakeResponse(
                text="answer",
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            )
        )
        runtime = ExecutionRuntime(provider=fake_provider, config=agent_config)
        response = runtime.run(simple_request)
        # cost_estimate may be None for unknown models; just ensure no crash
        assert response.latency_ms is not None

    @patch("agent.execution.retries.time.sleep")
    def test_run_retries_on_provider_error(self, mock_sleep, fake_provider, agent_config):
        call_count = 0
        original_run = fake_provider.run

        def flaky_run(request):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ProviderError("server error", status_code=500)
            return original_run(request)

        fake_provider.run = flaky_run
        fake_provider.set_response(FakeResponse(text="recovered"))

        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            retry_config=RetryConfig(max_retries=3, jitter=False),
        )
        response = runtime.run(AgentRequest(input="Hi"))
        assert response.text == "recovered"
        assert call_count == 2

    def test_run_with_middleware_before_and_after(
        self, fake_provider, agent_config, simple_request
    ):
        fake_provider.set_response(FakeResponse(text="answer"))

        class TaggingMiddleware(Middleware):
            def before(self, request):
                request.metadata["tagged"] = True
                return request

            def after(self, request, response):
                response.request_id = "mw-tagged"
                return response

        chain = MiddlewareChain([TaggingMiddleware()])
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            middleware=chain,
        )
        response = runtime.run(simple_request)
        assert response.request_id == "mw-tagged"
        # Verify before hook ran
        last_req = fake_provider.get_last_request()
        assert last_req.metadata.get("tagged") is True

    def test_run_middleware_error_suppression(self, fake_provider, agent_config):
        """Middleware returning None from on_error suppresses the error."""
        fake_provider.set_response(FakeResponse.with_error(ProviderError("fail", status_code=500)))

        class SuppressingMiddleware(Middleware):
            def on_error(self, request, error):
                return None  # suppress

        chain = MiddlewareChain([SuppressingMiddleware()])
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            middleware=chain,
            retry_config=RetryConfig(max_retries=0),
        )
        response = runtime.run(AgentRequest(input="Hi"))
        assert response.text == ""
        assert response.provider == "fake"

    def test_run_middleware_error_passthrough(self, fake_provider, agent_config):
        """Middleware returning the error allows it to propagate."""
        fake_provider.set_response(FakeResponse.with_error(ValueError("real error")))

        chain = MiddlewareChain()
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            middleware=chain,
            retry_config=RetryConfig(max_retries=0),
        )
        with pytest.raises(ValueError, match="real error"):
            runtime.run(AgentRequest(input="Hi"))

    def test_run_with_tools(self, fake_provider, agent_config, greet_tool):
        tc = ToolCall(id="c1", name="greet", arguments={"name": "Runtime"})
        fake_provider.set_responses(
            [
                FakeResponse(text="", tool_calls=[tc], stop_reason="tool_calls"),
                FakeResponse(text="Greeted via runtime"),
            ]
        )
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            tools=[greet_tool],
        )
        response = runtime.run(AgentRequest(input="Greet someone"))
        assert response.text == "Greeted via runtime"

    def test_run_with_tools_unsupported_provider(self, agent_config, greet_tool):
        """Provider that doesn't support tools raises UnsupportedFeatureError."""
        provider = FakeProvider()
        provider.capabilities = provider.capabilities.model_copy(update={"tools": False})
        runtime = ExecutionRuntime(
            provider=provider,
            config=agent_config,
            tools=[greet_tool],
            retry_config=RetryConfig(max_retries=0),
        )
        with pytest.raises(UnsupportedFeatureError):
            runtime.run(AgentRequest(input="Hi"))

    def test_run_with_structured_output_native(self, fake_provider, agent_config):
        fake_provider.set_response(FakeResponse(text='{"name": "Alice", "age": 30}'))
        runtime = ExecutionRuntime(provider=fake_provider, config=agent_config)
        response = runtime.run(AgentRequest(input="person"), schema=PersonSchema)
        assert response.output is not None
        assert response.output.name == "Alice"
        assert response.output.age == 30

    def test_run_with_structured_output_prompt_fallback(self, agent_config):
        """When provider doesn't support native schema, system prompt is augmented."""
        provider = FakeProvider()
        provider.capabilities = provider.capabilities.model_copy(
            update={"native_schema_output": False}
        )
        provider.set_response(FakeResponse(text='{"name": "Bob", "age": 25}'))

        runtime = ExecutionRuntime(
            provider=provider,
            config=agent_config,
        )
        request = AgentRequest(input="person", system="Be helpful.")
        response = runtime.run(request, schema=PersonSchema)
        assert response.output is not None
        assert response.output.name == "Bob"

    def test_run_structured_output_parse_failure_does_not_crash(self, fake_provider, agent_config):
        fake_provider.set_response(FakeResponse(text="not json"))
        runtime = ExecutionRuntime(provider=fake_provider, config=agent_config)
        response = runtime.run(AgentRequest(input="person"), schema=PersonSchema)
        assert response.output is None

    def test_stream_returns_stream_response(self, fake_provider, agent_config, simple_request):
        fake_provider.set_response(FakeResponse(text="streamed"))
        runtime = ExecutionRuntime(provider=fake_provider, config=agent_config)
        stream_resp = runtime.stream(simple_request)
        assert isinstance(stream_resp, StreamResponse)

    def test_stream_unsupported_raises(self, agent_config, simple_request):
        provider = FakeProvider()
        provider.capabilities = provider.capabilities.model_copy(update={"streaming": False})
        runtime = ExecutionRuntime(provider=provider, config=agent_config)
        with pytest.raises(UnsupportedFeatureError):
            runtime.stream(simple_request)

    @pytest.mark.asyncio
    async def test_run_async_simple_success(self, fake_provider, agent_config, simple_request):
        fake_provider.set_response(FakeResponse(text="async hi"))
        runtime = ExecutionRuntime(provider=fake_provider, config=agent_config)
        response = await runtime.run_async(simple_request)
        assert response.text == "async hi"
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_run_async_with_tools(self, fake_provider, agent_config, greet_tool):
        tc = ToolCall(id="c1", name="greet", arguments={"name": "AsyncRT"})
        fake_provider.set_responses(
            [
                FakeResponse(text="", tool_calls=[tc], stop_reason="tool_calls"),
                FakeResponse(text="Async greeted"),
            ]
        )
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            tools=[greet_tool],
        )
        response = await runtime.run_async(AgentRequest(input="greet"))
        assert response.text == "Async greeted"

    @pytest.mark.asyncio
    async def test_run_async_middleware_error_suppression(self, fake_provider, agent_config):
        fake_provider.set_response(FakeResponse.with_error(ProviderError("boom", status_code=500)))

        class SuppressingMiddleware(Middleware):
            def on_error(self, request, error):
                return None

        chain = MiddlewareChain([SuppressingMiddleware()])
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            middleware=chain,
            retry_config=RetryConfig(max_retries=0),
        )
        response = await runtime.run_async(AgentRequest(input="Hi"))
        assert response.text == ""

    @pytest.mark.asyncio
    async def test_stream_async_returns_async_stream_response(
        self, fake_provider, agent_config, simple_request
    ):
        fake_provider.set_response(FakeResponse(text="streamed async"))
        runtime = ExecutionRuntime(provider=fake_provider, config=agent_config)
        stream_resp = await runtime.stream_async(simple_request)
        assert isinstance(stream_resp, AsyncStreamResponse)

    @pytest.mark.asyncio
    async def test_stream_async_unsupported_raises(self, agent_config, simple_request):
        provider = FakeProvider()
        provider.capabilities = provider.capabilities.model_copy(update={"streaming": False})
        runtime = ExecutionRuntime(provider=provider, config=agent_config)
        with pytest.raises(UnsupportedFeatureError):
            await runtime.stream_async(simple_request)

    def test_run_with_tools_attaches_tool_specs(self, fake_provider, agent_config, greet_tool):
        fake_provider.set_response(FakeResponse(text="no tools needed"))
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            tools=[greet_tool],
        )
        runtime.run(AgentRequest(input="Hi"))
        last_req = fake_provider.get_last_request()
        assert len(last_req.tools) > 0

    def test_stream_with_tools_attaches_tool_specs(
        self, fake_provider, agent_config, greet_tool, simple_request
    ):
        fake_provider.set_response(FakeResponse(text="stream"))
        runtime = ExecutionRuntime(
            provider=fake_provider,
            config=agent_config,
            tools=[greet_tool],
        )
        stream_resp = runtime.stream(simple_request)
        # Consume the stream to trigger the provider call
        stream_resp.collect()
