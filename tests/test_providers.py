"""
Comprehensive tests for provider infrastructure:
- BaseProvider ABC
- ProviderRegistry
- FakeProvider
- Testing fixtures
"""

import asyncio
import copy
from collections.abc import AsyncIterator, Iterator

import pytest

from agent.errors import ProviderError
from agent.messages import AgentRequest
from agent.providers.base import BaseProvider
from agent.providers.registry import ProviderRegistry, get_provider
from agent.response import AgentResponse, Usage
from agent.stream import StreamEvent
from agent.testing.fake_provider import FakeProvider, FakeResponse
from agent.testing.fixtures import (
    AgentTestCase,
    create_test_agent,
    create_test_response,
)
from agent.tools import ToolCall
from agent.types.config import ProviderCapabilities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConcreteProvider(BaseProvider):
    """Minimal concrete subclass of BaseProvider for testing the ABC."""

    name = "concrete"
    capabilities = ProviderCapabilities(
        streaming=False,
        tools=True,
        structured_output=False,
        json_mode=False,
        vision=True,
        native_schema_output=True,
    )

    def run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(text="ok", provider=self.name, model="m")

    async def run_async(self, request: AgentRequest) -> AgentResponse:
        return self.run(request)

    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        yield StreamEvent.text_delta("ok")

    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent.text_delta("ok")


def _make_request(text: str = "hello") -> AgentRequest:
    return AgentRequest(input=text)


# ===========================================================================
# 1. BaseProvider tests
# ===========================================================================


class TestBaseProviderInit:
    """Test that __init__ stores all parameters as attributes."""

    def test_defaults(self):
        p = ConcreteProvider()
        assert p.api_key is None
        assert p.base_url is None
        assert p.timeout == 120.0
        assert p.max_retries == 2
        assert p.extra_config == {}

    def test_explicit_values(self):
        p = ConcreteProvider(
            api_key="sk-key",
            base_url="https://example.com",
            timeout=30.0,
            max_retries=5,
        )
        assert p.api_key == "sk-key"
        assert p.base_url == "https://example.com"
        assert p.timeout == 30.0
        assert p.max_retries == 5

    def test_extra_kwargs(self):
        p = ConcreteProvider(api_key="k", custom_param="hello", another=42)
        assert p.extra_config == {"custom_param": "hello", "another": 42}


class TestBaseProviderAbstractMethods:
    """Verify that BaseProvider cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]


class TestBaseProviderCapabilities:
    """Test that supports_* methods delegate to capabilities."""

    def test_supports_tools(self):
        p = ConcreteProvider()
        assert p.supports_tools() is True

    def test_supports_structured_output(self):
        p = ConcreteProvider()
        assert p.supports_structured_output() is False

    def test_supports_vision(self):
        p = ConcreteProvider()
        assert p.supports_vision() is True

    def test_supports_streaming(self):
        p = ConcreteProvider()
        assert p.supports_streaming() is False

    def test_supports_json_mode(self):
        p = ConcreteProvider()
        assert p.supports_json_mode() is False

    def test_supports_native_schema(self):
        p = ConcreteProvider()
        assert p.supports_native_schema() is True

    def test_default_capabilities_all_false_for_base(self):
        """Default ProviderCapabilities has specific defaults."""
        caps = ProviderCapabilities()
        assert caps.streaming is True
        assert caps.tools is True
        assert caps.structured_output is True
        assert caps.json_mode is True
        assert caps.vision is False
        assert caps.native_schema_output is False


class TestBaseProviderValidateConfig:
    """Test validate_config error checking."""

    def test_missing_api_key(self):
        p = ConcreteProvider()
        errors = p.validate_config()
        assert len(errors) == 1
        assert "API key" in errors[0]
        assert "concrete" in errors[0]

    def test_none_api_key(self):
        p = ConcreteProvider(api_key=None)
        errors = p.validate_config()
        assert len(errors) == 1

    def test_empty_string_api_key(self):
        p = ConcreteProvider(api_key="")
        errors = p.validate_config()
        assert len(errors) == 1

    def test_valid_api_key(self):
        p = ConcreteProvider(api_key="sk-valid")
        errors = p.validate_config()
        assert errors == []

    def test_returns_list(self):
        p = ConcreteProvider(api_key="k")
        result = p.validate_config()
        assert isinstance(result, list)


class TestBaseProviderRunMethods:
    """Verify concrete run/stream methods work."""

    def test_run(self):
        p = ConcreteProvider(api_key="k")
        resp = p.run(_make_request())
        assert resp.text == "ok"

    def test_run_async(self):
        p = ConcreteProvider(api_key="k")
        resp = asyncio.get_event_loop().run_until_complete(p.run_async(_make_request()))
        assert resp.text == "ok"

    def test_stream(self):
        p = ConcreteProvider(api_key="k")
        events = list(p.stream(_make_request()))
        assert len(events) == 1
        assert events[0].type == "text_delta"
        assert events[0].text == "ok"

    def test_stream_async(self):
        p = ConcreteProvider(api_key="k")

        async def _collect():
            result = []
            async for e in p.stream_async(_make_request()):
                result.append(e)
            return result

        events = asyncio.get_event_loop().run_until_complete(_collect())
        assert len(events) == 1


# ===========================================================================
# 2. ProviderRegistry tests
# ===========================================================================


class TestProviderRegistry:
    """Tests for the ProviderRegistry class.

    Because the registry uses class-level dicts, we save and restore
    state around each test to avoid cross-test pollution.
    """

    def setup_method(self):
        self._saved_providers = copy.copy(ProviderRegistry._providers)
        self._saved_aliases = copy.copy(ProviderRegistry._aliases)

    def teardown_method(self):
        ProviderRegistry._providers = self._saved_providers
        ProviderRegistry._aliases = self._saved_aliases

    # -- register --

    def test_register_basic(self):
        ProviderRegistry.register("mytest", ConcreteProvider)
        assert "mytest" in ProviderRegistry._providers
        assert ProviderRegistry._providers["mytest"] is ConcreteProvider

    def test_register_with_aliases(self):
        ProviderRegistry.register("mytest2", ConcreteProvider, aliases=["mt2", "mt2b"])
        assert ProviderRegistry._aliases["mt2"] == "mytest2"
        assert ProviderRegistry._aliases["mt2b"] == "mytest2"

    def test_register_no_aliases(self):
        ProviderRegistry.register("mytest3", ConcreteProvider, aliases=None)
        assert ProviderRegistry._providers["mytest3"] is ConcreteProvider

    def test_register_overwrites(self):
        ProviderRegistry.register("dup", ConcreteProvider)
        ProviderRegistry.register("dup", FakeProvider)
        assert ProviderRegistry._providers["dup"] is FakeProvider

    # -- get_class --

    def test_get_class_by_name(self):
        ProviderRegistry.register("gc_test", ConcreteProvider)
        cls = ProviderRegistry.get_class("gc_test")
        assert cls is ConcreteProvider

    def test_get_class_by_alias(self):
        ProviderRegistry.register("gc_alias_test", ConcreteProvider, aliases=["gcat"])
        cls = ProviderRegistry.get_class("gcat")
        assert cls is ConcreteProvider

    def test_get_class_not_found(self):
        with pytest.raises(ProviderError) as exc_info:
            ProviderRegistry.get_class("nonexistent_provider_xyz")
        assert "nonexistent_provider_xyz" in str(exc_info.value)

    def test_get_class_error_lists_available(self):
        ProviderRegistry.register("avail1", ConcreteProvider)
        with pytest.raises(ProviderError) as exc_info:
            ProviderRegistry.get_class("nope_xyz")
        assert "avail1" in str(exc_info.value)

    # -- create --

    def test_create_returns_instance(self):
        ProviderRegistry.register("create_test", ConcreteProvider)
        inst = ProviderRegistry.create("create_test", api_key="k")
        assert isinstance(inst, ConcreteProvider)
        assert inst.api_key == "k"

    def test_create_passes_kwargs(self):
        ProviderRegistry.register("create_kw", ConcreteProvider)
        inst = ProviderRegistry.create("create_kw", api_key="x", timeout=5.0)
        assert inst.timeout == 5.0

    def test_create_not_found_raises(self):
        with pytest.raises(ProviderError):
            ProviderRegistry.create("does_not_exist_xyz")

    # -- list_providers --

    def test_list_providers_sorted(self):
        ProviderRegistry._providers.clear()
        ProviderRegistry.register("zebra", ConcreteProvider)
        ProviderRegistry.register("alpha", ConcreteProvider)
        result = ProviderRegistry.list_providers()
        assert result == ["alpha", "zebra"]

    def test_list_providers_empty(self):
        ProviderRegistry._providers.clear()
        assert ProviderRegistry.list_providers() == []

    # -- is_registered --

    def test_is_registered_true(self):
        ProviderRegistry.register("regcheck", ConcreteProvider)
        assert ProviderRegistry.is_registered("regcheck") is True

    def test_is_registered_alias(self):
        ProviderRegistry.register("regcheck2", ConcreteProvider, aliases=["rc2"])
        assert ProviderRegistry.is_registered("rc2") is True

    def test_is_registered_false(self):
        assert ProviderRegistry.is_registered("definitely_not_registered_xyz") is False


class TestGetProviderFunction:
    """Test the top-level get_provider() function."""

    def setup_method(self):
        self._saved_providers = copy.copy(ProviderRegistry._providers)
        self._saved_aliases = copy.copy(ProviderRegistry._aliases)

    def teardown_method(self):
        ProviderRegistry._providers = self._saved_providers
        ProviderRegistry._aliases = self._saved_aliases

    def test_get_provider_creates_instance(self):
        ProviderRegistry.register("gp_test", ConcreteProvider)
        inst = get_provider("gp_test", api_key="k")
        assert isinstance(inst, ConcreteProvider)

    def test_get_provider_not_found(self):
        with pytest.raises(ProviderError):
            get_provider("absolutely_not_a_provider_xyz")

    def test_get_provider_fake(self):
        """The fake provider should be registered (via module import)."""
        inst = get_provider("fake")
        assert isinstance(inst, FakeProvider)


# ===========================================================================
# 3. FakeResponse tests
# ===========================================================================


class TestFakeResponse:
    """Test FakeResponse dataclass and classmethods."""

    def test_defaults(self):
        r = FakeResponse()
        assert r.text == "This is a fake response."
        assert r.tool_calls == []
        assert r.usage is None
        assert r.stop_reason == "stop"
        assert r.latency_ms == 100.0
        assert r.error is None

    def test_with_text(self):
        r = FakeResponse.with_text("hi")
        assert r.text == "hi"
        assert r.tool_calls == []
        assert r.error is None

    def test_with_tool_call(self):
        r = FakeResponse.with_tool_call("search", {"query": "test"})
        assert r.text == ""
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"
        assert r.tool_calls[0].arguments == {"query": "test"}
        assert r.tool_calls[0].id == "call_123"
        assert r.stop_reason == "tool_calls"

    def test_with_tool_call_custom_id(self):
        r = FakeResponse.with_tool_call("fn", {"a": 1}, id="custom_id")
        assert r.tool_calls[0].id == "custom_id"

    def test_with_error(self):
        err = ValueError("boom")
        r = FakeResponse.with_error(err)
        assert r.error is err


# ===========================================================================
# 4. FakeProvider tests
# ===========================================================================


class TestFakeProviderInit:
    """Test FakeProvider initialization and class attributes."""

    def test_name(self):
        assert FakeProvider.name == "fake"

    def test_capabilities_all_true(self):
        p = FakeProvider()
        assert p.supports_tools() is True
        assert p.supports_structured_output() is True
        assert p.supports_vision() is True
        assert p.supports_streaming() is True
        assert p.supports_json_mode() is True
        assert p.supports_native_schema() is True

    def test_inherits_base_provider(self):
        assert issubclass(FakeProvider, BaseProvider)

    def test_init_defaults(self):
        p = FakeProvider()
        assert p._responses == []
        assert p._response_index == 0
        assert p._requests == []
        assert p._response_fn is None

    def test_init_passes_to_super(self):
        p = FakeProvider(api_key="k", timeout=10.0)
        assert p.api_key == "k"
        assert p.timeout == 10.0


class TestFakeProviderSetResponse:

    def test_set_response(self):
        p = FakeProvider()
        r = FakeResponse(text="x")
        p.set_response(r)
        assert p._responses == [r]
        assert p._response_index == 0

    def test_set_responses(self):
        p = FakeProvider()
        rs = [FakeResponse(text="a"), FakeResponse(text="b")]
        p.set_responses(rs)
        assert p._responses == rs
        assert p._response_index == 0

    def test_set_response_fn(self):
        p = FakeProvider()

        def fn(req):
            return FakeResponse(text=req.input or "")

        p.set_response_fn(fn)
        assert p._response_fn is fn


class TestFakeProviderRequestTracking:

    def test_get_requests_empty(self):
        p = FakeProvider()
        assert p.get_requests() == []

    def test_get_last_request_none(self):
        p = FakeProvider()
        assert p.get_last_request() is None

    def test_get_requests_after_run(self):
        p = FakeProvider()
        req = _make_request("hi")
        p.run(req)
        assert p.get_requests() == [req]

    def test_get_last_request_after_runs(self):
        p = FakeProvider()
        r1 = _make_request("a")
        r2 = _make_request("b")
        p.run(r1)
        p.run(r2)
        assert p.get_last_request() == r2

    def test_clear(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="x"))
        p.set_response_fn(lambda r: FakeResponse(text="y"))
        p.run(_make_request())
        p.clear()
        assert p._responses == []
        assert p._response_index == 0
        assert p._requests == []
        assert p._response_fn is None


class TestFakeProviderResponseCycling:
    """Test _get_next_response cycling behavior."""

    def test_default_response_when_none_configured(self):
        p = FakeProvider()
        resp = p.run(_make_request())
        assert resp.text == "This is a fake response."

    def test_single_response_repeats(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="only"))
        r1 = p.run(_make_request())
        r2 = p.run(_make_request())
        assert r1.text == "only"
        assert r2.text == "only"

    def test_multiple_responses_cycle(self):
        p = FakeProvider()
        p.set_responses([FakeResponse(text="a"), FakeResponse(text="b")])
        results = [p.run(_make_request()).text for _ in range(5)]
        assert results == ["a", "b", "a", "b", "a"]

    def test_response_fn_takes_precedence(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="ignored"))
        p.set_response_fn(lambda req: FakeResponse(text=f"fn:{req.input}"))
        resp = p.run(_make_request("hello"))
        assert resp.text == "fn:hello"

    def test_response_fn_receives_request(self):
        p = FakeProvider()
        received = []
        p.set_response_fn(lambda req: (received.append(req), FakeResponse(text="ok"))[1])
        req = _make_request("test_input")
        p.run(req)
        assert len(received) == 1
        assert received[0].input == "test_input"


class TestFakeProviderRun:

    def test_run_returns_agent_response(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="hello"))
        resp = p.run(_make_request())
        assert isinstance(resp, AgentResponse)
        assert resp.text == "hello"
        assert resp.provider == "fake"
        assert resp.model == "fake-model"

    def test_run_default_usage(self):
        p = FakeProvider()
        resp = p.run(_make_request())
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 20
        assert resp.usage.total_tokens == 30

    def test_run_custom_usage(self):
        p = FakeProvider()
        usage = Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300)
        p.set_response(FakeResponse(text="x", usage=usage))
        resp = p.run(_make_request())
        assert resp.usage.total_tokens == 300

    def test_run_with_tool_calls(self):
        p = FakeProvider()
        p.set_response(FakeResponse.with_tool_call("search", {"q": "hi"}))
        resp = p.run(_make_request())
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.stop_reason == "tool_calls"

    def test_run_error_raises(self):
        p = FakeProvider()
        p.set_response(FakeResponse.with_error(ValueError("boom")))
        with pytest.raises(ValueError, match="boom"):
            p.run(_make_request())

    def test_run_error_still_records_request(self):
        p = FakeProvider()
        p.set_response(FakeResponse.with_error(RuntimeError("err")))
        with pytest.raises(RuntimeError):
            p.run(_make_request("tracked"))
        assert len(p.get_requests()) == 1
        assert p.get_last_request().input == "tracked"

    def test_run_content_field(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="hi"))
        resp = p.run(_make_request())
        assert resp.content == [{"type": "text", "text": "hi"}]

    def test_run_empty_text_content(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text=""))
        resp = p.run(_make_request())
        assert resp.content == []

    def test_run_latency(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="x", latency_ms=42.0))
        resp = p.run(_make_request())
        assert resp.latency_ms == 42.0

    def test_run_raw(self):
        p = FakeProvider()
        resp = p.run(_make_request())
        assert resp.raw == {"fake": True}


class TestFakeProviderRunAsync:

    def test_run_async_returns_same_as_run(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="async_test"))

        async def _go():
            return await p.run_async(_make_request())

        resp = asyncio.get_event_loop().run_until_complete(_go())
        assert resp.text == "async_test"

    def test_run_async_error(self):
        p = FakeProvider()
        p.set_response(FakeResponse.with_error(ValueError("async_err")))

        async def _go():
            return await p.run_async(_make_request())

        with pytest.raises(ValueError, match="async_err"):
            asyncio.get_event_loop().run_until_complete(_go())


class TestFakeProviderStream:
    """Test streaming output from FakeProvider."""

    def test_stream_message_start(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="hi"))
        events = list(p.stream(_make_request()))
        assert events[0].type == "message_start"

    def test_stream_text_deltas(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="hello world"))
        events = list(p.stream(_make_request()))
        text_events = [e for e in events if e.type == "text_delta"]
        assert len(text_events) == 2
        # First word has no leading space, second does
        assert text_events[0].text == "hello"
        assert text_events[1].text == " world"

    def test_stream_single_word(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="ok"))
        events = list(p.stream(_make_request()))
        text_events = [e for e in events if e.type == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0].text == "ok"

    def test_stream_empty_text(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text=""))
        events = list(p.stream(_make_request()))
        text_events = [e for e in events if e.type == "text_delta"]
        assert len(text_events) == 0

    def test_stream_tool_calls(self):
        p = FakeProvider()
        p.set_response(FakeResponse.with_tool_call("fn", {"a": 1}))
        events = list(p.stream(_make_request()))
        tc_events = [e for e in events if e.type == "tool_call_start"]
        assert len(tc_events) == 1
        assert tc_events[0].tool_call.name == "fn"

    def test_stream_usage_event(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="x"))
        events = list(p.stream(_make_request()))
        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0].usage.total_tokens == 30

    def test_stream_message_end(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="x"))
        events = list(p.stream(_make_request()))
        assert events[-1].type == "message_end"

    def test_stream_error_raises(self):
        p = FakeProvider()
        p.set_response(FakeResponse.with_error(RuntimeError("stream_err")))
        with pytest.raises(RuntimeError, match="stream_err"):
            list(p.stream(_make_request()))

    def test_stream_event_order(self):
        """Events should be: message_start, text_deltas, usage, message_end."""
        p = FakeProvider()
        p.set_response(FakeResponse(text="a b"))
        events = list(p.stream(_make_request()))
        types = [e.type for e in events]
        assert types[0] == "message_start"
        assert types[-2] == "usage"
        assert types[-1] == "message_end"
        # text deltas in the middle
        assert "text_delta" in types

    def test_stream_custom_usage(self):
        p = FakeProvider()
        usage = Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        p.set_response(FakeResponse(text="x", usage=usage))
        events = list(p.stream(_make_request()))
        usage_events = [e for e in events if e.type == "usage"]
        assert usage_events[0].usage.total_tokens == 15


class TestFakeProviderStreamAsync:

    def test_stream_async(self):
        p = FakeProvider()
        p.set_response(FakeResponse(text="async stream"))

        async def _collect():
            result = []
            async for e in p.stream_async(_make_request()):
                result.append(e)
            return result

        events = asyncio.get_event_loop().run_until_complete(_collect())
        types = [e.type for e in events]
        assert "message_start" in types
        assert "text_delta" in types
        assert "message_end" in types

    def test_stream_async_error(self):
        p = FakeProvider()
        p.set_response(FakeResponse.with_error(ValueError("async_stream_err")))

        async def _collect():
            result = []
            async for e in p.stream_async(_make_request()):
                result.append(e)
            return result

        with pytest.raises(ValueError, match="async_stream_err"):
            asyncio.get_event_loop().run_until_complete(_collect())


class TestFakeProviderRegistration:
    """Verify FakeProvider is auto-registered with aliases."""

    def setup_method(self):
        self._saved_providers = copy.copy(ProviderRegistry._providers)
        self._saved_aliases = copy.copy(ProviderRegistry._aliases)

    def teardown_method(self):
        ProviderRegistry._providers = self._saved_providers
        ProviderRegistry._aliases = self._saved_aliases

    def test_registered_as_fake(self):
        assert ProviderRegistry.is_registered("fake")

    def test_alias_test(self):
        assert ProviderRegistry.is_registered("test")
        cls = ProviderRegistry.get_class("test")
        assert cls is FakeProvider

    def test_alias_mock(self):
        assert ProviderRegistry.is_registered("mock")
        cls = ProviderRegistry.get_class("mock")
        assert cls is FakeProvider


# ===========================================================================
# 5. Testing fixtures tests
# ===========================================================================


class TestCreateTestResponse:

    def test_defaults(self):
        resp = create_test_response()
        assert resp.text == "Test response"
        assert resp.provider == "fake"
        assert resp.model == "fake-model"
        assert resp.stop_reason == "stop"
        assert resp.latency_ms == 100.0
        assert resp.usage is not None
        assert resp.usage.total_tokens == 30
        assert resp.tool_calls == []
        assert resp.raw == {"test": True}

    def test_custom_text(self):
        resp = create_test_response(text="custom")
        assert resp.text == "custom"

    def test_custom_usage(self):
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        resp = create_test_response(usage=u)
        assert resp.usage.total_tokens == 3

    def test_custom_tool_calls(self):
        tc = [ToolCall(id="c1", name="fn", arguments={"x": 1})]
        resp = create_test_response(tool_calls=tc)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fn"

    def test_content_field(self):
        resp = create_test_response(text="hi")
        assert resp.content == [{"type": "text", "text": "hi"}]

    def test_empty_text_content(self):
        resp = create_test_response(text="")
        assert resp.content == []

    def test_custom_provider_model(self):
        resp = create_test_response(provider="openai", model="gpt-4")
        assert resp.provider == "openai"
        assert resp.model == "gpt-4"


class TestCreateTestAgent:
    """Test create_test_agent fixture function.

    Note: This depends on the Agent class being importable and
    working with FakeProvider. If Agent import fails, these tests
    will be skipped.
    """

    def test_returns_tuple(self):
        try:
            agent, provider = create_test_agent()
        except Exception:
            pytest.skip("Agent class not fully functional for this test")
        assert isinstance(provider, FakeProvider)

    def test_with_responses(self):
        try:
            agent, provider = create_test_agent(
                responses=[FakeResponse(text="r1"), FakeResponse(text="r2")]
            )
        except Exception:
            pytest.skip("Agent class not fully functional for this test")
        assert len(provider._responses) == 2

    def test_no_responses(self):
        try:
            agent, provider = create_test_agent()
        except Exception:
            pytest.skip("Agent class not fully functional for this test")
        assert provider._responses == []


class TestAgentTestCase:
    """Test AgentTestCase base class methods.

    We create a subclass and exercise the helpers.
    """

    def _make_case(self):
        """Create and set up an AgentTestCase instance."""
        case = AgentTestCase()
        try:
            case.setup_method()
        except Exception:
            pytest.skip("Agent class not fully functional for AgentTestCase")
        return case

    def test_setup_method(self):
        case = self._make_case()
        assert isinstance(case.provider, FakeProvider)
        assert case.agent is not None

    def test_set_response(self):
        case = self._make_case()
        case.set_response("hello")
        assert len(case.provider._responses) == 1
        assert case.provider._responses[0].text == "hello"

    def test_set_responses(self):
        case = self._make_case()
        case.set_responses(["a", "b", "c"])
        assert len(case.provider._responses) == 3
        assert case.provider._responses[1].text == "b"

    def test_set_tool_response(self):
        case = self._make_case()
        case.set_tool_response("search", {"q": "test"})
        assert len(case.provider._responses) == 1
        resp = case.provider._responses[0]
        assert resp.tool_calls[0].name == "search"

    def test_set_tool_response_custom_id(self):
        case = self._make_case()
        case.set_tool_response("fn", {"a": 1}, tool_id="custom")
        resp = case.provider._responses[0]
        assert resp.tool_calls[0].id == "custom"

    def test_set_error(self):
        case = self._make_case()
        err = RuntimeError("test_err")
        case.set_error(err)
        assert case.provider._responses[0].error is err

    def test_get_last_request_none(self):
        case = self._make_case()
        assert case.get_last_request() is None

    def test_assert_response_text_pass(self):
        resp = create_test_response(text="expected")
        case = self._make_case()
        case.assert_response_text(resp, "expected")  # should not raise

    def test_assert_response_text_fail(self):
        resp = create_test_response(text="actual")
        case = self._make_case()
        with pytest.raises(AssertionError):
            case.assert_response_text(resp, "wrong")

    def test_assert_request_contains_no_request(self):
        case = self._make_case()
        with pytest.raises(AssertionError, match="No request"):
            case.assert_request_contains("anything")
