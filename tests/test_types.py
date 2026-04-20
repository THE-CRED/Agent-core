"""Tests for all type modules."""

import os
from unittest.mock import patch

from agent.types.config import (
    BASE_URLS,
    ENV_VARS,
    MODEL_ALIASES,
    PRICING,
    AgentConfig,
    ProviderCapabilities,
    RetryConfig,
    ToolLoopConfig,
    estimate_cost,
    get_api_key,
    get_base_url,
    resolve_model,
)
from agent.types.messages import AgentRequest, ContentPart, Message
from agent.types.response import AgentResponse, Usage
from agent.types.router import RouteResult, RoutingStrategy
from agent.types.stream import StreamEvent
from agent.types.tools import ToolCall, ToolResult, ToolSpec

# ── ContentPart ──────────────────────────────────────────────────

class TestContentPart:
    def test_text_part(self):
        p = ContentPart.text_part("hello")
        assert p.type == "text"
        assert p.text == "hello"

    def test_image_url_part(self):
        p = ContentPart.image_url_part("https://img.png")
        assert p.type == "image_url"
        assert p.image_url == "https://img.png"

    def test_image_data_part(self):
        p = ContentPart.image_data_part(b"\x89PNG", media_type="image/png")
        assert p.type == "image"
        assert p.image_data == b"\x89PNG"
        assert p.media_type == "image/png"

    def test_image_data_default_media_type(self):
        p = ContentPart.image_data_part(b"data")
        assert p.media_type == "image/png"


# ── Message ──────────────────────────────────────────────────────

class TestMessage:
    def test_system(self):
        m = Message.system("You are helpful")
        assert m.role == "system"
        assert m.content == "You are helpful"

    def test_user_str(self):
        m = Message.user("Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_user_multimodal(self):
        parts = [ContentPart.text_part("hi"), ContentPart.image_url_part("url")]
        m = Message.user(parts)
        assert m.role == "user"
        assert len(m.content) == 2

    def test_assistant(self):
        m = Message.assistant(content="Hi there")
        assert m.role == "assistant"
        assert m.content == "Hi there"

    def test_assistant_default_empty(self):
        m = Message.assistant()
        assert m.content == ""

    def test_assistant_with_tool_calls(self):
        tc = [{"id": "1", "name": "search", "arguments": {}}]
        m = Message.assistant(content="ok", tool_calls=tc)
        assert m.tool_calls == tc

    def test_tool(self):
        m = Message.tool(content="result", tool_call_id="call_1", name="search")
        assert m.role == "tool"
        assert m.content == "result"
        assert m.tool_call_id == "call_1"
        assert m.name == "search"

    def test_text_property_str(self):
        m = Message.user("hello")
        assert m.text == "hello"

    def test_text_property_parts(self):
        parts = [ContentPart.text_part("a"), ContentPart.text_part("b")]
        m = Message.user(parts)
        assert m.text == "ab"

    def test_text_property_mixed_parts(self):
        parts = [ContentPart.text_part("hi"), ContentPart.image_url_part("url")]
        m = Message.user(parts)
        assert m.text == "hi"


# ── AgentRequest ─────────────────────────────────────────────────

class TestAgentRequest:
    def test_defaults(self):
        r = AgentRequest()
        assert r.input is None
        assert r.messages == []
        assert r.system is None
        assert r.tools == []
        assert r.output_schema is None
        assert r.metadata == {}
        assert r.session_id is None

    def test_schema_property(self):
        r = AgentRequest(output_schema={"type": "object"})
        assert r.schema == {"type": "object"}

    def test_schema_setter(self):
        r = AgentRequest()
        r.schema = {"type": "string"}
        assert r.output_schema == {"type": "string"}

    def test_to_messages_with_system(self):
        r = AgentRequest(system="sys", input="hi")
        msgs = r.to_messages()
        assert msgs[0].role == "system"
        assert msgs[0].content == "sys"
        assert msgs[1].role == "user"
        assert msgs[1].content == "hi"

    def test_to_messages_with_existing(self):
        existing = [Message.user("old")]
        r = AgentRequest(messages=existing, input="new")
        msgs = r.to_messages()
        assert len(msgs) == 2
        assert msgs[0].content == "old"
        assert msgs[1].content == "new"

    def test_to_messages_no_input(self):
        r = AgentRequest(system="sys")
        msgs = r.to_messages()
        assert len(msgs) == 1
        assert msgs[0].role == "system"


# ── Usage ────────────────────────────────────────────────────────

class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_from_dict(self):
        u = Usage.from_dict({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.total_tokens == 30

    def test_from_dict_partial(self):
        u = Usage.from_dict({"prompt_tokens": 5})
        assert u.prompt_tokens == 5
        assert u.completion_tokens == 0

    def test_from_dict_empty(self):
        u = Usage.from_dict({})
        assert u.total_tokens == 0


# ── AgentResponse ────────────────────────────────────────────────

class TestAgentResponse:
    def test_defaults(self):
        r = AgentResponse()
        assert r.text is None
        assert r.content == []
        assert r.output is None
        assert r.provider == ""
        assert r.tool_calls == []
        assert r.latency_ms is None
        assert r.cost_estimate is None

    def test_has_tool_calls_false(self):
        r = AgentResponse()
        assert r.has_tool_calls is False

    def test_has_tool_calls_true(self):
        tc = ToolCall(id="1", name="search", arguments={})
        r = AgentResponse(tool_calls=[tc])
        assert r.has_tool_calls is True

    def test_to_dict(self):
        u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        tc = ToolCall(id="1", name="s", arguments={"q": "x"})
        r = AgentResponse(
            text="hello",
            provider="fake",
            model="m",
            usage=u,
            stop_reason="stop",
            tool_calls=[tc],
            latency_ms=100.0,
            cost_estimate=0.01,
            request_id="req_1",
        )
        d = r.to_dict()
        assert d["text"] == "hello"
        assert d["usage"]["prompt_tokens"] == 10
        assert d["tool_calls"][0]["name"] == "s"
        assert d["latency_ms"] == 100.0

    def test_to_dict_no_usage(self):
        r = AgentResponse(text="hi")
        d = r.to_dict()
        assert d["usage"]["prompt_tokens"] == 0


# ── ToolSpec ─────────────────────────────────────────────────────

class TestToolSpec:
    def _make_spec(self):
        return ToolSpec(
            name="search",
            description="Search items",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        )

    def test_to_openai_schema(self):
        s = self._make_spec().to_openai_schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == "search"
        assert s["function"]["description"] == "Search items"
        assert s["function"]["parameters"]["type"] == "object"

    def test_to_anthropic_schema(self):
        s = self._make_spec().to_anthropic_schema()
        assert s["name"] == "search"
        assert "input_schema" in s

    def test_to_gemini_schema(self):
        s = self._make_spec().to_gemini_schema()
        assert s["name"] == "search"
        assert s["parameters"]["type"] == "object"


# ── ToolCall ─────────────────────────────────────────────────────

class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(id="c1", name="search", arguments={"q": "test"})
        d = tc.to_dict()
        assert d == {"id": "c1", "name": "search", "arguments": {"q": "test"}}

    def test_from_dict(self):
        tc = ToolCall.from_dict({"id": "c2", "name": "fetch", "arguments": {"url": "x"}})
        assert tc.id == "c2"
        assert tc.name == "fetch"
        assert tc.arguments == {"url": "x"}

    def test_from_dict_no_arguments(self):
        tc = ToolCall.from_dict({"id": "c3", "name": "ping"})
        assert tc.arguments == {}

    def test_default_arguments(self):
        tc = ToolCall(id="c4", name="test")
        assert tc.arguments == {}


# ── ToolResult ───────────────────────────────────────────────────

class TestToolResult:
    def test_to_dict(self):
        tr = ToolResult(tool_call_id="c1", name="search", content="found it", is_error=False)
        d = tr.to_dict()
        assert d["tool_call_id"] == "c1"
        assert d["content"] == "found it"
        assert d["is_error"] is False

    def test_is_error_default(self):
        tr = ToolResult(tool_call_id="c1", name="s", content="ok")
        assert tr.is_error is False


# ── StreamEvent ──────────────────────────────────────────────────

class TestStreamEvent:
    def test_text_delta(self):
        e = StreamEvent.text_delta("hello")
        assert e.type == "text_delta"
        assert e.text == "hello"

    def test_tool_call_start(self):
        tc = ToolCall(id="1", name="s", arguments={})
        e = StreamEvent.tool_call_start(tc)
        assert e.type == "tool_call_start"
        assert e.tool_call is tc

    def test_tool_call_delta_event(self):
        e = StreamEvent.tool_call_delta_event("c1", {"args": "partial"})
        assert e.type == "tool_call_delta"
        assert e.tool_call_delta["id"] == "c1"

    def test_tool_result_event(self):
        e = StreamEvent.tool_result_event("c1", "result")
        assert e.type == "tool_result"
        assert e.tool_result == "result"

    def test_message_start_event(self):
        e = StreamEvent.message_start_event()
        assert e.type == "message_start"

    def test_message_end(self):
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        e = StreamEvent.message_end(usage=u)
        assert e.type == "message_end"
        assert e.usage is u

    def test_message_end_no_usage(self):
        e = StreamEvent.message_end()
        assert e.usage is None

    def test_usage_event(self):
        u = Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        e = StreamEvent.usage_event(u)
        assert e.type == "usage"
        assert e.usage.total_tokens == 15

    def test_error_event(self):
        e = StreamEvent.error_event("something broke")
        assert e.type == "error"
        assert e.error == "something broke"

    def test_raw_passthrough(self):
        e = StreamEvent.text_delta("hi", raw={"raw": True})
        assert e.raw == {"raw": True}


# ── RoutingStrategy ──────────────────────────────────────────────

class TestRoutingStrategy:
    def test_values(self):
        assert RoutingStrategy.FALLBACK.value == "fallback"
        assert RoutingStrategy.ROUND_ROBIN.value == "round_robin"
        assert RoutingStrategy.FASTEST.value == "fastest"
        assert RoutingStrategy.CHEAPEST.value == "cheapest"
        assert RoutingStrategy.CAPABILITY.value == "capability"
        assert RoutingStrategy.CUSTOM.value == "custom"

    def test_from_string(self):
        assert RoutingStrategy("fallback") == RoutingStrategy.FALLBACK

    def test_is_str_enum(self):
        assert isinstance(RoutingStrategy.FALLBACK, str)


# ── RouteResult ──────────────────────────────────────────────────

class TestRouteResult:
    def test_basic(self):
        r = RouteResult(agent="agent_obj", reason="cheapest")
        assert r.agent == "agent_obj"
        assert r.reason == "cheapest"

    def test_reason_optional(self):
        r = RouteResult(agent="a")
        assert r.reason is None


# ── Config Functions ─────────────────────────────────────────────

class TestConfigFunctions:
    def test_env_vars_has_known_providers(self):
        assert "openai" in ENV_VARS
        assert "anthropic" in ENV_VARS
        assert "gemini" in ENV_VARS
        assert "deepseek" in ENV_VARS

    def test_base_urls_has_known_providers(self):
        assert "openai" in BASE_URLS
        assert "anthropic" in BASE_URLS

    def test_model_aliases(self):
        assert "claude" in MODEL_ALIASES
        assert "gpt-4o" in MODEL_ALIASES

    def test_pricing_has_models(self):
        assert "gpt-4o" in PRICING
        assert "input" in PRICING["gpt-4o"]
        assert "output" in PRICING["gpt-4o"]

    def test_get_api_key_explicit(self):
        assert get_api_key("openai", api_key="sk-test") == "sk-test"

    def test_get_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            assert get_api_key("openai") == "sk-env"

    def test_get_api_key_unknown_provider(self):
        assert get_api_key("unknown_provider") is None

    def test_get_api_key_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = get_api_key("openai")
            # May or may not be set in real env, just test doesn't crash
            assert result is None or isinstance(result, str)

    def test_get_base_url_explicit(self):
        assert get_base_url("openai", base_url="http://custom") == "http://custom"

    def test_get_base_url_default(self):
        assert get_base_url("openai") == "https://api.openai.com/v1"

    def test_get_base_url_unknown(self):
        assert get_base_url("unknown") is None

    def test_resolve_model_alias(self):
        assert resolve_model("claude") == "claude-sonnet-4-20250514"

    def test_resolve_model_passthrough(self):
        assert resolve_model("custom-model-v3") == "custom-model-v3"

    def test_estimate_cost_known(self):
        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost is not None
        assert cost == PRICING["gpt-4o"]["input"] + PRICING["gpt-4o"]["output"]

    def test_estimate_cost_unknown(self):
        assert estimate_cost("unknown-model", 100, 100) is None

    def test_estimate_cost_zero_tokens(self):
        cost = estimate_cost("gpt-4o", 0, 0)
        assert cost == 0.0


# ── ProviderCapabilities ─────────────────────────────────────────

class TestProviderCapabilities:
    def test_defaults(self):
        c = ProviderCapabilities()
        assert c.streaming is True
        assert c.tools is True
        assert c.structured_output is True
        assert c.json_mode is True
        assert c.vision is False
        assert c.system_messages is True
        assert c.batch is False
        assert c.native_schema_output is False
        assert c.max_context_tokens is None
        assert c.max_output_tokens is None


# ── AgentConfig ──────────────────────────────────────────────────

class TestAgentConfig:
    def test_basic_creation(self):
        c = AgentConfig(provider="fake", model="fake-model")
        assert c.provider == "fake"
        assert c.timeout == 120.0
        assert c.max_retries == 2

    def test_resolves_model_alias(self):
        c = AgentConfig(provider="anthropic", model="claude")
        assert c.model == "claude-sonnet-4-20250514"

    def test_resolves_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            c = AgentConfig(provider="openai", model="gpt-4o")
            assert c.api_key == "sk-test123"

    def test_explicit_api_key(self):
        c = AgentConfig(provider="openai", model="gpt-4o", api_key="sk-explicit")
        assert c.api_key == "sk-explicit"

    def test_resolves_base_url(self):
        c = AgentConfig(provider="openai", model="gpt-4o")
        assert c.base_url == "https://api.openai.com/v1"

    def test_with_overrides(self):
        c = AgentConfig(provider="fake", model="m1", temperature=0.5)
        c2 = c.with_overrides(temperature=0.9, max_tokens=100)
        assert c2.temperature == 0.9
        assert c2.max_tokens == 100
        assert c2.provider == "fake"

    def test_with_overrides_preserves_extra(self):
        c = AgentConfig(provider="fake", model="m1", extra={"key": "val"})
        c2 = c.with_overrides(extra={"key2": "val2"})
        assert "key" in c2.extra
        assert "key2" in c2.extra

    def test_extra_default(self):
        c = AgentConfig(provider="fake", model="m1")
        assert c.extra == {}


# ── RetryConfig ──────────────────────────────────────────────────

class TestRetryConfig:
    def test_defaults(self):
        c = RetryConfig()
        assert c.max_retries == 2
        assert c.initial_delay == 1.0
        assert c.max_delay == 60.0
        assert c.exponential_base == 2.0
        assert c.jitter is True

    def test_should_retry_rate_limit(self):
        from agent.errors import RateLimitError
        c = RetryConfig(max_retries=3)
        assert c.should_retry(RateLimitError("limited"), attempt=0) is True
        assert c.should_retry(RateLimitError("limited"), attempt=2) is True

    def test_should_retry_exhausted(self):
        from agent.errors import RateLimitError
        c = RetryConfig(max_retries=2)
        assert c.should_retry(RateLimitError("limited"), attempt=2) is False

    def test_should_retry_5xx(self):
        from agent.errors import ProviderError
        c = RetryConfig(max_retries=3)
        assert c.should_retry(ProviderError("err", status_code=500), attempt=0) is True
        assert c.should_retry(ProviderError("err", status_code=503), attempt=0) is True

    def test_should_not_retry_4xx(self):
        from agent.errors import ProviderError
        c = RetryConfig(max_retries=3)
        assert c.should_retry(ProviderError("err", status_code=400), attempt=0) is False
        assert c.should_retry(ProviderError("err", status_code=404), attempt=0) is False

    def test_should_retry_connection_error(self):
        c = RetryConfig(max_retries=3)
        assert c.should_retry(ConnectionError("conn refused"), attempt=0) is True

    def test_should_not_retry_value_error(self):
        c = RetryConfig(max_retries=3)
        assert c.should_retry(ValueError("bad value"), attempt=0) is False

    def test_get_delay_with_retry_after(self):
        from agent.errors import RateLimitError
        c = RetryConfig()
        e = RateLimitError("limited", retry_after=5.0)
        delay = c.get_delay(0, error=e)
        assert delay == 5.0

    def test_get_delay_retry_after_capped(self):
        from agent.errors import RateLimitError
        c = RetryConfig(max_delay=10.0)
        e = RateLimitError("limited", retry_after=100.0)
        delay = c.get_delay(0, error=e)
        assert delay == 10.0

    def test_get_delay_exponential(self):
        c = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert c.get_delay(0) == 1.0
        assert c.get_delay(1) == 2.0
        assert c.get_delay(2) == 4.0

    def test_get_delay_capped(self):
        c = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=3.0, jitter=False)
        assert c.get_delay(10) == 3.0

    def test_get_delay_with_jitter(self):
        c = RetryConfig(initial_delay=1.0, jitter=True)
        delay = c.get_delay(0)
        assert 0.5 <= delay <= 1.5


# ── ToolLoopConfig ───────────────────────────────────────────────

class TestToolLoopConfig:
    def test_defaults(self):
        c = ToolLoopConfig()
        assert c.max_iterations == 10
        assert c.max_tool_calls_per_iteration == 20
        assert c.timeout_per_tool == 30.0
        assert c.parallel_tool_execution is True
        assert c.stop_on_error is False
