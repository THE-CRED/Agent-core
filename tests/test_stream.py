"""
Tests for agent.stream — StreamResponse and AsyncStreamResponse.
"""

import pytest

from agent.stream import AsyncStreamResponse, StreamResponse
from agent.types.response import Usage
from agent.types.stream import StreamEvent
from agent.types.tools import ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(name: str = "my_tool", tc_id: str = "tc_1") -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments={"x": 1})


def _make_usage(prompt: int = 10, completion: int = 20) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _events_iter(events: list[StreamEvent]):
    """Return a plain iterator over *events*."""
    return iter(events)


async def _async_events_iter(events: list[StreamEvent]):
    """Return an async iterator over *events*."""
    for e in events:
        yield e


# ===================================================================
# StreamResponse (synchronous)
# ===================================================================


class TestStreamResponseTextAccumulation:
    """Text deltas are accumulated and joined."""

    def test_single_text_delta(self):
        events = [StreamEvent.text_delta("hello")]
        sr = StreamResponse(iter(events))
        collected = list(sr)
        assert len(collected) == 1
        assert sr.text == "hello"

    def test_multiple_text_deltas(self):
        events = [
            StreamEvent.text_delta("Hello"),
            StreamEvent.text_delta(", "),
            StreamEvent.text_delta("world!"),
        ]
        sr = StreamResponse(iter(events))
        list(sr)
        assert sr.text == "Hello, world!"

    def test_text_empty_before_iteration(self):
        events = [StreamEvent.text_delta("data")]
        sr = StreamResponse(iter(events))
        # Before consuming, text is empty
        assert sr.text == ""


class TestStreamResponseToolCalls:
    """tool_call_start events accumulate into tool_calls."""

    def test_single_tool_call(self):
        tc = _make_tool_call()
        events = [StreamEvent.tool_call_start(tc)]
        sr = StreamResponse(iter(events)).collect()
        assert sr.tool_calls == [tc]

    def test_multiple_tool_calls(self):
        tc1 = _make_tool_call("tool_a", "tc_1")
        tc2 = _make_tool_call("tool_b", "tc_2")
        events = [
            StreamEvent.tool_call_start(tc1),
            StreamEvent.tool_call_start(tc2),
        ]
        sr = StreamResponse(iter(events)).collect()
        assert len(sr.tool_calls) == 2
        assert sr.tool_calls[0].name == "tool_a"
        assert sr.tool_calls[1].name == "tool_b"

    def test_tool_calls_empty_when_none_present(self):
        events = [StreamEvent.text_delta("hi")]
        sr = StreamResponse(iter(events)).collect()
        assert sr.tool_calls == []


class TestStreamResponseUsage:
    """Usage is captured from both 'usage' and 'message_end' events."""

    def test_usage_from_usage_event(self):
        usage = _make_usage(5, 15)
        events = [StreamEvent.usage_event(usage)]
        sr = StreamResponse(iter(events)).collect()
        assert sr.usage is not None
        assert sr.usage.prompt_tokens == 5
        assert sr.usage.completion_tokens == 15
        assert sr.usage.total_tokens == 20

    def test_usage_from_message_end(self):
        usage = _make_usage(8, 12)
        events = [StreamEvent.message_end(usage=usage)]
        sr = StreamResponse(iter(events)).collect()
        assert sr.usage is not None
        assert sr.usage.prompt_tokens == 8
        assert sr.usage.completion_tokens == 12

    def test_message_end_usage_overrides_earlier_usage(self):
        """If both a usage event and message_end carry usage, the last one wins."""
        early = _make_usage(1, 2)
        final = _make_usage(100, 200)
        events = [
            StreamEvent.usage_event(early),
            StreamEvent.message_end(usage=final),
        ]
        sr = StreamResponse(iter(events)).collect()
        assert sr.usage is not None
        assert sr.usage.prompt_tokens == 100

    def test_message_end_without_usage_keeps_prior(self):
        """message_end with usage=None does not overwrite an earlier usage event."""
        early = _make_usage(3, 7)
        events = [
            StreamEvent.usage_event(early),
            StreamEvent.message_end(usage=None),
        ]
        sr = StreamResponse(iter(events)).collect()
        assert sr.usage is not None
        assert sr.usage.prompt_tokens == 3

    def test_usage_none_when_absent(self):
        events = [StreamEvent.text_delta("x")]
        sr = StreamResponse(iter(events)).collect()
        assert sr.usage is None


class TestStreamResponseCollect:
    """collect() consumes the iterator and returns self."""

    def test_collect_returns_self(self):
        events = [StreamEvent.text_delta("abc")]
        sr = StreamResponse(iter(events))
        result = sr.collect()
        assert result is sr

    def test_collect_accumulates_everything(self):
        tc = _make_tool_call()
        usage = _make_usage()
        events = [
            StreamEvent.message_start_event(),
            StreamEvent.text_delta("Hello"),
            StreamEvent.text_delta(" World"),
            StreamEvent.tool_call_start(tc),
            StreamEvent.usage_event(usage),
            StreamEvent.message_end(),
        ]
        sr = StreamResponse(iter(events)).collect()
        assert sr.text == "Hello World"
        assert sr.tool_calls == [tc]
        assert sr.usage is usage
        assert sr._done is True


class TestStreamResponseEmptyStream:
    """An empty event stream produces empty accumulated state."""

    def test_empty_stream(self):
        sr = StreamResponse(iter([])).collect()
        assert sr.text == ""
        assert sr.tool_calls == []
        assert sr.usage is None
        assert sr._done is False


class TestStreamResponseProviderModel:
    """provider and model are stored on the instance."""

    def test_defaults(self):
        sr = StreamResponse(iter([]))
        assert sr.provider == ""
        assert sr.model == ""

    def test_custom_values(self):
        sr = StreamResponse(iter([]), provider="anthropic", model="claude-3")
        assert sr.provider == "anthropic"
        assert sr.model == "claude-3"


class TestStreamResponseIteration:
    """__iter__ yields every event unchanged."""

    def test_yields_all_events(self):
        events = [
            StreamEvent.text_delta("a"),
            StreamEvent.message_start_event(),
            StreamEvent.message_end(),
        ]
        sr = StreamResponse(iter(events))
        yielded = list(sr)
        assert len(yielded) == 3
        assert yielded[0].type == "text_delta"
        assert yielded[1].type == "message_start"
        assert yielded[2].type == "message_end"

    def test_done_flag_set_on_message_end(self):
        events = [
            StreamEvent.text_delta("x"),
            StreamEvent.message_end(),
        ]
        sr = StreamResponse(iter(events))
        assert sr._done is False
        list(sr)
        assert sr._done is True


class TestStreamResponseMixedEvents:
    """Non-accumulating event types are yielded but do not affect state."""

    def test_error_event_yielded_but_ignored(self):
        events = [
            StreamEvent.text_delta("ok"),
            StreamEvent.error_event("something broke"),
        ]
        sr = StreamResponse(iter(events)).collect()
        assert sr.text == "ok"
        assert sr.tool_calls == []
        assert sr.usage is None

    def test_tool_call_delta_not_accumulated(self):
        events = [
            StreamEvent.tool_call_delta_event("tc_1", {"args": "partial"}),
        ]
        sr = StreamResponse(iter(events)).collect()
        assert sr.tool_calls == []

    def test_tool_result_not_accumulated(self):
        events = [
            StreamEvent.tool_result_event("tc_1", "result_text"),
        ]
        sr = StreamResponse(iter(events)).collect()
        assert sr.tool_calls == []


# ===================================================================
# AsyncStreamResponse
# ===================================================================


class TestAsyncStreamResponseTextAccumulation:
    @pytest.mark.asyncio
    async def test_single_text_delta(self):
        events = [StreamEvent.text_delta("hello")]
        asr = AsyncStreamResponse(_async_events_iter(events))
        collected = [e async for e in asr]
        assert len(collected) == 1
        assert asr.text == "hello"

    @pytest.mark.asyncio
    async def test_multiple_text_deltas(self):
        events = [
            StreamEvent.text_delta("Hello"),
            StreamEvent.text_delta(", "),
            StreamEvent.text_delta("world!"),
        ]
        asr = AsyncStreamResponse(_async_events_iter(events))
        async for _ in asr:
            pass
        assert asr.text == "Hello, world!"


class TestAsyncStreamResponseToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_accumulation(self):
        tc = _make_tool_call()
        events = [StreamEvent.tool_call_start(tc)]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.tool_calls == [tc]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        tc1 = _make_tool_call("a", "1")
        tc2 = _make_tool_call("b", "2")
        events = [
            StreamEvent.tool_call_start(tc1),
            StreamEvent.tool_call_start(tc2),
        ]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert len(asr.tool_calls) == 2


class TestAsyncStreamResponseUsage:
    @pytest.mark.asyncio
    async def test_usage_from_usage_event(self):
        usage = _make_usage(10, 20)
        events = [StreamEvent.usage_event(usage)]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.usage is not None
        assert asr.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_usage_from_message_end(self):
        usage = _make_usage(5, 5)
        events = [StreamEvent.message_end(usage=usage)]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.usage is not None
        assert asr.usage.prompt_tokens == 5

    @pytest.mark.asyncio
    async def test_message_end_usage_overrides(self):
        early = _make_usage(1, 1)
        final = _make_usage(50, 50)
        events = [
            StreamEvent.usage_event(early),
            StreamEvent.message_end(usage=final),
        ]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.usage is not None
        assert asr.usage.prompt_tokens == 50

    @pytest.mark.asyncio
    async def test_message_end_without_usage_keeps_prior(self):
        early = _make_usage(7, 3)
        events = [
            StreamEvent.usage_event(early),
            StreamEvent.message_end(usage=None),
        ]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.usage is not None
        assert asr.usage.prompt_tokens == 7

    @pytest.mark.asyncio
    async def test_usage_none_when_absent(self):
        events = [StreamEvent.text_delta("x")]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.usage is None


class TestAsyncStreamResponseCollect:
    @pytest.mark.asyncio
    async def test_collect_returns_self(self):
        events = [StreamEvent.text_delta("z")]
        asr = AsyncStreamResponse(_async_events_iter(events))
        result = await asr.collect()
        assert result is asr

    @pytest.mark.asyncio
    async def test_collect_accumulates_everything(self):
        tc = _make_tool_call()
        usage = _make_usage()
        events = [
            StreamEvent.message_start_event(),
            StreamEvent.text_delta("Hi"),
            StreamEvent.tool_call_start(tc),
            StreamEvent.usage_event(usage),
            StreamEvent.message_end(),
        ]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.text == "Hi"
        assert asr.tool_calls == [tc]
        assert asr.usage is usage
        assert asr._done is True


class TestAsyncStreamResponseEmptyStream:
    @pytest.mark.asyncio
    async def test_empty_stream(self):
        asr = await AsyncStreamResponse(_async_events_iter([])).collect()
        assert asr.text == ""
        assert asr.tool_calls == []
        assert asr.usage is None
        assert asr._done is False


class TestAsyncStreamResponseProviderModel:
    @pytest.mark.asyncio
    async def test_defaults(self):
        asr = AsyncStreamResponse(_async_events_iter([]))
        assert asr.provider == ""
        assert asr.model == ""

    @pytest.mark.asyncio
    async def test_custom_values(self):
        asr = AsyncStreamResponse(
            _async_events_iter([]), provider="openai", model="gpt-4"
        )
        assert asr.provider == "openai"
        assert asr.model == "gpt-4"


class TestAsyncStreamResponseIteration:
    @pytest.mark.asyncio
    async def test_yields_all_events(self):
        events = [
            StreamEvent.text_delta("a"),
            StreamEvent.message_start_event(),
            StreamEvent.message_end(),
        ]
        asr = AsyncStreamResponse(_async_events_iter(events))
        yielded = [e async for e in asr]
        assert len(yielded) == 3
        assert yielded[0].type == "text_delta"
        assert yielded[2].type == "message_end"

    @pytest.mark.asyncio
    async def test_done_flag_set_on_message_end(self):
        events = [StreamEvent.message_end()]
        asr = AsyncStreamResponse(_async_events_iter(events))
        assert asr._done is False
        async for _ in asr:
            pass
        assert asr._done is True


class TestAsyncStreamResponseMixedEvents:
    @pytest.mark.asyncio
    async def test_error_event_yielded_but_ignored(self):
        events = [
            StreamEvent.text_delta("ok"),
            StreamEvent.error_event("err"),
        ]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.text == "ok"
        assert asr.tool_calls == []

    @pytest.mark.asyncio
    async def test_tool_call_delta_not_accumulated(self):
        events = [
            StreamEvent.tool_call_delta_event("tc_1", {"args": "partial"}),
        ]
        asr = await AsyncStreamResponse(_async_events_iter(events)).collect()
        assert asr.tool_calls == []
