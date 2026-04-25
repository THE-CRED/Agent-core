"""
Agent streaming types and utilities.

Normalized streaming event interface across providers.
"""

from collections.abc import Iterator
from typing import Any

from agent.types.response import Usage
from agent.types.stream import StreamEvent, StreamEventType
from agent.types.tools import ToolCall


class StreamResponse:
    """
    A streaming response that yields events and accumulates the final response.
    """

    def __init__(
        self,
        _events: Iterator[StreamEvent],
        provider: str = "",
        model: str = "",
    ):
        self._events = _events
        self.provider = provider
        self.model = model
        self._text_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._usage: Usage | None = None
        self._done: bool = False

    def __iter__(self) -> Iterator[StreamEvent]:
        """Iterate over stream events. Can only be iterated once."""
        if self._done:
            return
        for event in self._events:
            # Accumulate text
            if event.type == "text_delta" and event.text:
                self._text_parts.append(event.text)

            # Accumulate tool calls
            if event.type == "tool_call_start" and event.tool_call:
                self._tool_calls.append(event.tool_call)

            # Capture usage
            if event.type == "usage" and event.usage:
                self._usage = event.usage

            if event.type == "message_end":
                self._done = True
                if event.usage:
                    self._usage = event.usage

            yield event

    @property
    def text(self) -> str:
        """Get accumulated text (available after iteration)."""
        return "".join(self._text_parts)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Get accumulated tool calls (available after iteration)."""
        return self._tool_calls

    @property
    def usage(self) -> Usage | None:
        """Get usage information (available after iteration)."""
        return self._usage

    def collect(self) -> "StreamResponse":
        """Consume all events and return self with accumulated state."""
        for _ in self:
            pass
        return self


class AsyncStreamResponse:
    """
    An async streaming response that yields events and accumulates the final response.
    """

    def __init__(
        self,
        events: Any,  # AsyncIterator[StreamEvent]
        provider: str = "",
        model: str = "",
    ):
        self._events = events
        self.provider = provider
        self.model = model
        self._text_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._usage: Usage | None = None
        self._done: bool = False

    def __aiter__(self):
        """Return async iterator."""
        return self

    async def __anext__(self) -> StreamEvent:
        """Get next event."""
        try:
            event = await self._events.__anext__()

            # Accumulate text
            if event.type == "text_delta" and event.text:
                self._text_parts.append(event.text)

            # Accumulate tool calls
            if event.type == "tool_call_start" and event.tool_call:
                self._tool_calls.append(event.tool_call)

            # Capture usage
            if event.type == "usage" and event.usage:
                self._usage = event.usage

            if event.type == "message_end":
                self._done = True
                if event.usage:
                    self._usage = event.usage

            return event
        except StopAsyncIteration:
            raise

    @property
    def text(self) -> str:
        """Get accumulated text (available after iteration)."""
        return "".join(self._text_parts)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Get accumulated tool calls (available after iteration)."""
        return self._tool_calls

    @property
    def usage(self) -> Usage | None:
        """Get usage information (available after iteration)."""
        return self._usage

    async def collect(self) -> "AsyncStreamResponse":
        """Consume all events and return self with accumulated state."""
        async for _ in self:
            pass
        return self


# Re-export types for backwards compatibility
__all__ = ["StreamEvent", "StreamEventType", "StreamResponse", "AsyncStreamResponse"]
