"""
Streaming types for Agent.
"""

from typing import Any, Literal

from pydantic import BaseModel

from agent.types.response import Usage
from agent.types.tools import ToolCall

StreamEventType = Literal[
    "text_delta",
    "tool_call_start",
    "tool_call_delta",
    "tool_result",
    "message_start",
    "message_end",
    "usage",
    "error",
]


class StreamEvent(BaseModel):
    """A normalized streaming event."""

    type: StreamEventType
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_call_delta: dict[str, Any] | None = None
    tool_result: str | None = None
    usage: Usage | None = None
    error: str | None = None
    raw: Any = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def text_delta(cls, text: str, raw: Any = None) -> "StreamEvent":
        """Create a text delta event."""
        return cls(type="text_delta", text=text, raw=raw)

    @classmethod
    def tool_call_start(cls, tool_call: ToolCall, raw: Any = None) -> "StreamEvent":
        """Create a tool call start event."""
        return cls(type="tool_call_start", tool_call=tool_call, raw=raw)

    @classmethod
    def tool_call_delta_event(
        cls, tool_call_id: str, delta: dict[str, Any], raw: Any = None
    ) -> "StreamEvent":
        """Create a tool call delta event."""
        return cls(
            type="tool_call_delta",
            tool_call_delta={"id": tool_call_id, **delta},
            raw=raw,
        )

    @classmethod
    def tool_result_event(
        cls, tool_call_id: str, result: str, raw: Any = None
    ) -> "StreamEvent":
        """Create a tool result event."""
        return cls(type="tool_result", tool_result=result, raw=raw)

    @classmethod
    def message_start_event(cls, raw: Any = None) -> "StreamEvent":
        """Create a message start event."""
        return cls(type="message_start", raw=raw)

    @classmethod
    def message_end(cls, usage: Usage | None = None, raw: Any = None) -> "StreamEvent":
        """Create a message end event."""
        return cls(type="message_end", usage=usage, raw=raw)

    @classmethod
    def usage_event(cls, usage: Usage, raw: Any = None) -> "StreamEvent":
        """Create a usage event."""
        return cls(type="usage", usage=usage, raw=raw)

    @classmethod
    def error_event(cls, error: str, raw: Any = None) -> "StreamEvent":
        """Create an error event."""
        return cls(type="error", error=error, raw=raw)
