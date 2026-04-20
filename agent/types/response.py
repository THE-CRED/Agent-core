"""
Response types for Agent.
"""

from typing import Any

from pydantic import BaseModel, Field

from agent.types.tools import ToolCall


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Usage":
        """Create Usage from a dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


class AgentResponse(BaseModel):
    """Normalized response from any provider."""

    text: str | None = None
    content: list[Any] = Field(default_factory=list)
    output: Any = None  # Parsed structured output
    provider: str = ""
    model: str = ""
    usage: Usage | None = None
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw: Any = None
    latency_ms: float | None = None
    cost_estimate: float | None = None
    request_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert response to a dictionary."""
        return {
            "text": self.text,
            "content": self.content,
            "output": self.output,
            "provider": self.provider,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens if self.usage else 0,
                "completion_tokens": self.usage.completion_tokens if self.usage else 0,
                "total_tokens": self.usage.total_tokens if self.usage else 0,
            },
            "stop_reason": self.stop_reason,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "latency_ms": self.latency_ms,
            "cost_estimate": self.cost_estimate,
            "request_id": self.request_id,
        }
