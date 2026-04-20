"""
Tool-related types for Agent.
"""

from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """Specification for a tool that can be called by an LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Any | None = None  # Callable[..., Any]
    is_async: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_gemini_schema(self) -> dict[str, Any]:
        """Convert to Gemini function declaration format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


class ToolResult(BaseModel):
    """Result of executing a tool."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
        }
