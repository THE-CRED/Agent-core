"""
Message types for Agent.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    """A part of message content (text, image, etc.)."""

    type: Literal["text", "image", "image_url"]
    text: str | None = None
    image_url: str | None = None
    image_data: bytes | None = None
    media_type: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def text_part(cls, text: str) -> "ContentPart":
        """Create a text content part."""
        return cls(type="text", text=text)

    @classmethod
    def image_url_part(cls, url: str) -> "ContentPart":
        """Create an image URL content part."""
        return cls(type="image_url", image_url=url)

    @classmethod
    def image_data_part(cls, data: bytes, media_type: str = "image/png") -> "ContentPart":
        """Create an image data content part."""
        return cls(type="image", image_data=data, media_type=media_type)


class Message(BaseModel):
    """A normalized message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str | list[ContentPart]) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content or "", tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> "Message":
        """Create a tool result message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    @property
    def text(self) -> str:
        """Get the text content of the message."""
        if isinstance(self.content, str):
            return self.content
        text_parts = [p.text for p in self.content if p.type == "text" and p.text]
        return "".join(text_parts)


class AgentRequest(BaseModel):
    """A normalized request to be sent to a provider."""

    input: str | None = None
    messages: list[Message] = Field(default_factory=list)
    system: str | None = None
    tools: list[Any] = Field(default_factory=list)  # list[ToolSpec]
    output_schema: dict[str, Any] | None = Field(default=None)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None

    @property
    def schema(self) -> dict[str, Any] | None:
        """Alias for output_schema for backwards compatibility."""
        return self.output_schema

    @schema.setter
    def schema(self, value: dict[str, Any] | None) -> None:
        """Alias for output_schema for backwards compatibility."""
        self.output_schema = value

    def to_messages(self) -> list[Message]:
        """Convert request to a list of messages."""
        messages: list[Message] = []

        # Add system message if present
        if self.system:
            messages.append(Message.system(self.system))

        # Add existing messages
        messages.extend(self.messages)

        # Add input as user message if present
        if self.input:
            messages.append(Message.user(self.input))

        return messages
