"""
Anthropic provider adapter.

Supports Claude models via the Anthropic API.
"""

import base64
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agent.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from agent.errors import (
    TimeoutError as AgentTimeoutError,
)
from agent.messages import AgentRequest, Message
from agent.providers.base import BaseProvider
from agent.providers.registry import ProviderRegistry
from agent.response import AgentResponse
from agent.stream import StreamEvent
from agent.types.config import ProviderCapabilities
from agent.types.response import Usage
from agent.types.tools import ToolCall

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic: Any = None
    AsyncAnthropic: Any = None


class AnthropicProvider(BaseProvider):
    """
    Anthropic provider adapter.

    Supports Claude 3 Opus, Sonnet, Haiku and other Claude models.
    """

    name = "anthropic"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=False,  # Anthropic uses tool_use for structured output
        vision=True,
        system_messages=True,
        batch=True,
        native_schema_output=False,  # Uses prompt-based approach
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install agent-core-py[anthropic]"
            )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": 0,  # We handle retries ourselves
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        assert Anthropic is not None
        assert AsyncAnthropic is not None
        self._client = Anthropic(**client_kwargs)
        self._async_client = AsyncAnthropic(**client_kwargs)

    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a synchronous request."""
        try:
            messages = self._convert_messages(request)
            kwargs = self._build_kwargs(request)

            response = self._client.messages.create(  # type: ignore[no-matching-overload]
                messages=messages,
                **kwargs,
            )

            return self._convert_response(response)

        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except anthropic.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except anthropic.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except anthropic.APIError as e:
            raise ProviderError(
                str(e),
                provider=self.name,
                status_code=getattr(e, "status_code", None),
                raw=e,
            ) from e

    async def run_async(self, request: AgentRequest) -> AgentResponse:
        """Execute an asynchronous request."""
        try:
            messages = self._convert_messages(request)
            kwargs = self._build_kwargs(request)

            response = await self._async_client.messages.create(  # type: ignore[no-matching-overload]
                messages=messages,
                **kwargs,
            )

            return self._convert_response(response)

        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except anthropic.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except anthropic.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except anthropic.APIError as e:
            raise ProviderError(
                str(e),
                provider=self.name,
                status_code=getattr(e, "status_code", None),
                raw=e,
            ) from e

    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        """Execute a streaming request."""
        try:
            messages = self._convert_messages(request)
            kwargs = self._build_kwargs(request)

            with self._client.messages.stream(  # type: ignore[arg-type]
                messages=messages,
                **kwargs,
            ) as stream:
                yield from self._convert_stream(stream)

        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except anthropic.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except anthropic.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except anthropic.APIError as e:
            raise ProviderError(
                str(e),
                provider=self.name,
                status_code=getattr(e, "status_code", None),
                raw=e,
            ) from e

    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        """Execute an async streaming request."""
        try:
            messages = self._convert_messages(request)
            kwargs = self._build_kwargs(request)

            async with self._async_client.messages.stream(  # type: ignore[arg-type]
                messages=messages,
                **kwargs,
            ) as stream:
                async for event in stream:
                    for ev in self._convert_event(event):
                        yield ev

        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except anthropic.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except anthropic.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except anthropic.APIError as e:
            raise ProviderError(
                str(e),
                provider=self.name,
                status_code=getattr(e, "status_code", None),
                raw=e,
            ) from e

    def _convert_messages(self, request: AgentRequest) -> list[dict[str, Any]]:
        """Convert normalized messages to Anthropic format."""
        messages: list[dict[str, Any]] = []

        # Anthropic handles system separately, so skip system messages here
        for msg in request.messages:
            if msg.role == "system":
                continue
            anthropic_msg = self._convert_message(msg)
            messages.append(anthropic_msg)

        # Add input as user message
        if request.input:
            messages.append({"role": "user", "content": request.input})

        return messages

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """Convert a single message to Anthropic format."""
        result: dict[str, Any] = {"role": msg.role if msg.role != "tool" else "user"}

        # Handle content
        if isinstance(msg.content, str):
            if msg.role == "tool":
                # Tool results use tool_result block
                result["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                ]
            else:
                result["content"] = msg.content
        else:
            # Multi-part content
            parts = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    parts.append({"type": "text", "text": part.text})
                elif part.type == "image_url" and part.image_url:
                    # Anthropic needs base64 for images
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part.image_url,
                            },
                        }
                    )
                elif part.type == "image" and part.image_data:
                    b64_data = base64.b64encode(part.image_data).decode()
                    media_type = part.media_type or "image/png"
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        }
                    )
            result["content"] = parts

        # Handle assistant messages with tool calls
        if msg.role == "assistant" and msg.tool_calls:
            content = []
            if msg.content and isinstance(msg.content, str):
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc.get("arguments", {}),
                    }
                )
            result["content"] = content

        return result

    def _build_kwargs(self, request: AgentRequest) -> dict[str, Any]:
        """Build kwargs for the Anthropic API call."""
        kwargs: dict[str, Any] = {
            "model": self.extra_config.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": request.max_tokens or 4096,
        }

        # Add system prompt
        if request.system:
            kwargs["system"] = request.system

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop:
            kwargs["stop_sequences"] = request.stop

        # Handle tools
        if request.tools:
            kwargs["tools"] = [tool.to_anthropic_schema() for tool in request.tools]

        return kwargs

    def _convert_response(self, response: Any) -> AgentResponse:
        """Convert Anthropic response to normalized format."""
        # Extract text and tool calls from content blocks
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        text = "".join(text_parts) if text_parts else None

        # Extract usage
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

        return AgentResponse(
            text=text,
            content=[{"type": "text", "text": text}] if text else [],
            provider=self.name,
            model=response.model,
            usage=usage,
            stop_reason=response.stop_reason,
            tool_calls=tool_calls,
            raw=response,
            request_id=response.id,
        )

    def _convert_stream(self, stream: Any) -> Iterator[StreamEvent]:
        """Convert Anthropic stream to normalized events."""
        for event in stream:
            yield from self._convert_event(event)

    def _convert_event(self, event: Any) -> Iterator[StreamEvent]:
        """Convert a single stream event to normalized events."""
        event_type = getattr(event, "type", None)

        if event_type == "message_start":
            yield StreamEvent.message_start_event(raw=event)

        elif event_type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                yield StreamEvent.tool_call_start(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments={},
                    ),
                    raw=event,
                )

        elif event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                yield StreamEvent.text_delta(delta.text, raw=event)
            elif delta.type == "input_json_delta":
                yield StreamEvent.tool_call_delta_event(
                    "",  # ID not available in delta
                    {"arguments": delta.partial_json},
                    raw=event,
                )

        elif event_type == "message_delta":
            if hasattr(event, "usage") and event.usage:
                yield StreamEvent.usage_event(
                    Usage(
                        prompt_tokens=0,  # Not available in delta
                        completion_tokens=event.usage.output_tokens,
                        total_tokens=event.usage.output_tokens,
                    ),
                    raw=event,
                )

        elif event_type == "message_stop":
            yield StreamEvent.message_end(raw=event)

    def _extract_retry_after(self, error: Any) -> float | None:
        """Extract retry-after value from rate limit error."""
        if hasattr(error, "response") and error.response:
            headers = getattr(error.response, "headers", {})
            retry_after = headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return None


# Register the provider
ProviderRegistry.register("anthropic", AnthropicProvider, aliases=["claude"])
