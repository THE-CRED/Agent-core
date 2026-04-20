"""
OpenAI provider adapter.

Supports OpenAI API and compatible endpoints.
"""

import base64
import json
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
    import openai
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider adapter.

    Supports GPT-4, GPT-4o, GPT-3.5-turbo and other OpenAI models.
    Also works with OpenAI-compatible APIs.
    """

    name = "openai"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=True,
        system_messages=True,
        batch=True,
        native_schema_output=True,
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install agent-runtime[openai]"
            )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a synchronous request."""
        try:
            messages = self._convert_messages(request)
            kwargs = self._build_kwargs(request)

            response = self._client.chat.completions.create(
                messages=messages,
                **kwargs,
            )

            return self._convert_response(response)

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except openai.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except openai.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except openai.APIError as e:
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

            response = await self._async_client.chat.completions.create(
                messages=messages,
                **kwargs,
            )

            return self._convert_response(response)

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except openai.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except openai.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except openai.APIError as e:
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
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

            response = self._client.chat.completions.create(
                messages=messages,
                **kwargs,
            )

            yield from self._convert_stream(response)

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except openai.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except openai.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except openai.APIError as e:
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
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

            response = await self._async_client.chat.completions.create(
                messages=messages,
                **kwargs,
            )

            async for chunk in response:
                for event in self._convert_chunk(chunk):
                    yield event

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), raw=e) from e
        except openai.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider=self.name,
                retry_after=self._extract_retry_after(e),
                raw=e,
            ) from e
        except openai.APITimeoutError as e:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e) from e
        except openai.APIError as e:
            raise ProviderError(
                str(e),
                provider=self.name,
                status_code=getattr(e, "status_code", None),
                raw=e,
            ) from e

    def _convert_messages(self, request: AgentRequest) -> list[dict[str, Any]]:
        """Convert normalized messages to OpenAI format."""
        messages: list[dict[str, Any]] = []

        # Add system message
        if request.system:
            messages.append({"role": "system", "content": request.system})

        # Convert existing messages
        for msg in request.messages:
            openai_msg = self._convert_message(msg)
            messages.append(openai_msg)

        # Add input as user message
        if request.input:
            messages.append({"role": "user", "content": request.input})

        return messages

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """Convert a single message to OpenAI format."""
        result: dict[str, Any] = {"role": msg.role}

        # Handle content
        if isinstance(msg.content, str):
            result["content"] = msg.content
        else:
            # Multi-part content (text + images)
            parts = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    parts.append({"type": "text", "text": part.text})
                elif part.type == "image_url" and part.image_url:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url},
                    })
                elif part.type == "image" and part.image_data:
                    b64_data = base64.b64encode(part.image_data).decode()
                    media_type = part.media_type or "image/png"
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
                    })
            result["content"] = parts

        # Handle tool-related fields
        if msg.role == "assistant" and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments", {})),
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.role == "tool":
            result["tool_call_id"] = msg.tool_call_id

        if msg.name:
            result["name"] = msg.name

        return result

    def _build_kwargs(self, request: AgentRequest) -> dict[str, Any]:
        """Build kwargs for the OpenAI API call."""
        kwargs: dict[str, Any] = {
            "model": self.extra_config.get("model", "gpt-4o"),
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop:
            kwargs["stop"] = request.stop

        # Handle tools
        if request.tools:
            kwargs["tools"] = [tool.to_openai_schema() for tool in request.tools]

        # Handle structured output / JSON mode
        if request.schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema.get("title", "response"),
                    "schema": request.schema,
                    "strict": True,
                },
            }

        return kwargs

    def _convert_response(self, response: "ChatCompletion") -> AgentResponse:
        """Convert OpenAI response to normalized format."""
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None

        # Extract text
        text = message.content if message else None

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        if message and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                    )
                )

        # Extract usage
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return AgentResponse(
            text=text,
            content=[{"type": "text", "text": text}] if text else [],
            provider=self.name,
            model=response.model,
            usage=usage,
            stop_reason=choice.finish_reason if choice else None,
            tool_calls=tool_calls,
            raw=response,
            request_id=response.id,
        )

    def _convert_stream(self, response: Any) -> Iterator[StreamEvent]:
        """Convert OpenAI stream to normalized events."""
        current_tool_calls: dict[int, dict[str, Any]] = {}

        for chunk in response:
            yield from self._convert_chunk(chunk, current_tool_calls)

    def _convert_chunk(
        self,
        chunk: "ChatCompletionChunk",
        current_tool_calls: dict[int, dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        """Convert a single chunk to stream events."""
        if current_tool_calls is None:
            current_tool_calls = {}

        if not chunk.choices:
            # Usage chunk at the end
            if chunk.usage:
                yield StreamEvent.usage_event(
                    Usage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    ),
                    raw=chunk,
                )
            return

        choice = chunk.choices[0]
        delta = choice.delta

        # Text delta
        if delta.content:
            yield StreamEvent.text_delta(delta.content, raw=chunk)

        # Tool call deltas
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index

                if idx not in current_tool_calls:
                    # New tool call
                    current_tool_calls[idx] = {
                        "id": tc_delta.id or "",
                        "name": tc_delta.function.name if tc_delta.function else "",
                        "arguments": "",
                    }
                    if tc_delta.id:
                        yield StreamEvent.tool_call_start(
                            ToolCall(
                                id=tc_delta.id,
                                name=tc_delta.function.name if tc_delta.function else "",
                                arguments={},
                            ),
                            raw=chunk,
                        )

                # Accumulate arguments
                if tc_delta.function and tc_delta.function.arguments:
                    current_tool_calls[idx]["arguments"] += tc_delta.function.arguments
                    yield StreamEvent.tool_call_delta_event(
                        current_tool_calls[idx]["id"],
                        {"arguments": tc_delta.function.arguments},
                        raw=chunk,
                    )

        # End of message
        if choice.finish_reason:
            yield StreamEvent.message_end(raw=chunk)

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
ProviderRegistry.register("openai", OpenAIProvider, aliases=["gpt", "chatgpt"])
