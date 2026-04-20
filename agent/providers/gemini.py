"""
Google Gemini provider adapter.
"""

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
    import google.generativeai as genai
    from google.generativeai.types import (
        Content,
        GenerationConfig,
        Part,
    )
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None  # type: ignore


class GeminiProvider(BaseProvider):
    """
    Google Gemini provider adapter.

    Supports Gemini Pro, Gemini Flash, and other Gemini models.
    """

    name = "gemini"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=True,
        system_messages=True,
        batch=False,
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
        if not HAS_GEMINI:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Install with: pip install agent-runtime[gemini]"
            )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Configure the SDK
        genai.configure(api_key=api_key)

        # Store model name for later
        self._model_name = kwargs.get("model", "gemini-1.5-pro")

    def _get_model(self, request: AgentRequest) -> Any:
        """Get a configured Gemini model."""
        model_name = self.extra_config.get("model", self._model_name)

        # Build generation config
        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            top_p=request.top_p,
            stop_sequences=request.stop,
        )

        # Build tools if present
        tools = None
        if request.tools:
            tools = [self._convert_tool(t) for t in request.tools]

        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            tools=tools,
            system_instruction=request.system,
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a synchronous request."""
        try:
            model = self._get_model(request)
            contents = self._convert_messages(request)

            response = model.generate_content(contents)

            return self._convert_response(response)

        except Exception as e:
            self._handle_error(e)

    async def run_async(self, request: AgentRequest) -> AgentResponse:
        """Execute an asynchronous request."""
        try:
            model = self._get_model(request)
            contents = self._convert_messages(request)

            response = await model.generate_content_async(contents)

            return self._convert_response(response)

        except Exception as e:
            self._handle_error(e)

    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        """Execute a streaming request."""
        try:
            model = self._get_model(request)
            contents = self._convert_messages(request)

            response = model.generate_content(contents, stream=True)

            for chunk in response:
                yield from self._convert_chunk(chunk)

            yield StreamEvent.message_end()

        except Exception as e:
            self._handle_error(e)

    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        """Execute an async streaming request."""
        try:
            model = self._get_model(request)
            contents = self._convert_messages(request)

            response = await model.generate_content_async(contents, stream=True)

            async for chunk in response:
                for event in self._convert_chunk(chunk):
                    yield event

            yield StreamEvent.message_end()

        except Exception as e:
            self._handle_error(e)

    def _convert_messages(self, request: AgentRequest) -> list[Any]:
        """Convert normalized messages to Gemini format."""
        contents = []

        for msg in request.messages:
            content = self._convert_message(msg)
            if content:
                contents.append(content)

        # Add input as user message
        if request.input:
            contents.append(Content(role="user", parts=[Part.from_text(request.input)]))

        return contents

    def _convert_message(self, msg: Message) -> Any | None:
        """Convert a single message to Gemini format."""
        if msg.role == "system":
            # System messages handled separately in Gemini
            return None

        # Map roles
        role = "user" if msg.role in ("user", "tool") else "model"

        parts = []

        # Handle content
        if isinstance(msg.content, str):
            if msg.role == "tool":
                # Tool results need special handling
                parts.append(Part.from_function_response(
                    name=msg.name or "tool",
                    response={"result": msg.content},
                ))
            else:
                parts.append(Part.from_text(msg.content))
        else:
            for part in msg.content:
                if part.type == "text" and part.text:
                    parts.append(Part.from_text(part.text))
                elif part.type == "image" and part.image_data:
                    parts.append(Part.from_data(
                        data=part.image_data,
                        mime_type=part.media_type or "image/png",
                    ))
                elif part.type == "image_url" and part.image_url:
                    # Gemini prefers inline data, but we can try URL
                    parts.append(Part.from_uri(
                        uri=part.image_url,
                        mime_type="image/jpeg",
                    ))

        # Handle tool calls in assistant messages
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(Part.from_function_call(
                    name=tc["name"],
                    args=tc.get("arguments", {}),
                ))

        return Content(role=role, parts=parts)

    def _convert_tool(self, tool_spec: Any) -> Any:
        """Convert tool spec to Gemini format."""
        schema = tool_spec.to_gemini_schema()
        return genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            k: self._convert_schema_property(v)
                            for k, v in schema["parameters"].get("properties", {}).items()
                        },
                        required=schema["parameters"].get("required", []),
                    ),
                )
            ]
        )

    def _convert_schema_property(self, prop: dict[str, Any]) -> Any:
        """Convert a JSON Schema property to Gemini format."""
        type_map = {
            "string": genai.protos.Type.STRING,
            "integer": genai.protos.Type.INTEGER,
            "number": genai.protos.Type.NUMBER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT,
        }

        schema_type = type_map.get(prop.get("type", "string"), genai.protos.Type.STRING)

        return genai.protos.Schema(
            type=schema_type,
            description=prop.get("description", ""),
        )

    def _convert_response(self, response: Any) -> AgentResponse:
        """Convert Gemini response to normalized format."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{fc.name}_{len(tool_calls)}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        text = "".join(text_parts) if text_parts else None

        # Extract usage if available
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = Usage(
                prompt_tokens=getattr(um, "prompt_token_count", 0),
                completion_tokens=getattr(um, "candidates_token_count", 0),
                total_tokens=getattr(um, "total_token_count", 0),
            )

        # Determine stop reason
        stop_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            stop_reason = str(finish_reason.name) if finish_reason else None

        return AgentResponse(
            text=text,
            content=[{"type": "text", "text": text}] if text else [],
            provider=self.name,
            model=self._model_name,
            usage=usage,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            raw=response,
        )

    def _convert_chunk(self, chunk: Any) -> Iterator[StreamEvent]:
        """Convert a streaming chunk to events."""
        if chunk.candidates:
            candidate = chunk.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    yield StreamEvent.text_delta(part.text, raw=chunk)
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    yield StreamEvent.tool_call_start(
                        ToolCall(
                            id=f"call_{fc.name}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        ),
                        raw=chunk,
                    )

    def _handle_error(self, e: Exception) -> None:
        """Convert Gemini errors to Agent errors."""
        error_str = str(e).lower()

        if "api key" in error_str or "authentication" in error_str:
            raise AuthenticationError(str(e), raw=e)
        elif "rate limit" in error_str or "quota" in error_str:
            raise RateLimitError(str(e), provider=self.name, raw=e)
        elif "timeout" in error_str:
            raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e)
        else:
            raise ProviderError(str(e), provider=self.name, raw=e)


# Register the provider
ProviderRegistry.register("gemini", GeminiProvider, aliases=["google", "palm"])
