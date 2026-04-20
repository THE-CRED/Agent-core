# Provider Contract Specification

This document specifies the interface that all provider adapters must implement.

## BaseProvider Interface

```python
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator

from agent.messages import AgentRequest
from agent.response import AgentResponse
from agent.stream import StreamEvent
from agent.types import ProviderCapabilities


class BaseProvider(ABC):
    """Base class for all provider adapters."""
    
    name: str = "base"
    capabilities: ProviderCapabilities = ProviderCapabilities()
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs
    
    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a synchronous request."""
        ...
    
    @abstractmethod
    async def run_async(self, request: AgentRequest) -> AgentResponse:
        """Execute an asynchronous request."""
        ...
    
    @abstractmethod
    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        """Execute a streaming request."""
        ...
    
    @abstractmethod
    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        """Execute an asynchronous streaming request."""
        ...
```

## ProviderCapabilities

Declare what features your provider supports:

```python
class ProviderCapabilities(BaseModel):
    streaming: bool = True              # Supports streaming responses
    tools: bool = True                  # Supports tool/function calling
    structured_output: bool = True      # Supports structured output
    json_mode: bool = True              # Supports JSON mode
    vision: bool = False                # Supports image inputs
    system_messages: bool = True        # Supports system messages
    batch: bool = False                 # Supports batch requests
    native_schema_output: bool = False  # Supports native schema enforcement
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
```

## Request Conversion

Convert `AgentRequest` to provider format:

### Input Handling

```python
def _convert_messages(self, request: AgentRequest) -> list[dict]:
    messages = []
    
    # System message (provider-specific handling)
    if request.system:
        messages.append({"role": "system", "content": request.system})
    
    # Existing messages
    for msg in request.messages:
        messages.append(self._convert_message(msg))
    
    # User input
    if request.input:
        messages.append({"role": "user", "content": request.input})
    
    return messages
```

### Message Conversion

```python
def _convert_message(self, msg: Message) -> dict:
    result = {"role": msg.role, "content": msg.content}
    
    # Handle multimodal content
    if isinstance(msg.content, list):
        result["content"] = [self._convert_content_part(p) for p in msg.content]
    
    # Handle tool calls
    if msg.role == "assistant" and msg.tool_calls:
        result["tool_calls"] = [...]
    
    # Handle tool results
    if msg.role == "tool":
        result["tool_call_id"] = msg.tool_call_id
    
    return result
```

### Tool Conversion

```python
def _convert_tools(self, tools: list[ToolSpec]) -> list[dict]:
    return [tool.to_openai_schema() for tool in tools]
    # Or: [tool.to_anthropic_schema() for tool in tools]
    # Or: [tool.to_gemini_schema() for tool in tools]
```

### Parameters

```python
def _build_kwargs(self, request: AgentRequest) -> dict:
    kwargs = {"model": self.extra_config.get("model")}
    
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stop:
        kwargs["stop"] = request.stop
    if request.tools:
        kwargs["tools"] = self._convert_tools(request.tools)
    if request.schema:
        kwargs["response_format"] = {...}  # Provider-specific
    
    return kwargs
```

## Response Conversion

Convert provider response to `AgentResponse`:

```python
def _convert_response(self, api_response) -> AgentResponse:
    # Extract text
    text = api_response.choices[0].message.content
    
    # Extract tool calls
    tool_calls = []
    if api_response.choices[0].message.tool_calls:
        for tc in api_response.choices[0].message.tool_calls:
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            ))
    
    # Extract usage
    usage = None
    if api_response.usage:
        usage = Usage(
            prompt_tokens=api_response.usage.prompt_tokens,
            completion_tokens=api_response.usage.completion_tokens,
            total_tokens=api_response.usage.total_tokens,
        )
    
    return AgentResponse(
        text=text,
        content=[{"type": "text", "text": text}] if text else [],
        provider=self.name,
        model=api_response.model,
        usage=usage,
        stop_reason=api_response.choices[0].finish_reason,
        tool_calls=tool_calls,
        raw=api_response,
        request_id=api_response.id,
    )
```

## Stream Event Conversion

Convert streaming chunks to `StreamEvent`:

```python
def _convert_chunk(self, chunk) -> Iterator[StreamEvent]:
    # Text delta
    if chunk.choices[0].delta.content:
        yield StreamEvent.text_delta(
            chunk.choices[0].delta.content,
            raw=chunk,
        )
    
    # Tool call start
    if chunk.choices[0].delta.tool_calls:
        for tc in chunk.choices[0].delta.tool_calls:
            if tc.function.name:  # New tool call
                yield StreamEvent.tool_call_start(
                    ToolCall(id=tc.id, name=tc.function.name, arguments={}),
                    raw=chunk,
                )
            if tc.function.arguments:  # Arguments delta
                yield StreamEvent.tool_call_delta_event(
                    tc.id,
                    {"arguments": tc.function.arguments},
                    raw=chunk,
                )
    
    # Usage (end of stream)
    if chunk.usage:
        yield StreamEvent.usage_event(
            Usage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            ),
            raw=chunk,
        )
    
    # Message end
    if chunk.choices[0].finish_reason:
        yield StreamEvent.message_end(raw=chunk)
```

## Error Handling

Convert provider errors to Agent errors:

```python
from agent.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError as AgentTimeoutError,
)

def run(self, request: AgentRequest) -> AgentResponse:
    try:
        response = self._client.chat.completions.create(...)
        return self._convert_response(response)
    
    except provider.AuthenticationError as e:
        raise AuthenticationError(str(e), raw=e)
    
    except provider.RateLimitError as e:
        raise RateLimitError(
            str(e),
            provider=self.name,
            retry_after=self._extract_retry_after(e),
            raw=e,
        )
    
    except provider.APITimeoutError as e:
        raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e)
    
    except provider.APIError as e:
        raise ProviderError(
            str(e),
            provider=self.name,
            status_code=getattr(e, "status_code", None),
            raw=e,
        )
```

## Provider Registration

Register the provider for use:

```python
from agent.providers.registry import ProviderRegistry

# At module load time
ProviderRegistry.register(
    "myprovider",           # Provider name
    MyProvider,             # Provider class
    aliases=["my", "mp"],   # Optional aliases
)
```

## Capability Methods

Providers inherit these methods but can override:

```python
def supports_tools(self) -> bool:
    return self.capabilities.tools

def supports_structured_output(self) -> bool:
    return self.capabilities.structured_output

def supports_vision(self) -> bool:
    return self.capabilities.vision

def supports_streaming(self) -> bool:
    return self.capabilities.streaming

def supports_json_mode(self) -> bool:
    return self.capabilities.json_mode

def supports_native_schema(self) -> bool:
    return self.capabilities.native_schema_output
```

## Validation

Optional validation method:

```python
def validate_config(self) -> list[str]:
    """Return list of validation errors (empty if valid)."""
    errors = []
    if not self.api_key:
        errors.append(f"API key required for {self.name} provider")
    return errors
```

## Complete Example

```python
class MyProvider(BaseProvider):
    name = "myprovider"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=False,
        system_messages=True,
    )
    
    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self._client = MyAPIClient(api_key, base_url=self.base_url)
    
    def run(self, request: AgentRequest) -> AgentResponse:
        try:
            messages = self._convert_messages(request)
            kwargs = self._build_kwargs(request)
            response = self._client.chat(messages=messages, **kwargs)
            return self._convert_response(response)
        except MyAuthError as e:
            raise AuthenticationError(str(e), raw=e)
        except MyError as e:
            raise ProviderError(str(e), provider=self.name, raw=e)
    
    async def run_async(self, request: AgentRequest) -> AgentResponse:
        # Similar to run() but async
        ...
    
    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        messages = self._convert_messages(request)
        kwargs = self._build_kwargs(request)
        for chunk in self._client.chat_stream(messages=messages, **kwargs):
            yield from self._convert_chunk(chunk)
    
    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        # Similar to stream() but async
        ...
    
    def _convert_messages(self, request: AgentRequest) -> list[dict]:
        ...
    
    def _build_kwargs(self, request: AgentRequest) -> dict:
        ...
    
    def _convert_response(self, response) -> AgentResponse:
        ...
    
    def _convert_chunk(self, chunk) -> Iterator[StreamEvent]:
        ...


ProviderRegistry.register("myprovider", MyProvider)
```

## Testing Providers

Use the testing utilities:

```python
from agent.testing import FakeProvider, create_fake_response

def test_my_provider():
    provider = MyProvider(api_key="test")
    
    request = AgentRequest(input="Hello")
    response = provider.run(request)
    
    assert response.text is not None
    assert response.provider == "myprovider"

def test_streaming():
    provider = MyProvider(api_key="test")
    
    request = AgentRequest(input="Hello")
    events = list(provider.stream(request))
    
    assert any(e.type == "text_delta" for e in events)
    assert any(e.type == "message_end" for e in events)
```
