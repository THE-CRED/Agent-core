# Custom Providers

Implement custom provider adapters to support additional LLM backends or specialized endpoints.

## Provider Interface

All providers must extend `BaseProvider` and implement the abstract methods:

```python
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator

from agent.providers.base import BaseProvider
from agent.messages import AgentRequest
from agent.response import AgentResponse
from agent.stream import StreamEvent
from agent.types import ProviderCapabilities


class CustomProvider(BaseProvider):
    """Your custom provider implementation."""
    
    name = "custom"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=False,
        system_messages=True,
    )
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        super().__init__(api_key, base_url, timeout, max_retries, **kwargs)
        # Initialize your client
        self.client = MyAPIClient(api_key, base_url)
    
    def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a synchronous request."""
        # Convert request to your API format
        api_request = self._convert_request(request)
        
        # Call your API
        api_response = self.client.chat(api_request)
        
        # Convert to normalized response
        return self._convert_response(api_response)
    
    async def run_async(self, request: AgentRequest) -> AgentResponse:
        """Execute an asynchronous request."""
        api_request = self._convert_request(request)
        api_response = await self.client.chat_async(api_request)
        return self._convert_response(api_response)
    
    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        """Execute a streaming request."""
        api_request = self._convert_request(request)
        
        for chunk in self.client.chat_stream(api_request):
            yield self._convert_chunk(chunk)
    
    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        """Execute an async streaming request."""
        api_request = self._convert_request(request)
        
        async for chunk in self.client.chat_stream_async(api_request):
            yield self._convert_chunk(chunk)
```

## Converting Requests

Transform the normalized `AgentRequest` to your API format:

```python
def _convert_request(self, request: AgentRequest) -> dict:
    """Convert AgentRequest to API format."""
    messages = []
    
    # Handle system message
    if request.system:
        messages.append({
            "role": "system",
            "content": request.system,
        })
    
    # Convert existing messages
    for msg in request.messages:
        messages.append(self._convert_message(msg))
    
    # Add user input
    if request.input:
        messages.append({
            "role": "user",
            "content": request.input,
        })
    
    api_request = {
        "messages": messages,
        "model": self.extra_config.get("model", "default-model"),
    }
    
    # Add optional parameters
    if request.temperature is not None:
        api_request["temperature"] = request.temperature
    if request.max_tokens is not None:
        api_request["max_tokens"] = request.max_tokens
    
    # Handle tools
    if request.tools:
        api_request["tools"] = [
            tool.to_openai_schema()  # Or your own format
            for tool in request.tools
        ]
    
    return api_request

def _convert_message(self, msg: Message) -> dict:
    """Convert a single message."""
    result = {"role": msg.role, "content": msg.content}
    
    if msg.role == "assistant" and msg.tool_calls:
        result["tool_calls"] = msg.tool_calls
    
    if msg.role == "tool":
        result["tool_call_id"] = msg.tool_call_id
    
    return result
```

## Converting Responses

Transform API responses to the normalized `AgentResponse`:

```python
from agent.types import Usage, ToolCall

def _convert_response(self, api_response: dict) -> AgentResponse:
    """Convert API response to AgentResponse."""
    
    # Extract text content
    text = api_response.get("content", "")
    
    # Extract tool calls
    tool_calls = []
    for tc in api_response.get("tool_calls", []):
        tool_calls.append(ToolCall(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=json.loads(tc["function"]["arguments"]),
        ))
    
    # Extract usage
    usage = None
    if "usage" in api_response:
        usage = Usage(
            prompt_tokens=api_response["usage"]["prompt_tokens"],
            completion_tokens=api_response["usage"]["completion_tokens"],
            total_tokens=api_response["usage"]["total_tokens"],
        )
    
    return AgentResponse(
        text=text,
        content=[{"type": "text", "text": text}] if text else [],
        provider=self.name,
        model=api_response.get("model", ""),
        usage=usage,
        stop_reason=api_response.get("finish_reason"),
        tool_calls=tool_calls,
        raw=api_response,
    )
```

## Converting Stream Events

Transform streaming chunks to normalized `StreamEvent`:

```python
def _convert_chunk(self, chunk: dict) -> Iterator[StreamEvent]:
    """Convert a streaming chunk to events."""
    
    # Text delta
    if "delta" in chunk and "content" in chunk["delta"]:
        yield StreamEvent.text_delta(
            chunk["delta"]["content"],
            raw=chunk,
        )
    
    # Tool call
    if "delta" in chunk and "tool_calls" in chunk["delta"]:
        for tc in chunk["delta"]["tool_calls"]:
            if tc.get("function", {}).get("name"):
                # New tool call
                yield StreamEvent.tool_call_start(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments={},
                    ),
                    raw=chunk,
                )
            elif tc.get("function", {}).get("arguments"):
                # Argument delta
                yield StreamEvent.tool_call_delta_event(
                    tc["id"],
                    {"arguments": tc["function"]["arguments"]},
                    raw=chunk,
                )
    
    # Usage
    if "usage" in chunk:
        yield StreamEvent.usage_event(
            Usage(
                prompt_tokens=chunk["usage"]["prompt_tokens"],
                completion_tokens=chunk["usage"]["completion_tokens"],
                total_tokens=chunk["usage"]["total_tokens"],
            ),
            raw=chunk,
        )
    
    # End of message
    if chunk.get("finish_reason"):
        yield StreamEvent.message_end(raw=chunk)
```

## Error Handling

Convert API errors to Agent errors:

```python
from agent.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError as AgentTimeoutError,
)

def run(self, request: AgentRequest) -> AgentResponse:
    try:
        api_response = self.client.chat(...)
        return self._convert_response(api_response)
    except MyAuthError as e:
        raise AuthenticationError(str(e), raw=e)
    except MyRateLimitError as e:
        raise RateLimitError(
            str(e),
            provider=self.name,
            retry_after=e.retry_after,
            raw=e,
        )
    except MyTimeoutError as e:
        raise AgentTimeoutError(str(e), timeout=self.timeout, raw=e)
    except MyAPIError as e:
        raise ProviderError(
            str(e),
            provider=self.name,
            status_code=e.status_code,
            raw=e,
        )
```

## Registering Your Provider

Register the provider for use with the `Agent` class:

```python
from agent.providers.registry import ProviderRegistry

# Register with optional aliases
ProviderRegistry.register(
    "custom",
    CustomProvider,
    aliases=["my-provider", "custom-llm"],
)

# Now you can use it
from agent import Agent

agent = Agent(
    provider="custom",  # or "my-provider"
    model="my-model",
    api_key="...",
)
```

## OpenAI-Compatible Providers

For APIs that follow the OpenAI format, extend the OpenAI provider:

```python
from agent.providers.openai import OpenAIProvider
from agent.providers.registry import ProviderRegistry
from agent.types import ProviderCapabilities


class GroqProvider(OpenAIProvider):
    """Groq provider - OpenAI-compatible API."""
    
    name = "groq"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=False,  # Groq doesn't support vision
        system_messages=True,
    )
    
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or self.GROQ_BASE_URL,
            **kwargs,
        )


ProviderRegistry.register("groq", GroqProvider)
```

## Testing Your Provider

```python
import pytest
from agent import Agent
from agent.providers.registry import ProviderRegistry

# Register test provider
ProviderRegistry.register("test", TestProvider)

def test_basic_run():
    agent = Agent(provider="test", model="test-model")
    response = agent.run("Hello")
    assert response.text is not None
    assert response.provider == "test"

def test_streaming():
    agent = Agent(provider="test", model="test-model")
    events = list(agent.stream("Hello"))
    assert any(e.type == "text_delta" for e in events)
    assert any(e.type == "message_end" for e in events)

def test_tool_calling():
    @tool
    def my_tool(x: str) -> str:
        return x
    
    agent = Agent(provider="test", model="test-model", tools=[my_tool])
    response = agent.run("Use the tool")
    # Verify tool calls work

@pytest.mark.asyncio
async def test_async():
    agent = Agent(provider="test", model="test-model")
    response = await agent.run_async("Hello")
    assert response.text is not None
```

## Complete Example

Here's a complete example for a hypothetical API:

```python
"""
MyLLM Provider Adapter
"""

import json
from typing import Any, AsyncIterator, Iterator

import httpx

from agent.errors import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from agent.messages import AgentRequest, Message
from agent.providers.base import BaseProvider
from agent.providers.registry import ProviderRegistry
from agent.response import AgentResponse
from agent.stream import StreamEvent
from agent.types import ProviderCapabilities, Usage, ToolCall


class MyLLMProvider(BaseProvider):
    """Provider adapter for MyLLM API."""
    
    name = "myllm"
    capabilities = ProviderCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        vision=True,
        system_messages=True,
    )
    
    BASE_URL = "https://api.myllm.com/v1"
    
    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.client = httpx.Client(
            base_url=self.base_url or self.BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout,
        )
        self.async_client = httpx.AsyncClient(
            base_url=self.base_url or self.BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout,
        )
    
    def run(self, request: AgentRequest) -> AgentResponse:
        try:
            response = self.client.post(
                "/chat/completions",
                json=self._build_request(request),
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except httpx.HTTPStatusError as e:
            self._handle_error(e)
    
    async def run_async(self, request: AgentRequest) -> AgentResponse:
        try:
            response = await self.async_client.post(
                "/chat/completions",
                json=self._build_request(request),
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except httpx.HTTPStatusError as e:
            self._handle_error(e)
    
    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]:
        req = self._build_request(request)
        req["stream"] = True
        
        with self.client.stream("POST", "/chat/completions", json=req) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield from self._parse_chunk(data)
    
    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]:
        req = self._build_request(request)
        req["stream"] = True
        
        async with self.async_client.stream("POST", "/chat/completions", json=req) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    for event in self._parse_chunk(data):
                        yield event
    
    def _build_request(self, request: AgentRequest) -> dict:
        # ... implementation
        pass
    
    def _parse_response(self, data: dict) -> AgentResponse:
        # ... implementation
        pass
    
    def _parse_chunk(self, data: dict) -> Iterator[StreamEvent]:
        # ... implementation
        pass
    
    def _handle_error(self, error: httpx.HTTPStatusError):
        if error.response.status_code == 401:
            raise AuthenticationError("Invalid API key", raw=error)
        elif error.response.status_code == 429:
            raise RateLimitError("Rate limited", provider=self.name, raw=error)
        else:
            raise ProviderError(str(error), provider=self.name, raw=error)


# Register the provider
ProviderRegistry.register("myllm", MyLLMProvider, aliases=["my-llm"])
```

## Next Steps

- [Providers](providers.md) - See existing provider implementations
- [Type System](types.md) - Understanding Agent's types
- [Middleware](middleware.md) - Provider-specific middleware
