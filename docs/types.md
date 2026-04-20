# Type System

Agent uses Pydantic models for all core types, providing type safety, validation, and serialization.

## Type Organization

All types are centralized in `agent/types/`:

```
agent/types/
├── __init__.py      # Re-exports all types
├── messages.py      # ContentPart, Message, AgentRequest
├── response.py      # Usage, AgentResponse
├── tools.py         # ToolSpec, ToolCall, ToolResult
├── stream.py        # StreamEvent, StreamEventType
├── config.py        # AgentConfig, ProviderCapabilities, RetryConfig, ToolLoopConfig
└── router.py        # RoutingStrategy, RouteResult
```

## Importing Types

```python
# Import from main package
from agent import (
    Message,
    AgentRequest,
    AgentResponse,
    Usage,
    ToolSpec,
    ToolCall,
    ToolResult,
    StreamEvent,
    AgentConfig,
    ProviderCapabilities,
)

# Or import from types module directly
from agent.types import (
    ContentPart,
    Message,
    AgentRequest,
    Usage,
    AgentResponse,
    ToolSpec,
    ToolCall,
    ToolResult,
    StreamEvent,
    StreamEventType,
    AgentConfig,
    ProviderCapabilities,
    RetryConfig,
    ToolLoopConfig,
    RoutingStrategy,
    RouteResult,
)
```

## Message Types

### ContentPart

Represents a part of message content (text, image, etc.):

```python
from agent.types import ContentPart

# Text content
text = ContentPart.text_part("Hello, world!")

# Image from URL
image_url = ContentPart.image_url_part("https://example.com/image.jpg")

# Image from bytes
with open("photo.png", "rb") as f:
    image_data = ContentPart.image_data_part(f.read(), media_type="image/png")
```

**Fields:**
- `type`: `Literal["text", "image", "image_url"]`
- `text`: `str | None`
- `image_url`: `str | None`
- `image_data`: `bytes | None`
- `media_type`: `str | None`

### Message

A normalized message in a conversation:

```python
from agent.types import Message

# Create messages using factory methods
system = Message.system("You are a helpful assistant.")
user = Message.user("What's the weather?")
assistant = Message.assistant("I'd be happy to help with that.")
tool_result = Message.tool("72F and sunny", tool_call_id="call_123", name="get_weather")

# Access content
print(user.text)  # "What's the weather?"
print(user.role)  # "user"
```

**Fields:**
- `role`: `Literal["system", "user", "assistant", "tool"]`
- `content`: `str | list[ContentPart]`
- `name`: `str | None`
- `tool_call_id`: `str | None`
- `tool_calls`: `list[dict[str, Any]] | None`

### AgentRequest

A normalized request to be sent to a provider:

```python
from agent.types import AgentRequest, Message

request = AgentRequest(
    input="What's 2+2?",
    messages=[
        Message.system("You are a math tutor."),
    ],
    system="Be concise.",
    temperature=0.7,
    max_tokens=100,
)

# Convert to full message list
all_messages = request.to_messages()
```

**Fields:**
- `input`: `str | None`
- `messages`: `list[Message]`
- `system`: `str | None`
- `tools`: `list[Any]`
- `output_schema`: `dict[str, Any] | None` (alias: `schema`)
- `temperature`: `float | None`
- `max_tokens`: `int | None`
- `top_p`: `float | None`
- `stop`: `list[str] | None`
- `metadata`: `dict[str, Any]`
- `session_id`: `str | None`

## Response Types

### Usage

Token usage information:

```python
from agent.types import Usage

usage = Usage(
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
)

# From dictionary
usage = Usage.from_dict({"prompt_tokens": 100, "completion_tokens": 50})
```

**Fields:**
- `prompt_tokens`: `int`
- `completion_tokens`: `int`
- `total_tokens`: `int`

### AgentResponse

Normalized response from any provider:

```python
from agent.types import AgentResponse

response = AgentResponse(
    text="Hello!",
    provider="openai",
    model="gpt-4o",
    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
)

# Check for tool calls
if response.has_tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}, Args: {tc.arguments}")

# Serialize
data = response.to_dict()
```

**Fields:**
- `text`: `str | None`
- `content`: `list[Any]`
- `output`: `Any` - Parsed structured output
- `provider`: `str`
- `model`: `str`
- `usage`: `Usage | None`
- `stop_reason`: `str | None`
- `tool_calls`: `list[ToolCall]`
- `raw`: `Any` - Provider's raw response
- `latency_ms`: `float | None`
- `cost_estimate`: `float | None`
- `request_id`: `str | None`

## Tool Types

### ToolSpec

Specification for a tool:

```python
from agent.types import ToolSpec

spec = ToolSpec(
    name="get_weather",
    description="Get weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
        },
        "required": ["city"],
    },
)

# Convert to provider formats
openai_format = spec.to_openai_schema()
anthropic_format = spec.to_anthropic_schema()
gemini_format = spec.to_gemini_schema()
```

**Fields:**
- `name`: `str`
- `description`: `str`
- `parameters`: `dict[str, Any]`
- `function`: `Any | None`
- `is_async`: `bool`

### ToolCall

A tool call requested by the LLM:

```python
from agent.types import ToolCall

call = ToolCall(
    id="call_abc123",
    name="get_weather",
    arguments={"city": "Tokyo"},
)

# Serialize/deserialize
data = call.to_dict()
restored = ToolCall.from_dict(data)
```

**Fields:**
- `id`: `str`
- `name`: `str`
- `arguments`: `dict[str, Any]`

### ToolResult

Result of executing a tool:

```python
from agent.types import ToolResult

result = ToolResult(
    tool_call_id="call_abc123",
    name="get_weather",
    content="72F and sunny",
    is_error=False,
)

# Error result
error_result = ToolResult(
    tool_call_id="call_xyz",
    name="failing_tool",
    content="Error: connection timeout",
    is_error=True,
)
```

**Fields:**
- `tool_call_id`: `str`
- `name`: `str`
- `content`: `str`
- `is_error`: `bool`

## Stream Types

### StreamEventType

```python
from agent.types import StreamEventType

# Type alias for event types
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
```

### StreamEvent

A normalized streaming event:

```python
from agent.types import StreamEvent

# Create events using factory methods
text_event = StreamEvent.text_delta("Hello", raw=chunk)
tool_start = StreamEvent.tool_call_start(tool_call, raw=chunk)
end_event = StreamEvent.message_end(usage=usage, raw=chunk)
error_event = StreamEvent.error_event("Connection lost")
```

**Fields:**
- `type`: `StreamEventType`
- `text`: `str | None`
- `tool_call`: `ToolCall | None`
- `tool_call_delta`: `dict[str, Any] | None`
- `tool_result`: `str | None`
- `usage`: `Usage | None`
- `error`: `str | None`
- `raw`: `Any`

## Configuration Types

### AgentConfig

Configuration for an Agent instance:

```python
from agent.types import AgentConfig

config = AgentConfig(
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    timeout=120.0,
    max_retries=3,
    temperature=0.7,
)

# Create modified config
new_config = config.with_overrides(temperature=0.9, max_tokens=500)
```

**Fields:**
- `provider`: `str`
- `model`: `str`
- `api_key`: `str | None`
- `base_url`: `str | None`
- `timeout`: `float`
- `max_retries`: `int`
- `temperature`: `float | None`
- `max_tokens`: `int | None`
- `top_p`: `float | None`
- `default_system`: `str | None`
- `extra`: `dict[str, Any]`

### ProviderCapabilities

Declares what features a provider supports:

```python
from agent.types import ProviderCapabilities

caps = ProviderCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    vision=True,
    system_messages=True,
    batch=False,
    native_schema_output=True,
)
```

**Fields:**
- `streaming`: `bool`
- `tools`: `bool`
- `structured_output`: `bool`
- `json_mode`: `bool`
- `vision`: `bool`
- `system_messages`: `bool`
- `batch`: `bool`
- `native_schema_output`: `bool`
- `max_context_tokens`: `int | None`
- `max_output_tokens`: `int | None`

### RetryConfig

Configuration for retry behavior:

```python
from agent.types import RetryConfig

config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
)

# Check if should retry
if config.should_retry(error, attempt=1):
    delay = config.get_delay(attempt=1, error=error)
    time.sleep(delay)
```

**Fields:**
- `max_retries`: `int`
- `initial_delay`: `float`
- `max_delay`: `float`
- `exponential_base`: `float`
- `jitter`: `bool`
- `retryable_errors`: `tuple[type[Exception], ...]`

### ToolLoopConfig

Configuration for tool loop behavior:

```python
from agent.types import ToolLoopConfig

config = ToolLoopConfig(
    max_iterations=10,
    max_tool_calls_per_iteration=20,
    timeout_per_tool=30.0,
    parallel_tool_execution=True,
    stop_on_error=False,
)
```

**Fields:**
- `max_iterations`: `int`
- `max_tool_calls_per_iteration`: `int`
- `timeout_per_tool`: `float`
- `parallel_tool_execution`: `bool`
- `stop_on_error`: `bool`

## Router Types

### RoutingStrategy

```python
from agent.types import RoutingStrategy

strategy = RoutingStrategy.FALLBACK
# Options: FALLBACK, ROUND_ROBIN, FASTEST, CHEAPEST, CAPABILITY, CUSTOM
```

### RouteResult

```python
from agent.types import RouteResult

result = RouteResult(
    agent=selected_agent,
    reason="Cheapest available",
)
```

**Fields:**
- `agent`: `Any` (Agent)
- `reason`: `str | None`

## Type Validation

All Pydantic models include automatic validation:

```python
from agent.types import Message

# Valid
msg = Message(role="user", content="Hello")

# Invalid - raises ValidationError
msg = Message(role="invalid", content="Hello")
```

## Serialization

All types support JSON serialization:

```python
from agent.types import Message, AgentResponse

# To dictionary
msg = Message.user("Hello")
data = msg.model_dump()

# To JSON string
json_str = msg.model_dump_json()

# From dictionary
restored = Message.model_validate(data)

# From JSON string
restored = Message.model_validate_json(json_str)
```

## Type Hints

Use types for static type checking:

```python
from agent.types import Message, AgentResponse, ToolCall

def process_response(response: AgentResponse) -> list[ToolCall]:
    """Extract tool calls from response."""
    return response.tool_calls

def format_messages(messages: list[Message]) -> str:
    """Format messages for display."""
    return "\n".join(f"[{m.role}] {m.text}" for m in messages)
```

## Next Steps

- [Structured Outputs](structured-outputs.md) - Using Pydantic for outputs
- [Tools](tools.md) - Tool type specifications
- [Custom Providers](custom-providers.md) - Using types in providers
