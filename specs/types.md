# Type System Specification

This document specifies the type system for the Agent library, including all Pydantic models and their relationships.

## Overview

All types are defined as Pydantic models in the `agent/types/` module. This provides:

- **Type Safety**: Static type checking with mypy/pyright
- **Validation**: Automatic data validation
- **Serialization**: JSON encoding/decoding
- **Documentation**: Self-documenting schemas

## Module Structure

```
agent/types/
├── __init__.py          # Public exports
├── messages.py          # Message-related types
├── response.py          # Response types
├── tools.py             # Tool-related types
├── stream.py            # Streaming types
├── config.py            # Configuration types
└── router.py            # Router types
```

## Type Definitions

### messages.py

#### ContentPart

Represents a content part in a multimodal message.

```python
class ContentPart(BaseModel):
    type: Literal["text", "image", "image_url"]
    text: str | None = None
    image_url: str | None = None
    image_data: bytes | None = None
    media_type: str | None = None

    model_config = {"arbitrary_types_allowed": True}
```

**Invariants:**
- `type == "text"` requires `text` to be non-None
- `type == "image_url"` requires `image_url` to be non-None
- `type == "image"` requires `image_data` to be non-None

**Factory Methods:**
- `text_part(text: str) -> ContentPart`
- `image_url_part(url: str) -> ContentPart`
- `image_data_part(data: bytes, media_type: str = "image/png") -> ContentPart`

#### Message

Represents a conversation message.

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
```

**Invariants:**
- `role == "tool"` requires `tool_call_id` to be non-None
- `role == "assistant"` may have `tool_calls` for tool-calling responses

**Factory Methods:**
- `system(content: str) -> Message`
- `user(content: str | list[ContentPart]) -> Message`
- `assistant(content: str | None, tool_calls: list[dict] | None) -> Message`
- `tool(content: str, tool_call_id: str, name: str | None) -> Message`

**Properties:**
- `text: str` - Extracts text content from `content`

#### AgentRequest

Normalized request format for all providers.

```python
class AgentRequest(BaseModel):
    input: str | None = None
    messages: list[Message] = Field(default_factory=list)
    system: str | None = None
    tools: list[Any] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = Field(default=None)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
```

**Properties:**
- `schema: dict[str, Any] | None` - Alias for `output_schema` (backwards compatibility)

**Methods:**
- `to_messages() -> list[Message]` - Converts to full message list including system prompt

### response.py

#### Usage

Token usage information.

```python
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

**Class Methods:**
- `from_dict(data: dict[str, Any]) -> Usage`

#### AgentResponse

Normalized response from any provider.

```python
class AgentResponse(BaseModel):
    text: str | None = None
    content: list[Any] = Field(default_factory=list)
    output: Any = None
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
```

**Properties:**
- `has_tool_calls: bool` - True if `tool_calls` is non-empty

**Methods:**
- `to_dict() -> dict[str, Any]` - Serializes to dictionary

### tools.py

#### ToolSpec

Tool specification for LLM function calling.

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    function: Any | None = None
    is_async: bool = False

    model_config = {"arbitrary_types_allowed": True}
```

**Methods:**
- `to_openai_schema() -> dict[str, Any]`
- `to_anthropic_schema() -> dict[str, Any]`
- `to_gemini_schema() -> dict[str, Any]`

#### ToolCall

A tool call requested by the LLM.

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
```

**Methods:**
- `to_dict() -> dict[str, Any]`
- `from_dict(data: dict[str, Any]) -> ToolCall` (class method)

#### ToolResult

Result of executing a tool.

```python
class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
```

**Methods:**
- `to_dict() -> dict[str, Any]`

### stream.py

#### StreamEventType

```python
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

#### StreamEvent

Normalized streaming event.

```python
class StreamEvent(BaseModel):
    type: StreamEventType
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_call_delta: dict[str, Any] | None = None
    tool_result: str | None = None
    usage: Usage | None = None
    error: str | None = None
    raw: Any = None

    model_config = {"arbitrary_types_allowed": True}
```

**Factory Methods:**
- `text_delta(text: str, raw: Any = None) -> StreamEvent`
- `tool_call_start(tool_call: ToolCall, raw: Any = None) -> StreamEvent`
- `tool_call_delta_event(tool_call_id: str, delta: dict, raw: Any = None) -> StreamEvent`
- `tool_result_event(tool_call_id: str, result: str, raw: Any = None) -> StreamEvent`
- `message_start_event(raw: Any = None) -> StreamEvent`
- `message_end(usage: Usage | None = None, raw: Any = None) -> StreamEvent`
- `usage_event(usage: Usage, raw: Any = None) -> StreamEvent`
- `error_event(error: str, raw: Any = None) -> StreamEvent`

### config.py

#### ProviderCapabilities

Declares provider feature support.

```python
class ProviderCapabilities(BaseModel):
    streaming: bool = True
    tools: bool = True
    structured_output: bool = True
    json_mode: bool = True
    vision: bool = False
    system_messages: bool = True
    batch: bool = False
    native_schema_output: bool = False
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
```

#### AgentConfig

Agent instance configuration.

```python
class AgentConfig(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    default_system: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
```

**Validation:**
- `model` is resolved from aliases on construction
- `api_key` falls back to environment variable
- `base_url` falls back to provider default

**Methods:**
- `with_overrides(**kwargs) -> AgentConfig`

#### RetryConfig

Retry behavior configuration.

```python
class RetryConfig(BaseModel):
    max_retries: int = 2
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)

    model_config = {"arbitrary_types_allowed": True}
```

**Methods:**
- `should_retry(error: Exception, attempt: int) -> bool`
- `get_delay(attempt: int, error: Exception | None = None) -> float`

#### ToolLoopConfig

Tool loop behavior configuration.

```python
class ToolLoopConfig(BaseModel):
    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 20
    timeout_per_tool: float = 30.0
    parallel_tool_execution: bool = True
    stop_on_error: bool = False
```

### router.py

#### RoutingStrategy

```python
class RoutingStrategy(str, Enum):
    FALLBACK = "fallback"
    ROUND_ROBIN = "round_robin"
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    CAPABILITY = "capability"
    CUSTOM = "custom"
```

#### RouteResult

```python
class RouteResult(BaseModel):
    agent: Any  # Agent type (Any to avoid circular import)
    reason: str | None = None

    model_config = {"arbitrary_types_allowed": True}
```

## Type Relationships

```
AgentRequest
├── messages: list[Message]
│   └── content: str | list[ContentPart]
├── tools: list[ToolSpec]
└── schema: dict (JSON Schema)

AgentResponse
├── usage: Usage
├── tool_calls: list[ToolCall]
└── raw: Any (provider response)

StreamEvent
├── tool_call: ToolCall
└── usage: Usage

Tool (class, not model)
└── spec: ToolSpec

ToolLoop
├── config: ToolLoopConfig
└── registry: ToolRegistry
    └── tools: dict[str, Tool]

ExecutionRuntime
├── config: AgentConfig
├── retry_handler: RetryHandler
│   └── config: RetryConfig
└── tool_loop: ToolLoop
    └── config: ToolLoopConfig
```

## Serialization

All types support Pydantic serialization:

```python
# To dictionary
data = model.model_dump()

# To JSON string
json_str = model.model_dump_json()

# From dictionary
model = Model.model_validate(data)

# From JSON string
model = Model.model_validate_json(json_str)
```

## Backwards Compatibility

The original module locations re-export from `agent/types/`:

```python
# agent/messages.py
from agent.types.messages import ContentPart, Message, AgentRequest

# agent/response.py
from agent.types.response import Usage, AgentResponse

# agent/config.py
from agent.types.config import AgentConfig, ...
```

This ensures existing code continues to work while new code can import from `agent.types`.
