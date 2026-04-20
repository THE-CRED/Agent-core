# API Reference

Complete API documentation for the Agent library.

## Agent

The main class for interacting with LLM providers.

### Constructor

```python
Agent(
    provider: str,                      # Provider name: "openai", "anthropic", "gemini", "deepseek"
    model: str,                         # Model name: "gpt-4o", "claude-sonnet", etc.
    *,
    api_key: str | None = None,         # API key (defaults to env var)
    base_url: str | None = None,        # Custom API endpoint
    timeout: float = 120.0,             # Request timeout in seconds
    max_retries: int = 2,               # Retry attempts for transient errors
    temperature: float | None = None,   # Sampling temperature (0.0-2.0)
    max_tokens: int | None = None,      # Maximum tokens to generate
    top_p: float | None = None,         # Top-p sampling parameter
    tools: list[Tool] | None = None,    # Tools available to the agent
    middleware: list[Middleware] | None = None,  # Middleware chain
    default_system: str | None = None,  # Default system prompt
)
```

### Methods

#### run()

Execute a synchronous request.

```python
def run(
    input: str | None = None,           # User input text
    *,
    messages: list[Message] | None = None,  # Message history
    system: str | None = None,          # System prompt (overrides default)
    temperature: float | None = None,   # Temperature override
    max_tokens: int | None = None,      # Max tokens override
    stop: list[str] | None = None,      # Stop sequences
    metadata: dict[str, Any] | None = None,  # Request metadata
) -> AgentResponse
```

#### run_async()

Execute an asynchronous request.

```python
async def run_async(...) -> AgentResponse
# Same parameters as run()
```

#### stream()

Execute a streaming request.

```python
def stream(
    input: str | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> StreamResponse
```

#### stream_async()

Execute an async streaming request.

```python
async def stream_async(...) -> AsyncStreamResponse
# Same parameters as stream()
```

#### json()

Execute a request expecting structured JSON output.

```python
def json(
    input: str | None = None,
    *,
    schema: Type[BaseModel] | dict[str, Any],  # Output schema (required)
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### json_async()

```python
async def json_async(...) -> AgentResponse
# Same parameters as json()
```

#### session()

Create a new session for multi-turn conversation.

```python
def session(
    session_id: str | None = None,  # Optional session identifier
    system: str | None = None,      # System prompt for this session
) -> Session
```

#### with_config()

Create a new Agent with modified configuration.

```python
def with_config(**kwargs) -> Agent
```

### Properties

```python
agent.provider  # str: Provider name
agent.model     # str: Model name
agent.tools     # list[Tool]: Registered tools
agent.config    # AgentConfig: Full configuration
```

---

## AgentResponse

Normalized response from any provider.

### Attributes

```python
response.text           # str | None: Generated text
response.content        # list[Any]: Content blocks
response.output         # Any: Parsed structured output (for json())
response.provider       # str: Provider name
response.model          # str: Model name
response.usage          # Usage | None: Token usage
response.stop_reason    # str | None: Why generation stopped
response.tool_calls     # list[ToolCall]: Requested tool calls
response.raw            # Any: Raw provider response
response.latency_ms     # float | None: Request latency
response.cost_estimate  # float | None: Estimated cost in USD
response.request_id     # str | None: Provider request ID
```

### Properties

```python
response.has_tool_calls  # bool: True if response contains tool calls
```

### Methods

```python
response.to_dict()  # dict: Convert to dictionary
```

---

## Usage

Token usage information.

```python
usage.prompt_tokens      # int: Input tokens
usage.completion_tokens  # int: Output tokens
usage.total_tokens       # int: Total tokens
```

---

## Session

Multi-turn conversation manager.

### Constructor

```python
Session(
    agent: Agent,
    session_id: str | None = None,
    system: str | None = None,
    messages: list[Message] | None = None,
)
```

### Methods

#### run() / run_async()

Send a message and get a response. Automatically updates history.

```python
def run(
    input: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### stream() / stream_async()

Stream a response. History updated after consumption.

```python
def stream(
    input: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> StreamResponse
```

#### json()

Structured output in session context.

```python
def json(
    input: str,
    *,
    schema: Type[BaseModel] | dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### history()

Get message history (returns a copy).

```python
def history() -> list[Message]
```

#### clear()

Clear message history.

```python
def clear() -> None
```

#### fork()

Create a new session with copied history.

```python
def fork(session_id: str | None = None) -> Session
```

#### to_dict() / from_dict()

Serialize/deserialize session state.

```python
def to_dict() -> dict[str, Any]

@classmethod
def from_dict(data: dict[str, Any], agent: Agent) -> Session
```

### Properties

```python
session.session_id  # str: Session identifier
session.system      # str | None: System prompt
session.messages    # list[Message]: Message history (read-only)
```

---

## AgentRouter

Routes requests across multiple agents.

### Constructor

```python
AgentRouter(
    agents: list[Agent],                # Agents to route between
    strategy: RoutingStrategy | str = "fallback",  # Routing strategy
    custom_router: Callable | None = None,  # Custom routing function
)
```

### Strategies

```python
"fallback"      # Try each agent until one succeeds
"round_robin"   # Rotate through agents
"fastest"       # Race agents, use first response
"cheapest"      # Use cheapest available agent
"capability"    # Route based on required capabilities
"custom"        # User-provided routing function
```

### Methods

All methods mirror the Agent API:

```python
router.run(...)
router.run_async(...)
router.stream(...)
router.stream_async(...)
router.json(...)
```

---

## Message

A normalized message in a conversation.

### Constructor

```python
Message(
    role: Literal["system", "user", "assistant", "tool"],
    content: str | list[ContentPart],
    name: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict] | None = None,
)
```

### Class Methods

```python
Message.system(content: str) -> Message
Message.user(content: str | list[ContentPart]) -> Message
Message.assistant(content: str, tool_calls: list | None = None) -> Message
Message.tool(content: str, tool_call_id: str, name: str | None = None) -> Message
```

### Properties

```python
message.text  # str: Text content of the message
```

---

## Tool System

### @tool Decorator

Register a function as a tool.

```python
@tool
def my_function(param: str) -> str:
    """Description used by the LLM."""
    return "result"

@tool(name="custom_name", description="Custom description", timeout=30.0)
def another_function(x: int) -> str:
    """Original docstring."""
    return str(x)
```

### Tool Class

```python
tool.name       # str: Tool name
tool.spec       # ToolSpec: Tool specification
tool.function   # Callable: The wrapped function
tool.is_async   # bool: Whether function is async
tool.timeout    # float | None: Execution timeout

tool.execute_sync(arguments: dict) -> str
await tool.execute(arguments: dict) -> str
```

### ToolSpec

```python
spec.name         # str: Tool name
spec.description  # str: Tool description
spec.parameters   # dict: JSON Schema for parameters

spec.to_openai_schema() -> dict
spec.to_anthropic_schema() -> dict
spec.to_gemini_schema() -> dict
```

### ToolCall

```python
call.id         # str: Unique call ID
call.name       # str: Tool name
call.arguments  # dict: Call arguments

call.to_dict() -> dict
ToolCall.from_dict(data: dict) -> ToolCall
```

---

## Streaming

### StreamResponse

Synchronous streaming response.

```python
for event in stream:
    if event.type == "text_delta":
        print(event.text)

# After iteration:
stream.text        # str: Accumulated text
stream.tool_calls  # list[ToolCall]: Tool calls
stream.usage       # Usage | None: Token usage

stream.collect()   # Consume all events, return self
```

### AsyncStreamResponse

```python
async for event in stream:
    if event.type == "text_delta":
        print(event.text)
```

### StreamEvent

```python
event.type          # StreamEventType
event.text          # str | None (for text_delta)
event.tool_call     # ToolCall | None (for tool_call_start)
event.usage         # Usage | None (for usage, message_end)
event.error         # str | None (for error)
```

### Event Types

```python
"text_delta"       # Text chunk
"tool_call_start"  # Tool call started
"tool_call_delta"  # Tool call argument chunk
"tool_result"      # Tool execution result
"message_start"    # Message started
"message_end"      # Message complete
"usage"            # Usage information
"error"            # Error occurred
```

---

## Middleware

### Base Class

```python
class Middleware:
    def before(self, request: AgentRequest) -> AgentRequest:
        """Called before request. Can modify request."""
        return request

    def after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        """Called after response. Can modify response."""
        return response

    def on_error(self, request: AgentRequest, error: Exception) -> Exception | None:
        """Called on error. Return None to suppress."""
        return error
```

### Built-in Middleware

```python
from agent.middleware import LoggingMiddleware, MetricsMiddleware, RedactionMiddleware

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[LoggingMiddleware(), MetricsMiddleware()],
)
```

---

## Errors

All errors inherit from `AgentError`:

```python
from agent import (
    AgentError,           # Base error
    AuthenticationError,  # API key invalid
    ProviderError,        # Provider returned error
    RateLimitError,       # Rate limited
    TimeoutError,         # Request timed out
    ToolExecutionError,   # Tool failed
    SchemaValidationError, # Structured output invalid
    UnsupportedFeatureError, # Feature not supported
    RoutingError,         # All agents failed
)
```

### Error Attributes

```python
error.message    # str: Error message
error.raw        # Any: Raw error from provider

# ProviderError
error.provider     # str: Provider name
error.status_code  # int: HTTP status code

# RateLimitError
error.retry_after  # float: Seconds to wait

# ToolExecutionError
error.tool_name    # str: Which tool failed

# SchemaValidationError
error.schema       # The schema that failed
error.output       # The invalid output

# RoutingError
error.errors       # list[Exception]: All errors
```
