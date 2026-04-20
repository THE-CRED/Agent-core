# API Reference

Complete API documentation for the Agent library.

## Agent

The main class for interacting with LLM providers.

### Constructor

```python
Agent(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 2,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    tools: list[Tool] | None = None,
    middleware: list[Middleware] | None = None,
    default_system: str | None = None,
    **kwargs: Any,
)
```

### Methods

#### run()

Execute a synchronous request.

```python
def run(
    self,
    input: str | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### run_async()

Execute an asynchronous request.

```python
async def run_async(
    self,
    input: str | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### stream()

Execute a streaming request.

```python
def stream(
    self,
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
async def stream_async(
    self,
    input: str | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncStreamResponse
```

#### json()

Execute a request expecting structured JSON output.

```python
def json(
    self,
    input: str | None = None,
    *,
    schema: Type[BaseModel] | dict[str, Any],
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### json_async()

Execute an async request expecting structured JSON output.

```python
async def json_async(
    self,
    input: str | None = None,
    *,
    schema: Type[BaseModel] | dict[str, Any],
    messages: list[Message] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### session()

Create a new session for multi-turn conversation.

```python
def session(
    self,
    session_id: str | None = None,
    system: str | None = None,
) -> Session
```

#### with_config()

Create a new Agent with modified configuration.

```python
def with_config(self, **kwargs: Any) -> Agent
```

### Properties

- `provider: str` - Provider name
- `model: str` - Model name
- `tools: list[Tool]` - Registered tools

---

## Session

A session for multi-turn conversation.

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

#### run()

Send a message and get a response.

```python
def run(
    self,
    input: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### run_async()

Send a message asynchronously.

```python
async def run_async(
    self,
    input: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### stream()

Send a message and stream the response.

```python
def stream(
    self,
    input: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> StreamResponse
```

#### stream_async()

Send a message and stream the response asynchronously.

```python
async def stream_async(
    self,
    input: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncStreamResponse
```

#### json()

Send a message expecting structured JSON output.

```python
def json(
    self,
    input: str,
    *,
    schema: Type[BaseModel] | dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentResponse
```

#### history()

Get the full message history.

```python
def history(self) -> list[Message]
```

#### clear()

Clear the message history.

```python
def clear(self) -> None
```

#### fork()

Create a new session with a copy of the current history.

```python
def fork(self, session_id: str | None = None) -> Session
```

#### add_message()

Manually add a message to history.

```python
def add_message(self, message: Message) -> None
```

#### to_dict()

Serialize session state to a dictionary.

```python
def to_dict(self) -> dict[str, Any]
```

#### from_dict()

Deserialize session state from a dictionary.

```python
@classmethod
def from_dict(cls, data: dict[str, Any], agent: Agent) -> Session
```

### Properties

- `session_id: str` - Session identifier
- `system: str | None` - System prompt
- `messages: list[Message]` - Message history (read-only copy)

---

## AgentRouter

Routes requests across multiple agents with fallback support.

### Constructor

```python
AgentRouter(
    agents: list[Agent],
    strategy: RoutingStrategy | str = "fallback",
    custom_router: Callable[[AgentRequest, list[Agent]], RouteResult] | None = None,
)
```

### Methods

Same as `Agent`: `run()`, `run_async()`, `stream()`, `stream_async()`, `json()`

---

## tool decorator

Decorator to register a function as a tool.

```python
@tool
def my_tool(arg: str) -> str:
    """Tool description."""
    return result

@tool(name="custom_name", description="Custom description", timeout=30.0)
def another_tool(x: int) -> str:
    return str(x)
```

---

## Middleware

Base class for middleware.

### Methods

```python
def before(self, request: AgentRequest) -> AgentRequest:
    """Called before the request is sent."""
    return request

def after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
    """Called after receiving a response."""
    return response

def on_error(self, request: AgentRequest, error: Exception) -> Exception | None:
    """Called when an error occurs. Return None to suppress."""
    return error
```

---

## Types

### Message

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    
    @classmethod
    def system(cls, content: str) -> Message
    
    @classmethod
    def user(cls, content: str | list[ContentPart]) -> Message
    
    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[dict] | None = None) -> Message
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> Message
    
    @property
    def text(self) -> str
```

### ContentPart

```python
class ContentPart(BaseModel):
    type: Literal["text", "image", "image_url"]
    text: str | None = None
    image_url: str | None = None
    image_data: bytes | None = None
    media_type: str | None = None
    
    @classmethod
    def text_part(cls, text: str) -> ContentPart
    
    @classmethod
    def image_url_part(cls, url: str) -> ContentPart
    
    @classmethod
    def image_data_part(cls, data: bytes, media_type: str = "image/png") -> ContentPart
```

### AgentRequest

```python
class AgentRequest(BaseModel):
    input: str | None = None
    messages: list[Message] = []
    system: str | None = None
    tools: list[Any] = []
    output_schema: dict[str, Any] | None = None  # alias: schema
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    metadata: dict[str, Any] = {}
    session_id: str | None = None
    
    def to_messages(self) -> list[Message]
```

### AgentResponse

```python
class AgentResponse(BaseModel):
    text: str | None = None
    content: list[Any] = []
    output: Any = None
    provider: str = ""
    model: str = ""
    usage: Usage | None = None
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = []
    raw: Any = None
    latency_ms: float | None = None
    cost_estimate: float | None = None
    request_id: str | None = None
    
    @property
    def has_tool_calls(self) -> bool
    
    def to_dict(self) -> dict[str, Any]
```

### Usage

```python
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage
```

### ToolSpec

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    function: Any | None = None
    is_async: bool = False
    
    def to_openai_schema(self) -> dict[str, Any]
    def to_anthropic_schema(self) -> dict[str, Any]
    def to_gemini_schema(self) -> dict[str, Any]
```

### ToolCall

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = {}
    
    def to_dict(self) -> dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall
```

### ToolResult

```python
class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    
    def to_dict(self) -> dict[str, Any]
```

### StreamEvent

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
    
    @classmethod
    def text_delta(cls, text: str, raw: Any = None) -> StreamEvent
    
    @classmethod
    def tool_call_start(cls, tool_call: ToolCall, raw: Any = None) -> StreamEvent
    
    @classmethod
    def message_end(cls, usage: Usage | None = None, raw: Any = None) -> StreamEvent
    
    @classmethod
    def error_event(cls, error: str, raw: Any = None) -> StreamEvent
```

---

## Errors

```python
class AgentError(Exception):
    message: str
    raw: Any

class AuthenticationError(AgentError): ...

class ProviderError(AgentError):
    provider: str | None
    status_code: int | None

class RateLimitError(ProviderError):
    retry_after: float | None

class TimeoutError(AgentError):
    timeout: float | None

class ToolExecutionError(AgentError):
    tool_name: str | None

class SchemaValidationError(AgentError):
    schema: Any
    output: Any

class UnsupportedFeatureError(AgentError):
    feature: str | None
    provider: str | None

class RoutingError(AgentError):
    errors: list[Exception]
```
