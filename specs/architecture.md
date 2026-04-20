# Architecture Specification

This document describes the overall architecture of the Agent library.

## Design Principles

1. **Provider Agnostic**: One interface, many backends
2. **Type Safe**: Pydantic models throughout
3. **Composable**: Middleware, tools, and routing combine freely
4. **Async First**: Native async with sync wrappers
5. **Minimal Dependencies**: Core has few dependencies; providers are optional

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│                                                                  │
│   agent = Agent(provider="openai", model="gpt-4o")              │
│   response = agent.run("Hello!")                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Agent                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Config     │  │    Tools     │  │  Middleware  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Execution Runtime                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Retry Handler│  │  Tool Loop   │  │Struct Output │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Provider                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    OpenAI    │  │  Anthropic   │  │    Gemini    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM API                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### Agent

The main entry point. Responsibilities:
- Configuration management
- Tool registration
- Middleware chain
- Session creation
- Request building

```python
class Agent:
    config: AgentConfig
    _provider: BaseProvider
    _tools: list[Tool]
    _middleware: MiddlewareChain
    _runtime: ExecutionRuntime
```

### ExecutionRuntime

Orchestrates request execution. Responsibilities:
- Middleware execution (before/after)
- Retry handling
- Tool loop orchestration
- Structured output parsing
- Cost estimation

```python
class ExecutionRuntime:
    provider: BaseProvider
    config: AgentConfig
    tools: list[Tool]
    middleware: MiddlewareChain
    retry_handler: RetryHandler
    tool_loop: ToolLoop | None
```

### BaseProvider

Abstract interface for LLM providers. Each provider must implement:
- `run(request) -> AgentResponse`
- `run_async(request) -> AgentResponse`
- `stream(request) -> Iterator[StreamEvent]`
- `stream_async(request) -> AsyncIterator[StreamEvent]`

```python
class BaseProvider(ABC):
    name: str
    capabilities: ProviderCapabilities
    
    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResponse: ...
    
    @abstractmethod
    async def run_async(self, request: AgentRequest) -> AgentResponse: ...
    
    @abstractmethod
    def stream(self, request: AgentRequest) -> Iterator[StreamEvent]: ...
    
    @abstractmethod
    async def stream_async(self, request: AgentRequest) -> AsyncIterator[StreamEvent]: ...
```

### Session

Manages conversation state. Responsibilities:
- Message history
- History serialization
- Session forking

```python
class Session:
    _agent: Agent
    _session_id: str
    _system: str | None
    _messages: list[Message]
```

### AgentRouter

Multi-agent routing. Responsibilities:
- Strategy execution (fallback, round-robin, etc.)
- Error aggregation
- Capability-based routing

```python
class AgentRouter:
    agents: list[Agent]
    strategy: RoutingStrategy
    custom_router: Callable | None
```

## Data Flow

### Simple Request

```
User Input
    │
    ▼
Agent.run(input)
    │
    ▼
Build AgentRequest
    │
    ▼
MiddlewareChain.run_before(request)
    │
    ▼
RetryHandler.execute(
    │   provider.run(request)
    │)
    │
    ▼
MiddlewareChain.run_after(request, response)
    │
    ▼
AgentResponse
```

### Tool-Calling Request

```
User Input
    │
    ▼
Agent.run(input)
    │
    ▼
Build AgentRequest (with tools)
    │
    ▼
ToolLoop.run_loop(request, provider.run)
    │
    ├─► Provider.run(request)
    │       │
    │       ▼
    │   Response with tool_calls
    │       │
    │       ▼
    │   ToolLoop.execute_tool_calls(tool_calls)
    │       │
    │       ▼
    │   Build tool result messages
    │       │
    │       ▼
    └── Provider.run(request + tool_results)
            │
            ▼
        Final Response (no tool_calls)
            │
            ▼
AgentResponse
```

### Streaming Request

```
User Input
    │
    ▼
Agent.stream(input)
    │
    ▼
Provider.stream(request)
    │
    ▼
StreamResponse (wraps Iterator[StreamEvent])
    │
    ├─► StreamEvent(type="text_delta", text="Hello")
    ├─► StreamEvent(type="text_delta", text=" world")
    ├─► StreamEvent(type="message_end", usage=Usage(...))
    │
    ▼
Accumulated: text="Hello world", usage=Usage(...)
```

## Type System

All data is represented as Pydantic models:

```
agent/types/
├── messages.py    ContentPart, Message, AgentRequest
├── response.py    Usage, AgentResponse
├── tools.py       ToolSpec, ToolCall, ToolResult
├── stream.py      StreamEvent
├── config.py      AgentConfig, ProviderCapabilities, RetryConfig, ToolLoopConfig
└── router.py      RoutingStrategy, RouteResult
```

Benefits:
- Automatic validation
- JSON serialization
- Type checking
- Self-documenting

## Provider Architecture

Each provider converts between normalized types and provider-specific formats:

```
AgentRequest                    Provider Format
─────────────                   ───────────────
input           ──────────►     messages[-1].content
messages        ──────────►     messages[:-1]
system          ──────────►     system (Anthropic) / messages[0] (OpenAI)
tools           ──────────►     tools/functions
temperature     ──────────►     temperature
max_tokens      ──────────►     max_tokens
```

```
Provider Response              AgentResponse
─────────────────              ─────────────
content         ──────────►    text
tool_calls      ──────────►    tool_calls (normalized ToolCall)
usage           ──────────►    usage (normalized Usage)
finish_reason   ──────────►    stop_reason
raw response    ──────────►    raw
```

## Middleware System

Middleware forms a chain with before/after hooks:

```
Request Flow:
    │
    ├─► Middleware1.before(request)
    ├─► Middleware2.before(request)
    ├─► Middleware3.before(request)
    │
    ▼
Provider.run(request)
    │
    ▼
    ├─► Middleware3.after(request, response)
    ├─► Middleware2.after(request, response)
    ├─► Middleware1.after(request, response)
    │
    ▼
Response
```

Error handling:
```
    ├─► Middleware1.on_error(request, error)
    ├─► Middleware2.on_error(request, error)
    │       │
    │       └── Returns None (suppress error)
    │
    ▼
Error suppressed, return empty response
```

## Error Hierarchy

```
AgentError
├── AuthenticationError
├── ProviderError
│   └── RateLimitError
├── TimeoutError
├── ToolExecutionError
├── SchemaValidationError
├── UnsupportedFeatureError
└── RoutingError
```

## Extension Points

1. **Custom Providers**: Implement `BaseProvider`
2. **Custom Middleware**: Extend `Middleware`
3. **Custom Tools**: Use `@tool` decorator
4. **Custom Routing**: Pass `custom_router` to `AgentRouter`
5. **Custom Session Stores**: Implement `SessionStore`

## Thread Safety

- `Agent` instances are **not** thread-safe for concurrent requests
- Create separate `Agent` instances per thread, or
- Use async methods with proper concurrency control
- `Session` instances should not be shared across threads

## Async Model

- All I/O operations have async variants
- Sync methods run async code in executor
- Streaming is natively async with sync wrappers
- Tool execution supports both sync and async tools
