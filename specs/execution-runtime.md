# Execution Runtime Specification

This document specifies the execution runtime that processes requests.

## Overview

The `ExecutionRuntime` orchestrates the complete lifecycle of a request:

1. Middleware preprocessing
2. Request preparation (tools, schema)
3. Retry handling
4. Tool loop orchestration
5. Structured output parsing
6. Middleware postprocessing
7. Cost estimation

## Class Definition

```python
class ExecutionRuntime:
    def __init__(
        self,
        provider: BaseProvider,
        config: AgentConfig,
        tools: list[Tool] | None = None,
        middleware: MiddlewareChain | None = None,
        retry_config: RetryConfig | None = None,
        tool_loop_config: ToolLoopConfig | None = None,
    ):
        self.provider = provider
        self.config = config
        self.tools = tools or []
        self.middleware = middleware or MiddlewareChain()
        self.retry_handler = RetryHandler(retry_config or RetryConfig(
            max_retries=config.max_retries
        ))
        self.tool_loop = ToolLoop(self.tools, tool_loop_config) if self.tools else None
```

## Request Flow

### run() - Synchronous Execution

```python
def run(
    self,
    request: AgentRequest,
    schema: Type[BaseModel] | dict[str, Any] | None = None,
) -> AgentResponse:
```

**Flow:**

```
1. Start timer
       │
       ▼
2. Middleware: run_before(request)
       │
       ▼
3. Structured Output Setup (if schema)
   ├── Native support: Add schema to request
   └── Prompt-based: Add instructions to system
       │
       ▼
4. Tool Setup (if tools)
   └── Add tool specs to request
       │
       ▼
5. Execute (with retries)
   ├── If tools: ToolLoop.run_loop(request, provider.run)
   └── Else: provider.run(request)
       │
       ▼
6. Parse Structured Output (if schema and response.text)
       │
       ▼
7. Calculate latency_ms and cost_estimate
       │
       ▼
8. Middleware: run_after(request, response)
       │
       ▼
9. Return AgentResponse
```

### run_async() - Asynchronous Execution

Same flow as `run()` but uses:
- `await provider.run_async(request)`
- `await tool_loop.run_loop_async(request, provider.run_async)`
- `await retry_handler.execute_async(...)`

### stream() - Streaming Execution

```python
def stream(self, request: AgentRequest) -> StreamResponse:
```

**Flow:**

```
1. Middleware: run_before(request)
       │
       ▼
2. Check provider.supports_streaming()
       │
       ▼
3. Tool Setup (if tools)
       │
       ▼
4. Get stream iterator: provider.stream(request)
       │
       ▼
5. Return StreamResponse(events, provider, model)
```

Note: Streaming does **not** support:
- Automatic retries (fails immediately)
- Tool loops (tools stream but don't auto-continue)
- Structured output parsing

### stream_async() - Async Streaming

Same as `stream()` but returns `AsyncStreamResponse` with async iterator.

## Retry Handler

### RetryConfig

```python
class RetryConfig(BaseModel):
    max_retries: int = 2
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple[type[Exception], ...]
```

### Retry Logic

```python
def should_retry(error: Exception, attempt: int) -> bool:
    if attempt >= max_retries:
        return False
    
    if isinstance(error, RateLimitError):
        return True
    
    if isinstance(error, ProviderError):
        if 500 <= error.status_code < 600:
            return True  # Server error
        if 400 <= error.status_code < 500:
            return False  # Client error
    
    return isinstance(error, retryable_errors)
```

### Delay Calculation

```python
def get_delay(attempt: int, error: Exception | None) -> float:
    # Use retry-after header if available
    if isinstance(error, RateLimitError) and error.retry_after:
        return min(error.retry_after, max_delay)
    
    # Exponential backoff
    delay = initial_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)
    
    # Add jitter
    if jitter:
        delay = delay * (0.5 + random.random())
    
    return delay
```

## Tool Loop

### ToolLoopConfig

```python
class ToolLoopConfig(BaseModel):
    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 20
    timeout_per_tool: float = 30.0
    parallel_tool_execution: bool = True
    stop_on_error: bool = False
```

### Tool Loop Flow

```
Initial Request
      │
      ▼
┌─────────────────────────────────────────────┐
│ For iteration in range(max_iterations):     │
│      │                                      │
│      ▼                                      │
│ Build request with current messages         │
│      │                                      │
│      ▼                                      │
│ response = run_fn(request)                  │
│      │                                      │
│      ▼                                      │
│ if not response.has_tool_calls:             │
│    return response  ◄──────────────────────┼── Done
│      │                                      │
│      ▼                                      │
│ results = execute_tool_calls(tool_calls)    │
│      │                                      │
│      ▼                                      │
│ messages += build_tool_messages(response,   │
│                                  results)   │
│      │                                      │
└──────┴──────────────────────────────────────┘
      │
      ▼
Return last response (max iterations reached)
```

### Tool Execution

**Sync execution:**
```python
for call in tool_calls[:max_tool_calls_per_iteration]:
    tool = registry.get(call.name)
    if tool is None:
        results.append(ToolResult(is_error=True, content="Unknown tool"))
        continue
    
    try:
        content = tool.execute_sync(call.arguments)
        results.append(ToolResult(content=content))
    except Exception as e:
        if stop_on_error:
            raise ToolExecutionError(...)
        results.append(ToolResult(is_error=True, content=f"Error: {e}"))
```

**Async execution (parallel):**
```python
if parallel_tool_execution:
    tasks = [execute_single_tool_async(call) for call in calls]
    results = await asyncio.gather(*tasks)
else:
    results = [await execute_single_tool_async(call) for call in calls]
```

## Structured Output Handler

### Schema Handling

```python
class StructuredOutputHandler:
    def __init__(self, schema: Type[BaseModel] | dict, strict: bool = True, repair_attempts: int = 1):
        self.schema = Schema(schema, strict=strict, repair_attempts=repair_attempts)
```

### Native vs Prompt-Based

**Native schema (OpenAI):**
```python
if provider.supports_native_schema():
    request.schema = handler.get_json_schema()
```

**Prompt-based (Anthropic):**
```python
else:
    schema_prompt = handler.get_system_prompt_addition()
    request.system = f"{request.system}\n\n{schema_prompt}"
```

### Response Parsing

```python
def parse_response(text: str) -> Any:
    for attempt in range(repair_attempts + 1):
        try:
            data = extract_json(text)
            return schema.validate(data)
        except (ValueError, SchemaValidationError) as e:
            if attempt < repair_attempts:
                text = repair_json(text, e)
    
    raise SchemaValidationError(...)
```

## Error Handling

### Middleware Error Hook

```python
try:
    response = self._execute(request)
except Exception as e:
    handled_error = self.middleware.run_on_error(request, e)
    if handled_error is None:
        # Error suppressed - return empty response
        return AgentResponse(text="", provider=self.provider.name)
    raise handled_error
```

### Error Flow

```
Exception raised
      │
      ▼
Middleware1.on_error(request, error)
      │
      ├── Returns modified error → Continue chain
      └── Returns None → Suppress error
      │
      ▼
Middleware2.on_error(request, error)
      │
      ├── Returns error → Raise
      └── Returns None → Return empty response
```

## Cost Estimation

```python
if response.usage:
    response.cost_estimate = estimate_cost(
        self.config.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    model = resolve_model(model)
    pricing = PRICING.get(model)
    if not pricing:
        return None
    
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

## Latency Tracking

```python
start_time = time.time()
# ... execution ...
response.latency_ms = (time.time() - start_time) * 1000
```
