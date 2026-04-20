# Error Handling

Agent provides a hierarchy of typed exceptions for handling errors gracefully across all providers.

## Error Hierarchy

```
AgentError (base)
├── AuthenticationError
├── ProviderError
│   └── RateLimitError
├── TimeoutError
├── ToolExecutionError
├── SchemaValidationError
├── UnsupportedFeatureError
└── RoutingError
```

## Basic Error Handling

```python
from agent import Agent
from agent.errors import (
    AgentError,
    AuthenticationError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ToolExecutionError,
    SchemaValidationError,
    UnsupportedFeatureError,
    RoutingError,
)

agent = Agent(provider="openai", model="gpt-4o")

try:
    response = agent.run("Hello!")
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except TimeoutError as e:
    print(f"Request timed out after {e.timeout}s")
except ProviderError as e:
    print(f"Provider error ({e.status_code}): {e}")
except AgentError as e:
    print(f"Agent error: {e}")
```

## Error Types

### AgentError

Base exception for all Agent errors.

```python
class AgentError(Exception):
    message: str   # Error message
    raw: Any       # Original exception (if any)
```

### AuthenticationError

Raised when API authentication fails.

```python
try:
    agent = Agent(provider="openai", model="gpt-4o", api_key="invalid")
    agent.run("Hello")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
    # Handle: check API key, refresh credentials
```

### ProviderError

Raised when the provider returns an error.

```python
class ProviderError(AgentError):
    provider: str | None    # Provider name
    status_code: int | None # HTTP status code

try:
    response = agent.run("Hello")
except ProviderError as e:
    print(f"Provider: {e.provider}")
    print(f"Status: {e.status_code}")
    print(f"Message: {e.message}")
    
    if e.status_code and 500 <= e.status_code < 600:
        print("Server error - try again later")
    elif e.status_code and 400 <= e.status_code < 500:
        print("Client error - check request")
```

### RateLimitError

Raised when rate limited by the provider.

```python
class RateLimitError(ProviderError):
    retry_after: float | None  # Seconds to wait

try:
    response = agent.run("Hello")
except RateLimitError as e:
    if e.retry_after:
        print(f"Waiting {e.retry_after}s...")
        time.sleep(e.retry_after)
        # Retry
    else:
        print("Rate limited, using exponential backoff")
        time.sleep(60)
```

### TimeoutError

Raised when a request times out.

```python
class TimeoutError(AgentError):
    timeout: float | None  # Configured timeout

try:
    response = agent.run("Complex query")
except TimeoutError as e:
    print(f"Timed out after {e.timeout}s")
    # Consider: increase timeout, simplify query
```

### ToolExecutionError

Raised when a tool fails to execute.

```python
class ToolExecutionError(AgentError):
    tool_name: str | None  # Name of failed tool

@tool
def risky_tool(data: str) -> str:
    raise ValueError("Something went wrong")

try:
    response = agent.run("Use the risky tool")
except ToolExecutionError as e:
    print(f"Tool '{e.tool_name}' failed: {e.message}")
```

### SchemaValidationError

Raised when structured output fails validation.

```python
class SchemaValidationError(AgentError):
    schema: Any  # The schema that failed
    output: Any  # The invalid output

try:
    response = agent.json("Get data", schema=MyModel)
except SchemaValidationError as e:
    print(f"Invalid output: {e.output}")
    print(f"Schema: {e.schema}")
    # Consider: retry, use different model, adjust schema
```

### UnsupportedFeatureError

Raised when a feature is not supported by the provider.

```python
class UnsupportedFeatureError(AgentError):
    feature: str | None   # Unsupported feature
    provider: str | None  # Provider name

agent = Agent(provider="deepseek", model="deepseek-chat")

try:
    response = agent.run(messages=[image_message])  # DeepSeek doesn't support vision
except UnsupportedFeatureError as e:
    print(f"Provider '{e.provider}' doesn't support '{e.feature}'")
    # Fall back to a different provider
```

### RoutingError

Raised when routing fails across all configured agents.

```python
class RoutingError(AgentError):
    errors: list[Exception]  # Errors from each agent

try:
    response = router.run("Hello")
except RoutingError as e:
    print(f"All {len(e.errors)} agents failed:")
    for i, error in enumerate(e.errors):
        print(f"  Agent {i}: {error}")
```

## Retry Strategies

### Automatic Retries

Agent automatically retries on transient errors:

```python
agent = Agent(
    provider="openai",
    model="gpt-4o",
    max_retries=3,  # Retry up to 3 times
)

# Automatically retries on:
# - Rate limit errors (with backoff)
# - 5xx server errors
# - Connection timeouts
```

### Custom Retry Logic

```python
import time
from agent.errors import RateLimitError, ProviderError

def run_with_retry(agent, prompt, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return agent.run(prompt)
        except RateLimitError as e:
            if attempt < max_attempts - 1:
                wait = e.retry_after or (2 ** attempt)
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except ProviderError as e:
            if e.status_code and e.status_code >= 500:
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            else:
                raise  # Don't retry client errors
```

## Error Handling Patterns

### Graceful Degradation

```python
def get_response(prompt: str) -> str:
    """Get response with graceful degradation."""
    
    # Try primary agent
    try:
        return primary_agent.run(prompt).text
    except AgentError:
        pass
    
    # Try backup agent
    try:
        return backup_agent.run(prompt).text
    except AgentError:
        pass
    
    # Return fallback
    return "I'm sorry, I'm having trouble responding right now."
```

### Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = "half-open"
            else:
                raise CircuitOpenError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except AgentError as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "open"
            raise

breaker = CircuitBreaker()

try:
    response = breaker.call(agent.run, "Hello")
except CircuitOpenError:
    # Use fallback
    pass
```

### Logging Errors

```python
import logging

logger = logging.getLogger(__name__)

class ErrorLoggingMiddleware(Middleware):
    def on_error(self, request, error):
        logger.error(
            "Agent error",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "provider": getattr(error, "provider", None),
                "status_code": getattr(error, "status_code", None),
                "input_preview": (request.input or "")[:100],
            }
        )
        return error

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[ErrorLoggingMiddleware()],
)
```

## Async Error Handling

```python
import asyncio

async def main():
    agent = Agent(provider="openai", model="gpt-4o")
    
    try:
        response = await agent.run_async("Hello")
    except AuthenticationError:
        print("Check your API key")
    except RateLimitError as e:
        await asyncio.sleep(e.retry_after or 60)
        response = await agent.run_async("Hello")  # Retry
    except AgentError as e:
        print(f"Error: {e}")
```

## Streaming Error Handling

```python
try:
    for event in agent.stream("Hello"):
        if event.type == "error":
            print(f"Stream error: {event.error}")
            break
        if event.type == "text_delta":
            print(event.text, end="")
except AgentError as e:
    print(f"Connection error: {e}")
```

## Best Practices

### 1. Catch Specific Exceptions

```python
# Good - handle specific cases
try:
    response = agent.run(prompt)
except RateLimitError:
    handle_rate_limit()
except AuthenticationError:
    handle_auth_error()
except AgentError:
    handle_generic_error()

# Bad - catch everything
try:
    response = agent.run(prompt)
except Exception:
    pass  # Unknown what went wrong
```

### 2. Include Context in Logs

```python
except AgentError as e:
    logger.error(
        f"Failed to process: {e}",
        extra={
            "prompt_length": len(prompt),
            "provider": agent.provider,
            "model": agent.model,
        }
    )
```

### 3. Don't Expose Raw Errors to Users

```python
# Good - user-friendly message
except AgentError:
    return "I'm having trouble right now. Please try again."

# Bad - expose internal details
except AgentError as e:
    return f"OpenAI API error: {e.raw}"
```

## Next Steps

- [Middleware](middleware.md) - Error handling in middleware
- [Routing](routing.md) - Handle routing failures
- [Configuration](configuration.md) - Configure retry behavior
