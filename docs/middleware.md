# Middleware

Middleware provides hooks into the request/response lifecycle for logging, tracing, metrics, and custom policies.

## Basic Usage

```python
from agent import Agent, Middleware

class LoggingMiddleware(Middleware):
    def before(self, request):
        print(f"Request: {request.input[:50]}...")
        return request
    
    def after(self, request, response):
        print(f"Response: {response.text[:50]}...")
        return response
    
    def on_error(self, request, error):
        print(f"Error: {error}")
        return error  # Re-raise the error

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[LoggingMiddleware()],
)
```

## Middleware Hooks

### before(request) -> AgentRequest

Called before the request is sent to the provider. Can modify or replace the request.

```python
class SystemPromptMiddleware(Middleware):
    def __init__(self, prefix: str):
        self.prefix = prefix
    
    def before(self, request):
        if request.system:
            request.system = f"{self.prefix}\n\n{request.system}"
        else:
            request.system = self.prefix
        return request
```

### after(request, response) -> AgentResponse

Called after receiving a response. Can modify or replace the response.

```python
class MetadataMiddleware(Middleware):
    def after(self, request, response):
        response.metadata["processed_at"] = datetime.now().isoformat()
        response.metadata["input_length"] = len(request.input or "")
        return response
```

### on_error(request, error) -> Exception | None

Called when an error occurs. Return `None` to suppress the error.

```python
class ErrorRecoveryMiddleware(Middleware):
    def on_error(self, request, error):
        if isinstance(error, RateLimitError):
            print(f"Rate limited, will retry...")
            return error  # Let retry handler deal with it
        
        if isinstance(error, ProviderError) and error.status_code == 503:
            print("Service unavailable, suppressing error")
            return None  # Suppress the error
        
        return error  # Re-raise other errors
```

## Built-in Middleware

### LoggingMiddleware

```python
from agent.middleware import LoggingMiddleware

# With default print
agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[LoggingMiddleware()],
)

# With custom logger
import logging
logger = logging.getLogger(__name__)

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[LoggingMiddleware(log_fn=logger.info)],
)
```

### MetricsMiddleware

```python
from agent.middleware import MetricsMiddleware

metrics = MetricsMiddleware()

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[metrics],
)

# Make some requests
agent.run("Hello")
agent.run("World")

# Get metrics
stats = metrics.stats()
print(f"Requests: {stats['request_count']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Errors: {stats['error_count']}")
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
```

### RedactionMiddleware

```python
from agent.middleware import RedactionMiddleware

# Redact sensitive patterns
redactor = RedactionMiddleware(patterns=[
    r"sk-[a-zA-Z0-9]{20,}",           # API keys
    r"\b\d{16}\b",                     # Credit cards
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Emails
])

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[redactor],
)
```

### RetryPolicyMiddleware

```python
from agent.middleware import RetryPolicyMiddleware
from agent.errors import RateLimitError, ProviderError

retry_policy = RetryPolicyMiddleware(
    max_retries=5,
    retryable_errors=(RateLimitError, ProviderError),
)

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[retry_policy],
)
```

## Chaining Middleware

Middleware executes in order for `before` hooks and reverse order for `after` hooks:

```python
class TimingMiddleware(Middleware):
    def before(self, request):
        request.metadata["start_time"] = time.time()
        return request
    
    def after(self, request, response):
        elapsed = time.time() - request.metadata["start_time"]
        print(f"Total time: {elapsed:.2f}s")
        return response

class CachingMiddleware(Middleware):
    def __init__(self):
        self.cache = {}
    
    def before(self, request):
        key = hash(request.input)
        if key in self.cache:
            # Return cached response (skip provider call)
            raise CacheHit(self.cache[key])
        return request
    
    def after(self, request, response):
        key = hash(request.input)
        self.cache[key] = response
        return response

# Order: TimingMiddleware.before -> CachingMiddleware.before -> Provider
#        CachingMiddleware.after -> TimingMiddleware.after
agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[TimingMiddleware(), CachingMiddleware()],
)
```

## Custom Middleware Examples

### Rate Limiting

```python
import time
from collections import deque

class RateLimiter(Middleware):
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests = deque()
    
    def before(self, request):
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()
        
        # Check rate limit
        if len(self.requests) >= self.rpm:
            wait_time = 60 - (now - self.requests[0])
            time.sleep(wait_time)
        
        self.requests.append(now)
        return request
```

### Content Filtering

```python
class ContentFilter(Middleware):
    def __init__(self, blocked_words: list[str]):
        self.blocked = set(w.lower() for w in blocked_words)
    
    def before(self, request):
        if request.input:
            words = set(request.input.lower().split())
            if words & self.blocked:
                raise ValueError("Request contains blocked content")
        return request
    
    def after(self, request, response):
        if response.text:
            words = set(response.text.lower().split())
            if words & self.blocked:
                response.text = "[Content filtered]"
        return response
```

### Request Tracing

```python
import uuid

class TracingMiddleware(Middleware):
    def before(self, request):
        trace_id = str(uuid.uuid4())
        request.metadata["trace_id"] = trace_id
        print(f"[{trace_id}] Starting request")
        return request
    
    def after(self, request, response):
        trace_id = request.metadata.get("trace_id", "unknown")
        print(f"[{trace_id}] Completed in {response.latency_ms}ms")
        return response
    
    def on_error(self, request, error):
        trace_id = request.metadata.get("trace_id", "unknown")
        print(f"[{trace_id}] Error: {error}")
        return error
```

### Cost Tracking

```python
class CostTracker(Middleware):
    def __init__(self):
        self.total_cost = 0.0
        self.budget = float('inf')
    
    def set_budget(self, amount: float):
        self.budget = amount
    
    def before(self, request):
        if self.total_cost >= self.budget:
            raise BudgetExceededError(f"Budget of ${self.budget} exceeded")
        return request
    
    def after(self, request, response):
        if response.cost_estimate:
            self.total_cost += response.cost_estimate
            print(f"Cost: ${response.cost_estimate:.4f} (Total: ${self.total_cost:.4f})")
        return response
```

### Response Validation

```python
class ResponseValidator(Middleware):
    def __init__(self, min_length: int = 10):
        self.min_length = min_length
    
    def after(self, request, response):
        if response.text and len(response.text) < self.min_length:
            raise ValueError(f"Response too short: {len(response.text)} chars")
        return response
```

## Middleware Chain

Access the middleware chain programmatically:

```python
from agent.middleware import MiddlewareChain

chain = MiddlewareChain([
    LoggingMiddleware(),
    MetricsMiddleware(),
])

# Add more middleware
chain.add(RateLimiter())

# Use with agent
agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=chain.middlewares,
)
```

## Best Practices

### 1. Keep Middleware Focused

```python
# Good - single responsibility
class LoggingMiddleware(Middleware):
    def before(self, request):
        print(f"Request: {request.input}")
        return request

class MetricsMiddleware(Middleware):
    def after(self, request, response):
        self.record_latency(response.latency_ms)
        return response

# Bad - too many responsibilities
class KitchenSinkMiddleware(Middleware):
    def before(self, request):
        print(f"Request: {request.input}")
        self.check_rate_limit()
        self.validate_input()
        self.add_tracing()
        return request
```

### 2. Don't Swallow Errors Silently

```python
# Good - log before suppressing
def on_error(self, request, error):
    if should_suppress(error):
        logger.warning(f"Suppressing error: {error}")
        return None
    return error

# Bad - silent suppression
def on_error(self, request, error):
    return None  # What happened?
```

### 3. Be Careful with Request Modification

```python
# Good - copy before modifying
def before(self, request):
    modified = AgentRequest(
        input=request.input,
        system=f"PREFIX: {request.system}",
        # ... copy other fields
    )
    return modified

# Risky - mutating shared state
def before(self, request):
    request.system = f"PREFIX: {request.system}"
    return request
```

## Next Steps

- [Error Handling](error-handling.md) - Handle errors in middleware
- [Routing](routing.md) - Apply middleware to routers
- [Custom Providers](custom-providers.md) - Provider-specific middleware
