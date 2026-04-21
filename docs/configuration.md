# Configuration

Agent provides flexible configuration options for customizing behavior, authentication, and provider settings.

## Agent Configuration

### Basic Configuration


```python
from agent import Agent

agent = Agent(
    # Required
    provider="openai",
    model="gpt-4o",
    
    # Authentication
    api_key="sk-...",              # Or use environment variable
    base_url="https://custom.api", # Custom endpoint
    
    # Network settings
    timeout=120.0,                 # Request timeout (seconds)
    max_retries=3,                 # Retry on transient errors
    
    # Generation defaults
    temperature=0.7,               # Sampling temperature (0-2)
    max_tokens=4096,               # Max tokens to generate
    top_p=0.9,                     # Nucleus sampling
    
    # System prompt
    default_system="You are a helpful assistant.",
)
```

### Environment Variables

Agent automatically reads API keys from environment variables:

```bash
# Provider-specific keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

```python
# No need to pass api_key if env var is set
agent = Agent(provider="openai", model="gpt-4o")
```

### Configuration Overrides

Create new agents with modified settings:

```python
# Base agent
agent = Agent(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
)

# Create variant with different settings
creative_agent = agent.with_config(temperature=1.2)
precise_agent = agent.with_config(temperature=0.1)

# Per-request overrides
response = agent.run(
    "Be creative",
    temperature=1.5,  # Override for this request only
    max_tokens=2000,
)
```

## Model Aliases

Agent supports convenient model aliases:

```python
# These are equivalent
agent = Agent(provider="openai", model="gpt-4o")
agent = Agent(provider="openai", model="gpt-4o")  # Direct name

# Anthropic aliases
agent = Agent(provider="anthropic", model="claude")        # -> claude-sonnet-4-20250514
agent = Agent(provider="anthropic", model="claude-sonnet") # -> claude-sonnet-4-20250514
agent = Agent(provider="anthropic", model="claude-opus")   # -> claude-opus-4-20250514
agent = Agent(provider="anthropic", model="claude-haiku")  # -> claude-3-5-haiku-20241022

# Gemini aliases
agent = Agent(provider="gemini", model="gemini-pro")   # -> gemini-1.5-pro
agent = Agent(provider="gemini", model="gemini-flash") # -> gemini-1.5-flash
```

## Provider Configuration

### Custom Base URLs

```python
# Azure OpenAI
agent = Agent(
    provider="openai",
    model="gpt-4",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    api_key="your-azure-key",
)

# Local LLM (Ollama, vLLM, etc.)
agent = Agent(
    provider="openai",
    model="llama2",
    base_url="http://localhost:11434/v1",
    api_key="not-needed",
)

# Custom proxy
agent = Agent(
    provider="anthropic",
    model="claude-sonnet",
    base_url="https://your-proxy.com/anthropic",
)
```

### Provider-Specific Options

Pass extra options to providers:

```python
agent = Agent(
    provider="openai",
    model="gpt-4o",
    # Extra options passed to provider
    organization="org-xxx",
    seed=42,
)
```

## Retry Configuration

### Basic Retry Settings

```python
agent = Agent(
    provider="openai",
    model="gpt-4o",
    max_retries=5,  # Retry up to 5 times
)
```

### Advanced Retry Configuration

```python
from agent.types import RetryConfig

config = RetryConfig(
    max_retries=5,
    initial_delay=1.0,      # Start with 1 second delay
    max_delay=60.0,         # Cap delay at 60 seconds
    exponential_base=2.0,   # Double delay each retry
    jitter=True,            # Add randomness to avoid thundering herd
)

# Use with execution runtime (advanced)
from agent.execution.retries import RetryHandler
handler = RetryHandler(config)
```

### Retryable Errors

By default, these errors trigger retries:
- `RateLimitError` - Always retried with backoff
- `ProviderError` with 5xx status codes
- `TimeoutError`
- `ConnectionError`

## Tool Loop Configuration

```python
from agent.types import ToolLoopConfig

config = ToolLoopConfig(
    max_iterations=10,              # Max tool calling rounds
    max_tool_calls_per_iteration=20, # Max tools per round
    timeout_per_tool=30.0,          # Per-tool timeout
    parallel_tool_execution=True,   # Run tools in parallel
    stop_on_error=False,            # Continue on tool errors
)
```

## Middleware Configuration

```python
from agent import Agent
from agent.middleware import LoggingMiddleware, MetricsMiddleware

metrics = MetricsMiddleware()

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[
        LoggingMiddleware(),
        metrics,
    ],
)

# Access metrics
print(metrics.stats())
```

## Router Configuration

```python
from agent import Agent, AgentRouter

router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="openai", model="gpt-4o"),
    ],
    strategy="fallback",  # or "round_robin", "fastest", "cheapest"
)
```

## Configuration Best Practices

### 1. Use Environment Variables for Secrets

```python
# Good - uses env var
agent = Agent(provider="openai", model="gpt-4o")

# Avoid - hardcoded secrets
agent = Agent(provider="openai", model="gpt-4o", api_key="sk-secret")
```

### 2. Set Reasonable Timeouts

```python
# For simple queries
agent = Agent(provider="openai", model="gpt-4o", timeout=30.0)

# For complex operations
agent = Agent(provider="openai", model="gpt-4o", timeout=300.0)
```

### 3. Configure Retries Appropriately

```python
# Production - more retries
agent = Agent(provider="openai", model="gpt-4o", max_retries=5)

# Development - faster failures
agent = Agent(provider="openai", model="gpt-4o", max_retries=1)
```

### 4. Use Temperature Wisely

```python
# Deterministic tasks (classification, extraction)
agent = Agent(provider="openai", model="gpt-4o", temperature=0.0)

# Creative tasks (writing, brainstorming)
agent = Agent(provider="openai", model="gpt-4o", temperature=1.0)

# Balanced (general assistant)
agent = Agent(provider="openai", model="gpt-4o", temperature=0.7)
```

## Configuration Reference

### AgentConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | Required | Provider name |
| `model` | `str` | Required | Model name or alias |
| `api_key` | `str \| None` | None | API key (or use env var) |
| `base_url` | `str \| None` | None | Custom API endpoint |
| `timeout` | `float` | 120.0 | Request timeout (seconds) |
| `max_retries` | `int` | 2 | Max retry attempts |
| `temperature` | `float \| None` | None | Sampling temperature |
| `max_tokens` | `int \| None` | None | Max tokens to generate |
| `top_p` | `float \| None` | None | Nucleus sampling |
| `default_system` | `str \| None` | None | Default system prompt |

### Environment Variables

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GOOGLE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

### Model Pricing (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-sonnet | $3.00 | $15.00 |
| claude-opus | $15.00 | $75.00 |
| claude-haiku | $0.25 | $1.25 |
| gemini-1.5-pro | $1.25 | $5.00 |
| gemini-1.5-flash | $0.075 | $0.30 |

## Next Steps

- [Providers](providers.md) - Provider-specific configuration
- [Middleware](middleware.md) - Configure middleware
- [Error Handling](error-handling.md) - Configure retry behavior
