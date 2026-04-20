# Quick Start Guide

Get up and running with Agent in 5 minutes.

## Basic Usage

### 1. Create an Agent

```python
from agent import Agent

# Create an agent with a provider and model
agent = Agent(
    provider="openai",
    model="gpt-4o",
)
```

### 2. Generate Text

```python
response = agent.run("Explain quantum computing in one paragraph")
print(response.text)
```

### 3. Access Response Metadata

```python
response = agent.run("Hello!")

print(f"Text: {response.text}")
print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Latency: {response.latency_ms}ms")
print(f"Cost estimate: ${response.cost_estimate:.4f}")
```

## Async Usage

Agent supports async/await for non-blocking operations:

```python
import asyncio
from agent import Agent

async def main():
    agent = Agent(provider="anthropic", model="claude-sonnet")
    response = await agent.run_async("Write a haiku about Python")
    print(response.text)

asyncio.run(main())
```

## Streaming

Stream responses for real-time output:

```python
from agent import Agent

agent = Agent(provider="anthropic", model="claude-sonnet")

for event in agent.stream("Write a short story"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
print()  # Final newline
```

Async streaming:

```python
async for event in await agent.stream_async("Write a poem"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

## Structured Outputs

Get type-safe responses using Pydantic models:

```python
from pydantic import BaseModel
from agent import Agent

class MovieReview(BaseModel):
    title: str
    rating: float
    pros: list[str]
    cons: list[str]
    summary: str

agent = Agent(provider="openai", model="gpt-4o")
response = agent.json(
    "Review the movie 'Inception'",
    schema=MovieReview
)

review = response.output  # Typed as MovieReview
print(f"{review.title}: {review.rating}/10")
print(f"Pros: {', '.join(review.pros)}")
```

## Tool Calling

Register Python functions as tools:

```python
from agent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Your implementation here
    return f"Weather in {city}: 72F, Sunny"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

agent = Agent(
    provider="anthropic",
    model="claude-sonnet",
    tools=[get_weather, calculate],
)

response = agent.run("What's the weather in Tokyo and what's 25 * 4?")
print(response.text)
```

## Sessions (Multi-turn Conversations)

Maintain conversation history:

```python
from agent import Agent

agent = Agent(provider="openai", model="gpt-4o")

# Create a session
session = agent.session(system="You are a helpful math tutor.")

# Multi-turn conversation
session.run("What is calculus?")
session.run("Can you give me a simple example?")
response = session.run("How does that relate to physics?")

print(response.text)

# Check history
print(f"Messages in history: {len(session.messages)}")
```

## Configuration Options

```python
from agent import Agent

agent = Agent(
    # Required
    provider="openai",
    model="gpt-4o",
    
    # Authentication
    api_key="sk-...",  # Or use OPENAI_API_KEY env var
    base_url="https://custom-endpoint.com/v1",  # Custom endpoint
    
    # Request settings
    timeout=120.0,      # Request timeout in seconds
    max_retries=3,      # Retry on transient errors
    
    # Generation settings
    temperature=0.7,    # Sampling temperature
    max_tokens=1000,    # Maximum tokens to generate
    top_p=0.9,          # Nucleus sampling
    
    # Default prompts
    default_system="You are a helpful assistant.",
)
```

## Error Handling

```python
from agent import Agent
from agent.errors import (
    AgentError,
    AuthenticationError,
    RateLimitError,
    ProviderError,
)

agent = Agent(provider="openai", model="gpt-4o")

try:
    response = agent.run("Hello!")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ProviderError as e:
    print(f"Provider error: {e.message}")
except AgentError as e:
    print(f"General error: {e}")
```

## Next Steps

- [Providers](providers.md) - Configure different LLM providers
- [Tools](tools.md) - Deep dive into the tool system
- [Sessions](sessions.md) - Advanced conversation management
- [Routing](routing.md) - Multi-agent routing and fallback
