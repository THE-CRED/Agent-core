# Getting Started

Agent is a clean Python runtime for multi-provider LLM apps and agent workflows. Write your agent logic once, run it anywhere, switch providers anytime.

## Installation

```bash
# Base installation
pip install agent-core-py

# With specific providers
pip install agent-core-py[openai]
pip install agent-core-py[anthropic]
pip install agent-core-py[gemini]

# All providers
pip install agent-core-py[all]
```

## Quick Start

### 1. Set Up API Keys

Set environment variables for your providers:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

### 2. Basic Usage

```python
from agent import Agent

# Create an agent
agent = Agent(
    provider="openai",
    model="gpt-4o",
)

# Generate text
response = agent.run("Explain quantum computing in one sentence")
print(response.text)
```

### 3. Switch Providers Easily

```python
# Same code, different provider
agent = Agent(
    provider="anthropic",
    model="claude-sonnet",
)

response = agent.run("Explain quantum computing in one sentence")
print(response.text)
```

## Core Concepts

### Agent

The `Agent` class is your main interface. It handles:
- Text generation
- Streaming responses
- Structured output
- Tool calling
- Session management

```python
agent = Agent(
    provider="openai",      # Which provider to use
    model="gpt-4o",         # Which model
    temperature=0.7,        # Sampling temperature
    max_tokens=1000,        # Max output tokens
    default_system="...",   # Default system prompt
)
```

### Response

Every call returns an `AgentResponse` with:

```python
response = agent.run("Hello")

response.text          # The generated text
response.usage         # Token usage (prompt, completion, total)
response.provider      # Which provider was used
response.model         # Which model was used
response.latency_ms    # Request latency
response.cost_estimate # Estimated cost (if available)
response.tool_calls    # Any tool calls made
response.raw           # Raw provider response
```

### Streaming

Stream responses for real-time output:

```python
for event in agent.stream("Write a story"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

### Structured Output

Get typed responses with Pydantic:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]

response = agent.json(
    "Analyze this review: Great product!",
    schema=Analysis,
)

result: Analysis = response.output
print(result.sentiment)  # "positive"
```

### Tools

Give your agent capabilities:

```python
from agent import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72F, sunny"

agent = Agent(
    provider="openai",
    model="gpt-4o",
    tools=[get_weather],
)

response = agent.run("What's the weather in Tokyo?")
```

### Sessions

Multi-turn conversations with memory:

```python
session = agent.session()

session.run("My name is Alice")
session.run("I work as an engineer")

response = session.run("What do you know about me?")
# Agent remembers: name is Alice, works as engineer
```

### Routing & Fallback

Use multiple providers with automatic failover:

```python
from agent import AgentRouter

router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="openai", model="gpt-4o"),
    ],
    strategy="fallback",
)

# Automatically falls back if first provider fails
response = router.run("Hello!")
```

## Next Steps

- [API Reference](./api-reference.md) - Complete API documentation
- [Providers](./providers.md) - Provider setup guides
- [Tools](./tools.md) - Building agent tools
- [Structured Output](./structured-output.md) - Typed responses
- [Sessions](./sessions.md) - Multi-turn conversations

## CLI

Agent includes a CLI for quick experimentation:

```bash
# Single prompt
agent run "What is Python?" --provider openai

# Interactive chat
agent chat --provider anthropic --model claude-sonnet

# List providers
agent providers

# Test configuration
agent doctor
```
