# Agent

**Write agent logic once. Run it anywhere. Switch providers anytime.**

Agent is a clean Python runtime for multi-provider LLM apps and agent workflows. One elegant, provider-agnostic interface for building LLM applications and tool-using agents.

## Features

- **One API, Many Providers**: OpenAI, Anthropic, Gemini, DeepSeek, and more
- **Provider Portability**: Switch providers with minimal code changes
- **Structured Outputs**: Pydantic-based typed responses
- **Tool Calling**: Register Python functions as tools with automatic schema generation
- **Sessions**: Multi-turn conversations with pluggable persistence
- **Streaming**: Normalized streaming events across providers
- **Routing & Fallback**: Automatic failover and smart routing strategies
- **Middleware**: Extensible hooks for logging, tracing, and policy control

## Installation

```bash
pip install agent-runtime

# With provider extras
pip install agent-runtime[openai]
pip install agent-runtime[anthropic]
pip install agent-runtime[all]
```

## Quick Start

```python
from agent import Agent

# Create an agent
agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
)

# Simple text generation
response = agent.run("Explain quantum computing in one paragraph")
print(response.text)
```

## Structured Outputs

```python
from pydantic import BaseModel
from agent import Agent

class Summary(BaseModel):
    title: str
    bullets: list[str]
    sentiment: str

agent = Agent(provider="openai", model="gpt-4o")
response = agent.json("Summarize this article about AI progress", schema=Summary)
print(response.output)  # Typed Summary object
```

## Tool Calling

```python
from agent import Agent, tool

@tool
def search_code(query: str) -> str:
    """Search the codebase for matching patterns."""
    # Your implementation
    return f"Found 5 matches for: {query}"

@tool
def read_file(path: str) -> str:
    """Read contents of a file."""
    with open(path) as f:
        return f.read()

agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    tools=[search_code, read_file],
)

response = agent.run("Find all TODO comments in the codebase")
```

## Sessions

```python
from agent import Agent

agent = Agent(provider="openai", model="gpt-4o")

# Create a session for multi-turn conversation
session = agent.session()
session.run("My name is Alice")
response = session.run("What's my name?")
print(response.text)  # "Your name is Alice"
```

## Streaming

```python
from agent import Agent

agent = Agent(provider="anthropic", model="claude-sonnet-4-20250514")

for event in agent.stream("Write a short poem about coding"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

## Router & Fallback

```python
from agent import AgentRouter

router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet-4-20250514"),
        Agent(provider="openai", model="gpt-4o"),
    ],
    strategy="fallback",
)

# Automatically falls back if first provider fails
response = router.run("Hello, world!")
```

## Middleware

```python
from agent import Agent, Middleware

class LoggingMiddleware(Middleware):
    def before(self, request):
        print(f"Request: {request.input[:50]}...")
        return request
    
    def after(self, request, response):
        print(f"Response: {response.text[:50]}...")
        return response

agent = Agent(
    provider="openai",
    model="gpt-4o",
    middleware=[LoggingMiddleware()],
)
```

## Configuration

```python
from agent import Agent

# Explicit configuration
agent = Agent(
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    timeout=120.0,
    max_retries=3,
    temperature=0.7,
)

# Or use environment variables
# OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
agent = Agent(provider="openai", model="gpt-4o")
```

## CLI

```bash
# Quick chat
agent chat --provider openai --model gpt-4o

# Single prompt
agent run "What is the capital of France?" --provider anthropic

# List providers
agent providers

# Test configuration
agent doctor
```

## Supported Providers

| Provider | Text | Streaming | Tools | Structured Output | Vision |
|----------|------|-----------|-------|-------------------|--------|
| OpenAI | Yes | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | Yes | Yes |
| Gemini | Yes | Yes | Yes | Yes | Yes |
| DeepSeek | Yes | Yes | Yes | Yes | No |

## Documentation

- [Installation Guide](docs/installation.md)
- [Provider Setup](docs/providers.md)
- [Structured Outputs](docs/structured-outputs.md)
- [Tool System](docs/tools.md)
- [Sessions](docs/sessions.md)
- [Routing & Fallback](docs/routing.md)
- [Middleware](docs/middleware.md)
- [Custom Providers](docs/custom-providers.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
