# Providers

Agent provides a unified interface across multiple LLM providers. This guide covers provider configuration, capabilities, and best practices.

## Supported Providers

| Provider | Text | Streaming | Tools | Structured Output | Vision |
|----------|------|-----------|-------|-------------------|--------|
| OpenAI | Yes | Yes | Yes | Yes (native) | Yes |
| Anthropic | Yes | Yes | Yes | Yes | Yes |
| Gemini | Yes | Yes | Yes | Yes (native) | Yes |
| DeepSeek | Yes | Yes | Yes | Yes | No |

## OpenAI

### Setup

```bash
pip install agent-runtime[openai]
export OPENAI_API_KEY="sk-..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="openai",
    model="gpt-4o",  # or "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
)
```

### Available Models

| Model | Alias | Context | Best For |
|-------|-------|---------|----------|
| gpt-4o | `gpt-4o` | 128K | Complex tasks, vision |
| gpt-4o-mini | `gpt-4o-mini` | 128K | Fast, cost-effective |
| gpt-4-turbo | `gpt-4` | 128K | Complex reasoning |
| gpt-3.5-turbo | `gpt-3.5` | 16K | Simple tasks |

### OpenAI-Specific Features

**Native JSON Schema**:
```python
# OpenAI supports native schema-enforced output
response = agent.json("Extract data", schema=MyModel)
# Uses response_format with json_schema
```

**Vision**:
```python
from agent import Message, ContentPart

message = Message.user([
    ContentPart.text_part("What's in this image?"),
    ContentPart.image_url_part("https://example.com/image.jpg"),
])
response = agent.run(messages=[message])
```

## Anthropic

### Setup

```bash
pip install agent-runtime[anthropic]
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",  # or use alias "claude-sonnet"
)
```

### Available Models

| Model | Alias | Context | Best For |
|-------|-------|---------|----------|
| claude-opus-4-20250514 | `claude-opus` | 200K | Complex analysis |
| claude-sonnet-4-20250514 | `claude-sonnet`, `claude` | 200K | Balanced performance |
| claude-3-5-haiku-20241022 | `claude-haiku` | 200K | Fast responses |

### Anthropic-Specific Features

**Long Context**:
```python
# Claude supports very long contexts (200K tokens)
agent = Agent(provider="anthropic", model="claude-sonnet")
response = agent.run(very_long_document)
```

**System Prompts**:
```python
# Anthropic handles system prompts separately
response = agent.run(
    "Analyze this code",
    system="You are an expert code reviewer. Be thorough and constructive."
)
```

## Google Gemini

### Setup

```bash
pip install agent-runtime[gemini]
export GOOGLE_API_KEY="..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="gemini",
    model="gemini-1.5-pro",  # or "gemini-1.5-flash"
)
```

### Available Models

| Model | Alias | Context | Best For |
|-------|-------|---------|----------|
| gemini-1.5-pro | `gemini-pro` | 1M | Complex tasks |
| gemini-1.5-flash | `gemini-flash` | 1M | Fast, efficient |

### Gemini-Specific Features

**Million Token Context**:
```python
# Gemini supports up to 1M tokens
agent = Agent(provider="gemini", model="gemini-pro")
response = agent.run(enormous_document)
```

## DeepSeek

### Setup

```bash
pip install agent-runtime[deepseek]
export DEEPSEEK_API_KEY="..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="deepseek",
    model="deepseek-chat",  # or "deepseek-coder"
)
```

### Available Models

| Model | Alias | Best For |
|-------|-------|----------|
| deepseek-chat | `deepseek` | General conversation |
| deepseek-coder | `deepseek-coder` | Code generation |

## Custom Endpoints

Use OpenAI-compatible APIs:

```python
from agent import Agent

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
```

## Provider Capabilities

Check what a provider supports at runtime:

```python
from agent import Agent

agent = Agent(provider="openai", model="gpt-4o")

# Check capabilities via the provider
print(agent._provider.supports_tools())            # True
print(agent._provider.supports_streaming())        # True
print(agent._provider.supports_vision())           # True
print(agent._provider.supports_structured_output()) # True
print(agent._provider.supports_native_schema())    # True
```

## Provider-Agnostic Code

Write code that works across providers:

```python
from agent import Agent

def analyze_text(text: str, provider: str = "openai") -> str:
    """Analyze text using any provider."""
    models = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet",
        "gemini": "gemini-pro",
        "deepseek": "deepseek-chat",
    }
    
    agent = Agent(
        provider=provider,
        model=models[provider],
        temperature=0.3,  # Works across all providers
    )
    
    return agent.run(f"Analyze: {text}").text
```

## Cost Optimization

Agent tracks usage and estimates costs:

```python
response = agent.run("Hello!")

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Estimated cost: ${response.cost_estimate:.6f}")
```

Use the router for cost-optimized routing:

```python
from agent import Agent, AgentRouter

router = AgentRouter(
    agents=[
        Agent(provider="openai", model="gpt-4o-mini"),  # Cheapest
        Agent(provider="anthropic", model="claude-haiku"),
        Agent(provider="openai", model="gpt-4o"),  # Most expensive
    ],
    strategy="cheapest",  # Use cheapest available
)
```

## Next Steps

- [Tools](tools.md) - Register functions as LLM tools
- [Structured Outputs](structured-outputs.md) - Type-safe responses
- [Routing](routing.md) - Multi-provider fallback
