# Providers

Agent supports multiple LLM providers through a unified interface. This guide covers setup and configuration for each provider.

## Supported Providers

| Provider | Package Extra | Environment Variable | Features |
|----------|--------------|---------------------|----------|
| OpenAI | `agent-core-py[openai]` | `OPENAI_API_KEY` | Streaming, Tools, JSON Mode, Vision |
| Anthropic | `agent-core-py[anthropic]` | `ANTHROPIC_API_KEY` | Streaming, Tools, Vision |
| Gemini | `agent-core-py[gemini]` | `GOOGLE_API_KEY` | Streaming, Tools, Vision |
| DeepSeek | `agent-core-py[deepseek]` | `DEEPSEEK_API_KEY` | Streaming, Tools |

## OpenAI

### Installation

```bash
pip install agent-core-py[openai]
```

### Configuration

```bash
export OPENAI_API_KEY="sk-..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="openai",
    model="gpt-4o",  # or "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
)

response = agent.run("Hello!")
```

### Available Models

| Model | Alias | Best For |
|-------|-------|----------|
| `gpt-4o` | - | Best quality, multimodal |
| `gpt-4o-mini` | - | Fast, cost-effective |
| `gpt-4-turbo-preview` | `gpt-4` | High quality |
| `gpt-3.5-turbo` | `gpt-3.5` | Budget option |

### Features

```python
# Vision
from agent.messages import Message, ContentPart

agent = Agent(provider="openai", model="gpt-4o")
response = agent.run(
    messages=[
        Message.user([
            ContentPart.text_part("What's in this image?"),
            ContentPart.image_url_part("https://example.com/image.jpg"),
        ])
    ]
)

# JSON Mode (native)
response = agent.json("Extract data", schema=MySchema)  # Uses native JSON mode

# Tools
agent = Agent(provider="openai", model="gpt-4o", tools=[my_tool])
```

### Custom Endpoint

```python
# Use Azure OpenAI or compatible API
agent = Agent(
    provider="openai",
    model="gpt-4",
    base_url="https://your-resource.openai.azure.com/",
    api_key="your-azure-key",
)
```

---

## Anthropic

### Installation

```bash
pip install agent-core-py[anthropic]
```

### Configuration

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",  # or alias "claude-sonnet"
)

response = agent.run("Hello!")
```

### Available Models

| Model | Alias | Best For |
|-------|-------|----------|
| `claude-opus-4-20250514` | `claude-opus` | Most capable |
| `claude-sonnet-4-20250514` | `claude-sonnet`, `claude` | Balanced |
| `claude-3-5-haiku-20241022` | `claude-haiku` | Fast, efficient |

### Features

```python
# Vision
response = agent.run(
    messages=[
        Message.user([
            ContentPart.text_part("Describe this image"),
            ContentPart.image_data_part(image_bytes, "image/png"),
        ])
    ]
)

# Long context
agent = Agent(
    provider="anthropic",
    model="claude-sonnet",
    max_tokens=4096,
)

# Tools (native support)
agent = Agent(provider="anthropic", model="claude-sonnet", tools=[my_tool])
```

### System Prompts

Anthropic handles system prompts specially:

```python
agent = Agent(
    provider="anthropic",
    model="claude-sonnet",
    default_system="You are a helpful assistant.",
)

# Or per-request
response = agent.run("Hello", system="Be concise.")
```

---

## Google Gemini

### Installation

```bash
pip install agent-core-py[gemini]
```

### Configuration

```bash
export GOOGLE_API_KEY="..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="gemini",
    model="gemini-1.5-pro",  # or "gemini-1.5-flash"
)

response = agent.run("Hello!")
```

### Available Models

| Model | Alias | Best For |
|-------|-------|----------|
| `gemini-1.5-pro` | `gemini-pro` | Best quality |
| `gemini-1.5-flash` | `gemini-flash` | Fast, efficient |

### Features

```python
# Vision (native multimodal)
response = agent.run(
    messages=[
        Message.user([
            ContentPart.text_part("What's in this image?"),
            ContentPart.image_data_part(image_bytes),
        ])
    ]
)

# Tools
agent = Agent(provider="gemini", model="gemini-1.5-pro", tools=[my_tool])

# Long context (up to 1M tokens)
agent = Agent(
    provider="gemini",
    model="gemini-1.5-pro",
    max_tokens=8192,
)
```

---

## DeepSeek

### Installation

```bash
pip install agent-core-py[deepseek]
```

### Configuration

```bash
export DEEPSEEK_API_KEY="..."
```

### Usage

```python
from agent import Agent

agent = Agent(
    provider="deepseek",
    model="deepseek-chat",  # or "deepseek-coder"
)

response = agent.run("Hello!")
```

### Available Models

| Model | Alias | Best For |
|-------|-------|----------|
| `deepseek-chat` | `deepseek` | General chat |
| `deepseek-coder` | - | Code generation |

### Features

DeepSeek uses an OpenAI-compatible API:

```python
# Tools (OpenAI-compatible)
agent = Agent(provider="deepseek", model="deepseek-chat", tools=[my_tool])

# Structured output
response = agent.json("Extract data", schema=MySchema)
```

---

## Provider Comparison

### Capability Matrix

| Feature | OpenAI | Anthropic | Gemini | DeepSeek |
|---------|--------|-----------|--------|----------|
| Streaming | Yes | Yes | Yes | Yes |
| Tools | Yes | Yes | Yes | Yes |
| Structured Output | Native | Prompt-based | Native | Prompt-based |
| Vision | Yes | Yes | Yes | No |
| JSON Mode | Yes | No | Yes | Yes |
| Max Context | 128K | 200K | 1M | 64K |

### Pricing (Approximate)

| Provider | Model | Input/1M | Output/1M |
|----------|-------|----------|-----------|
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Anthropic | claude-sonnet | $3.00 | $15.00 |
| Anthropic | claude-haiku | $0.25 | $1.25 |
| Gemini | gemini-1.5-pro | $1.25 | $5.00 |
| Gemini | gemini-1.5-flash | $0.075 | $0.30 |
| DeepSeek | deepseek-chat | $0.14 | $0.28 |

---

## Custom/Compatible Providers

### OpenAI-Compatible APIs

Many providers offer OpenAI-compatible APIs:

```python
# Together AI
agent = Agent(
    provider="openai",
    model="meta-llama/Llama-3-70b-chat-hf",
    base_url="https://api.together.xyz/v1",
    api_key="your-together-key",
)

# Groq
agent = Agent(
    provider="openai",
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai/v1",
    api_key="your-groq-key",
)

# Local (Ollama)
agent = Agent(
    provider="openai",
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key
)
```

### Adding Custom Providers

See [Provider Contract](./provider-contract.md) for building custom adapters.

---

## Best Practices

### 1. Use Environment Variables

```python
# Don't hardcode keys
agent = Agent(provider="openai", model="gpt-4o")  # Uses OPENAI_API_KEY

# Or use explicit key from secure source
import os
agent = Agent(
    provider="openai",
    model="gpt-4o",
    api_key=os.environ["MY_OPENAI_KEY"],
)
```

### 2. Configure Timeouts

```python
agent = Agent(
    provider="openai",
    model="gpt-4o",
    timeout=60.0,      # Request timeout
    max_retries=3,     # Retry on transient errors
)
```

### 3. Use Router for Reliability

```python
from agent import AgentRouter

router = AgentRouter(
    agents=[
        Agent(provider="anthropic", model="claude-sonnet"),
        Agent(provider="openai", model="gpt-4o"),
    ],
    strategy="fallback",
)
```

### 4. Check Capabilities

```python
# The agent will raise UnsupportedFeatureError if capability missing
try:
    response = agent.run("Describe this image", ...)
except UnsupportedFeatureError as e:
    print(f"Provider doesn't support: {e.feature}")
```
