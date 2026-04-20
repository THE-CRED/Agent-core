# Installation Guide

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Basic Installation

Install the core package:

```bash
pip install agent-core-py
```

Or using uv:

```bash
uv add agent-core-py
```

## Provider Extras

Agent supports multiple LLM providers. Install the extras for the providers you want to use:

### OpenAI

```bash
pip install agent-core-py[openai]
```

Required environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic

```bash
pip install agent-core-py[anthropic]
```

Required environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Google Gemini

```bash
pip install agent-core-py[gemini]
```

Required environment variable:
```bash
export GOOGLE_API_KEY="..."
```

### DeepSeek

```bash
pip install agent-core-py[deepseek]
```

Required environment variable:
```bash
export DEEPSEEK_API_KEY="..."
```

### All Providers

Install all provider dependencies at once:

```bash
pip install agent-core-py[all]
```

## Development Installation

For contributing or development:

```bash
git clone https://github.com/THE-CRED/Agent-core.git
cd agent
pip install -e ".[dev,all]"
```

This installs:
- All provider dependencies
- Testing tools (pytest, pytest-cov, pytest-asyncio)
- Linting tools (ruff, mypy)
- Documentation tools

## Verifying Installation

Test your installation:

```python
from agent import Agent

# Check available providers
from agent.providers.registry import ProviderRegistry
print(ProviderRegistry.list_providers())

# Quick test (requires API key)
agent = Agent(provider="openai", model="gpt-4o")
response = agent.run("Hello!")
print(response.text)
```

Or use the CLI:

```bash
agent doctor  # Test all configurations
agent providers  # List available providers
```

## Troubleshooting

### ImportError: No module named 'openai'

You need to install the provider extra:
```bash
pip install agent-core-py[openai]
```

### AuthenticationError

Make sure you've set the correct environment variable for your provider:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Connection Errors

Check your network connection and ensure you can reach the provider's API:
- OpenAI: https://api.openai.com
- Anthropic: https://api.anthropic.com
- Gemini: https://generativelanguage.googleapis.com
- DeepSeek: https://api.deepseek.com

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Provider Configuration](providers.md)
