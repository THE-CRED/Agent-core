# Contributing to Agent

Thank you for your interest in contributing to Agent! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/THE-CRED/Agent-core.git
cd agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev,all]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent --cov-report=html

# Run specific test file
pytest tests/test_agent.py

# Run integration tests (requires API keys)
OPENAI_API_KEY=... pytest tests/integration/
```

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy agent
```

## Adding a New Provider

1. Create a new file in `agent/providers/`:

```python
# agent/providers/newprovider.py
from agent.providers.base import BaseProvider
from agent.messages import AgentRequest
from agent.response import AgentResponse

class NewProvider(BaseProvider):
    name = "newprovider"
    
    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        # Initialize provider-specific client
    
    def run(self, request: AgentRequest) -> AgentResponse:
        # Implement synchronous generation
        pass
    
    async def run_async(self, request: AgentRequest) -> AgentResponse:
        # Implement async generation
        pass
    
    def stream(self, request: AgentRequest):
        # Implement streaming
        pass
    
    def supports_tools(self) -> bool:
        return True  # or False
    
    def supports_structured_output(self) -> bool:
        return True  # or False
    
    def supports_vision(self) -> bool:
        return True  # or False
```

2. Register the provider in `agent/providers/__init__.py`

3. Add conformance tests in `tests/providers/test_newprovider.py`

4. Update documentation in `docs/providers.md`

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Ensure all tests pass and code quality checks pass
4. Update documentation if needed
5. Submit a pull request with a clear description

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all public APIs
- Write docstrings for public classes and functions
- Keep functions focused and small
- Prefer composition over inheritance

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for streaming in Gemini provider
fix: handle rate limit errors in OpenAI adapter
docs: update tool registration examples
test: add conformance tests for structured output
```

## Questions?

Open an issue for questions or discussions about contributing.
