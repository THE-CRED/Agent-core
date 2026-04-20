# Agent Specifications

This directory contains comprehensive technical specifications for the Agent library.

## User Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](./getting-started.md) | Installation and first steps |
| [API Reference](./api-reference.md) | Complete API documentation |
| [Providers](./providers.md) | Provider setup and configuration |
| [Tools](./tools.md) | Creating and using tools |
| [Structured Output](./structured-output.md) | Typed responses with Pydantic |
| [Sessions](./sessions.md) | Multi-turn conversations |
| [Routing & Fallback](./routing.md) | Multi-agent routing strategies |

## Type System

| Document | Description |
|----------|-------------|
| [Type System](./types.md) | Pydantic models and type definitions |

## Internal Documentation

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System architecture overview |
| [Execution Runtime](./execution-runtime.md) | How requests are processed |
| [Provider Contract](./provider-contract.md) | Building provider adapters |

## Module Structure

```
agent/
├── __init__.py              # Public API exports
├── agent.py                 # Main Agent class
├── session.py               # Session management
├── router.py                # AgentRouter for multi-agent routing
├── errors.py                # Exception hierarchy
├── schemas.py               # JSON Schema utilities
├── middleware.py            # Middleware system
├── tools.py                 # Tool decorator and registry
├── stream.py                # Streaming response classes
├── messages.py              # Re-exports from types
├── response.py              # Re-exports from types
├── config.py                # Re-exports from types
│
├── types/                   # Pydantic type definitions
│   ├── __init__.py          # Type exports
│   ├── messages.py          # ContentPart, Message, AgentRequest
│   ├── response.py          # Usage, AgentResponse
│   ├── tools.py             # ToolSpec, ToolCall, ToolResult
│   ├── stream.py            # StreamEvent, StreamEventType
│   ├── config.py            # Configuration types
│   └── router.py            # Router types
│
├── providers/               # LLM provider adapters
│   ├── base.py              # BaseProvider interface
│   ├── registry.py          # Provider registration
│   ├── openai.py            # OpenAI adapter
│   ├── anthropic.py         # Anthropic adapter
│   ├── gemini.py            # Google Gemini adapter
│   └── deepseek.py          # DeepSeek adapter
│
├── execution/               # Request execution
│   ├── runtime.py           # ExecutionRuntime
│   ├── retries.py           # Retry handling
│   ├── tool_loop.py         # Tool execution loop
│   └── structured_output.py # Structured output parsing
│
├── stores/                  # Session persistence
│   ├── base.py              # SessionStore interface
│   ├── memory.py            # In-memory store
│   └── sqlite.py            # SQLite store
│
├── testing/                 # Test utilities
│   ├── fake_provider.py     # Mock provider for testing
│   └── fixtures.py          # Pytest fixtures
│
└── cli/                     # Command-line interface
    └── main.py              # CLI entry point
```

## Quick Links

- **New to Agent?** Start with [Getting Started](./getting-started.md)
- **Building tools?** See [Tools](./tools.md)
- **Understanding types?** See [Type System](./types.md)
- **Adding a provider?** See [Provider Contract](./provider-contract.md)
- **Understanding internals?** See [Architecture](./architecture.md)

## Related Documentation

For user-facing documentation, see the [docs/](../docs/) directory which includes:
- Installation guide
- Quick start tutorial
- Streaming guide
- Error handling guide
- Custom provider guide
- Configuration reference
