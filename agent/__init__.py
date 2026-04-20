"""
Agent - A clean Python runtime for multi-provider LLM apps and agent workflows.

Write agent logic once. Run it anywhere. Switch providers anytime.
"""

from agent.agent import Agent
from agent.errors import (
    AgentError,
    AuthenticationError,
    ProviderError,
    RateLimitError,
    RoutingError,
    SchemaValidationError,
    TimeoutError,
    ToolExecutionError,
    UnsupportedFeatureError,
)
from agent.messages import AgentRequest, Message
from agent.middleware import Middleware
from agent.response import AgentResponse
from agent.router import AgentRouter
from agent.schemas import Schema
from agent.session import Session
from agent.stream import StreamEvent
from agent.tools import Tool, tool

# Re-export types from types module
from agent.types import (
    AgentConfig,
    ContentPart,
    ProviderCapabilities,
    RetryConfig,
    RouteResult,
    RoutingStrategy,
    StreamEventType,
    ToolCall,
    ToolLoopConfig,
    ToolResult,
    ToolSpec,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "Session",
    "AgentRouter",
    "AgentResponse",
    # Messages
    "Message",
    "AgentRequest",
    "ContentPart",
    # Tools
    "tool",
    "Tool",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    # Middleware
    "Middleware",
    # Errors
    "AgentError",
    "AuthenticationError",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "ToolExecutionError",
    "SchemaValidationError",
    "UnsupportedFeatureError",
    "RoutingError",
    # Schema
    "Schema",
    # Streaming
    "StreamEvent",
    "StreamEventType",
    # Config
    "AgentConfig",
    "ProviderCapabilities",
    "RetryConfig",
    "ToolLoopConfig",
    # Response
    "Usage",
    # Router
    "RoutingStrategy",
    "RouteResult",
]
