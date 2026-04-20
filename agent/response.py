"""
Agent response model.

Normalized response object returned by all providers.
"""

# Re-export from types module for backwards compatibility
from agent.types.response import AgentResponse, Usage

__all__ = ["Usage", "AgentResponse"]
