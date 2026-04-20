"""
Agent message types and request model.

These are the normalized internal representations used across all providers.
"""

# Re-export from types module for backwards compatibility
from agent.types.messages import AgentRequest, ContentPart, Message

__all__ = ["ContentPart", "Message", "AgentRequest"]
