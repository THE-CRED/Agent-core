"""
Router types for Agent.
"""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class RoutingStrategy(str, Enum):
    """Available routing strategies."""

    FALLBACK = "fallback"  # Try each agent in order until one succeeds
    ROUND_ROBIN = "round_robin"  # Rotate through agents
    FASTEST = "fastest"  # Race agents, use first response
    CHEAPEST = "cheapest"  # Use cheapest available agent
    CAPABILITY = "capability"  # Route based on required capabilities
    CUSTOM = "custom"  # User-provided routing function


class RouteResult(BaseModel):
    """Result of a routing decision."""

    agent: Any  # Agent type - using Any to avoid circular import
    reason: str | None = None

    model_config = {"arbitrary_types_allowed": True}
