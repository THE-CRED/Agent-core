"""
Agent configuration management.

Handles API keys, environment variables, and configuration loading.
"""

# Re-export from types module for backwards compatibility
from agent.types.config import (
    BASE_URLS,
    ENV_VARS,
    MODEL_ALIASES,
    PRICING,
    AgentConfig,
    estimate_cost,
    get_api_key,
    get_base_url,
    resolve_model,
)

__all__ = [
    "ENV_VARS",
    "BASE_URLS",
    "MODEL_ALIASES",
    "PRICING",
    "get_api_key",
    "get_base_url",
    "resolve_model",
    "estimate_cost",
    "AgentConfig",
]
