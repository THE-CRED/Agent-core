"""
Configuration types for Agent.
"""

import os
import random
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Provider-specific environment variable names
ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "KIMI_API_KEY",
}

# Default base URLs
BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.cn/v1",
}

# Model aliases for convenience
MODEL_ALIASES = {
    # OpenAI
    "gpt-4": "gpt-4-turbo-preview",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-3.5": "gpt-3.5-turbo",
    # Anthropic
    "claude": "claude-sonnet-4-20250514",
    "claude-opus": "claude-opus-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    # Gemini
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
    # DeepSeek
    "deepseek": "deepseek-chat",
    "deepseek-coder": "deepseek-coder",
}

# Pricing per 1M tokens (approximate, may change)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-coder": {"input": 0.14, "output": 0.28},
}


def get_api_key(provider: str, api_key: str | None = None) -> str | None:
    """Get API key for a provider."""
    if api_key:
        return api_key
    env_var = ENV_VARS.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def get_base_url(provider: str, base_url: str | None = None) -> str | None:
    """Get base URL for a provider."""
    if base_url:
        return base_url
    return BASE_URLS.get(provider)


def resolve_model(model: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_ALIASES.get(model, model)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate cost for a request."""
    model = resolve_model(model)
    pricing = PRICING.get(model)
    if not pricing:
        return None
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class ProviderCapabilities(BaseModel):
    """Declares what features a provider supports."""

    streaming: bool = True
    tools: bool = True
    structured_output: bool = True
    json_mode: bool = True
    vision: bool = False
    system_messages: bool = True
    batch: bool = False
    native_schema_output: bool = False
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None


class AgentConfig(BaseModel):
    """Configuration for an Agent instance."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    default_system: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_config(self) -> "AgentConfig":
        """Resolve model alias and get API key/base URL from environment."""
        self.model = resolve_model(self.model)
        if not self.api_key:
            self.api_key = get_api_key(self.provider)
        if not self.base_url:
            self.base_url = get_base_url(self.provider)
        return self

    def with_overrides(self, **kwargs: Any) -> "AgentConfig":
        """Create a new config with overrides."""
        return AgentConfig(
            provider=kwargs.get("provider", self.provider),
            model=kwargs.get("model", self.model),
            api_key=kwargs.get("api_key", self.api_key),
            base_url=kwargs.get("base_url", self.base_url),
            timeout=kwargs.get("timeout", self.timeout),
            max_retries=kwargs.get("max_retries", self.max_retries),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            default_system=kwargs.get("default_system", self.default_system),
            extra={**self.extra, **kwargs.get("extra", {})},
        )


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = 2
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple[type[Exception], ...] = Field(
        default=(
            ConnectionError,
            TimeoutError,
        )
    )

    model_config = {"arbitrary_types_allowed": True}

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if we should retry for this error."""
        from agent.errors import ProviderError, RateLimitError

        if attempt >= self.max_retries:
            return False

        # Always retry rate limit errors if we have retries left
        if isinstance(error, RateLimitError):
            return True

        # Retry provider errors that are likely transient (5xx)
        if isinstance(error, ProviderError):
            if error.status_code and 500 <= error.status_code < 600:
                return True
            # Don't retry client errors (4xx except rate limit)
            if error.status_code and 400 <= error.status_code < 500:
                return False

        return isinstance(error, self.retryable_errors)

    def get_delay(self, attempt: int, error: Exception | None = None) -> float:
        """Calculate delay before next retry."""
        from agent.errors import RateLimitError

        # Use retry-after header if available
        if isinstance(error, RateLimitError) and error.retry_after:
            return min(error.retry_after, self.max_delay)

        # Exponential backoff
        delay = self.initial_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        # Add jitter
        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


class ToolLoopConfig(BaseModel):
    """Configuration for tool loop behavior."""

    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 20
    timeout_per_tool: float = 30.0
    parallel_tool_execution: bool = True
    stop_on_error: bool = False
