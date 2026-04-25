"""
Agent error types.

All provider-specific exceptions are normalized to these types.
"""

from typing import Any


class AgentError(Exception):
    """Base exception for all Agent errors."""

    def __init__(self, message: str, *, raw: Any = None):
        super().__init__(message)
        self.message = message
        self.raw = raw


class AuthenticationError(AgentError):
    """Raised when API authentication fails."""

    pass


class ProviderError(AgentError):
    """Raised when the provider returns an error."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        raw: Any = None,
    ):
        super().__init__(message, raw=raw)
        self.provider = provider
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Raised when rate limited by the provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
        raw: Any = None,
    ):
        super().__init__(message, provider=provider, raw=raw)
        self.retry_after = retry_after


class RequestTimeoutError(AgentError):
    """Raised when a request times out."""

    def __init__(self, message: str, *, timeout: float | None = None, raw: Any = None):
        super().__init__(message, raw=raw)
        self.timeout = timeout


# Backwards compatibility alias
TimeoutError = RequestTimeoutError


class ToolExecutionError(AgentError):
    """Raised when a tool fails to execute."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        raw: Any = None,
    ):
        super().__init__(message, raw=raw)
        self.tool_name = tool_name


class SchemaValidationError(AgentError):
    """Raised when structured output fails validation."""

    def __init__(
        self,
        message: str,
        *,
        schema: Any = None,
        output: Any = None,
        raw: Any = None,
    ):
        super().__init__(message, raw=raw)
        self.schema = schema
        self.output = output


class UnsupportedFeatureError(AgentError):
    """Raised when a requested feature is not supported by the provider."""

    def __init__(
        self,
        message: str,
        *,
        feature: str | None = None,
        provider: str | None = None,
        raw: Any = None,
    ):
        super().__init__(message, raw=raw)
        self.feature = feature
        self.provider = provider


class RoutingError(AgentError):
    """Raised when routing fails across all configured agents."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[Exception] | None = None,
        raw: Any = None,
    ):
        super().__init__(message, raw=raw)
        self.errors = errors or []
