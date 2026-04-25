"""
Agent middleware system.

Allows extension without bloating the core API.
"""

import re
from typing import Any

from agent.errors import ProviderError, RateLimitError, TimeoutError
from agent.messages import AgentRequest
from agent.response import AgentResponse


class Middleware:
    """
    Base middleware class.

    Implement before(), after(), and/or on_error() to hook into the request lifecycle.
    """

    def before(self, request: AgentRequest) -> AgentRequest:
        """
        Called before the request is sent to the provider.

        Can modify the request or return a new one.
        """
        return request

    def after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        """
        Called after receiving a response from the provider.

        Can modify the response or return a new one.
        """
        return response

    def on_error(self, request: AgentRequest, error: Exception) -> Exception | None:
        """
        Called when an error occurs.

        Return None to suppress the error, or return the error (modified or not).
        """
        return error


class LoggingMiddleware(Middleware):
    """Simple logging middleware."""

    def __init__(self, log_fn: Any = None):
        self.log_fn = log_fn or print

    def before(self, request: AgentRequest) -> AgentRequest:
        input_preview = (request.input or "")[:100]
        self.log_fn(f"[Agent] Request: {input_preview}...")
        return request

    def after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        text_preview = (response.text or "")[:100]
        self.log_fn(f"[Agent] Response: {text_preview}...")
        if response.usage:
            self.log_fn(f"[Agent] Tokens: {response.usage.total_tokens}")
        return response

    def on_error(self, request: AgentRequest, error: Exception) -> Exception:
        self.log_fn(f"[Agent] Error: {error}")
        return error


class MetricsMiddleware(Middleware):
    """Middleware that collects basic metrics."""

    def __init__(self):
        self.request_count = 0
        self.total_tokens = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

    def before(self, request: AgentRequest) -> AgentRequest:
        self.request_count += 1
        return request

    def after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        if response.latency_ms:
            self.total_latency_ms += response.latency_ms
        return response

    def on_error(self, request: AgentRequest, error: Exception) -> Exception:
        self.error_count += 1
        return error

    def stats(self) -> dict[str, Any]:
        """Get collected metrics."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": (
                self.total_latency_ms / self.request_count if self.request_count > 0 else 0
            ),
        }


class RedactionMiddleware(Middleware):
    """Middleware that redacts sensitive information from logs/traces."""

    def __init__(self, patterns: list[str] | None = None):
        self.patterns = patterns or [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Emails
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card numbers
        ]
        self._compiled = [re.compile(p) for p in self.patterns]

    def _redact(self, text: str) -> str:
        """Redact sensitive patterns from text."""
        for pattern in self._compiled:
            text = pattern.sub("[REDACTED]", text)
        return text

    def before(self, request: AgentRequest) -> AgentRequest:
        # Redact sensitive data from request input before it reaches logs/traces
        if request.input:
            request.input = self._redact(request.input)
        return request

    def after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        # Redact sensitive data from response text
        if response.text:
            response.text = self._redact(response.text)
        return response


class RetryPolicyMiddleware(Middleware):
    """Middleware that applies custom retry policies."""

    def __init__(
        self,
        max_retries: int = 3,
        retryable_errors: tuple[type[Exception], ...] | None = None,
    ):
        self.max_retries = max_retries
        self.retryable_errors = retryable_errors or (
            RateLimitError,
            ProviderError,
            TimeoutError,
        )
        self._retry_count = 0

    def on_error(self, request: AgentRequest, error: Exception) -> Exception | None:
        if isinstance(error, self.retryable_errors):
            self._retry_count += 1
            # The actual retry logic is handled by the execution layer
            # This middleware just tracks retries
        return error


class MiddlewareChain:
    """Chain of middleware to be executed in order."""

    def __init__(self, middlewares: list[Middleware] | None = None):
        self.middlewares = middlewares or []

    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """Add a middleware to the chain."""
        self.middlewares.append(middleware)
        return self

    def run_before(self, request: AgentRequest) -> AgentRequest:
        """Run all before hooks."""
        for mw in self.middlewares:
            request = mw.before(request)
        return request

    def run_after(self, request: AgentRequest, response: AgentResponse) -> AgentResponse:
        """Run all after hooks in reverse order."""
        for mw in reversed(self.middlewares):
            response = mw.after(request, response)
        return response

    def run_on_error(self, request: AgentRequest, error: Exception) -> Exception | None:
        """Run all error hooks."""
        for mw in self.middlewares:
            result = mw.on_error(request, error)
            if result is None:
                return None
            error = result
        return error
