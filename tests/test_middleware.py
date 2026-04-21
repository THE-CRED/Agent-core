"""Tests for agent middleware system."""

from agent.errors import ProviderError, RateLimitError
from agent.errors import TimeoutError as AgentTimeoutError
from agent.messages import AgentRequest
from agent.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewareChain,
    RedactionMiddleware,
    RetryPolicyMiddleware,
)
from agent.response import AgentResponse, Usage


def _make_request(input_text="test input"):
    return AgentRequest(input=input_text, messages=[])


def _make_response(text="test response", usage=None, latency_ms=None):
    return AgentResponse(
        text=text,
        usage=usage,
        latency_ms=latency_ms,
    )


# ── Base Middleware ───────────────────────────────────────────────


class TestMiddleware:
    def test_before_passthrough(self):
        mw = Middleware()
        req = _make_request()
        assert mw.before(req) is req

    def test_after_passthrough(self):
        mw = Middleware()
        req = _make_request()
        resp = _make_response()
        assert mw.after(req, resp) is resp

    def test_on_error_passthrough(self):
        mw = Middleware()
        req = _make_request()
        err = Exception("test")
        assert mw.on_error(req, err) is err


# ── LoggingMiddleware ────────────────────────────────────────────


class TestLoggingMiddleware:
    def test_before_logs_input(self):
        logs = []
        mw = LoggingMiddleware(log_fn=logs.append)
        req = _make_request("Hello world")
        result = mw.before(req)
        assert result is req
        assert len(logs) == 1
        assert "Hello world" in logs[0]

    def test_after_logs_response(self):
        logs = []
        mw = LoggingMiddleware(log_fn=logs.append)
        req = _make_request()
        resp = _make_response("The answer is 42")
        mw.after(req, resp)
        assert any("The answer" in log for log in logs)

    def test_after_logs_usage(self):
        logs = []
        mw = LoggingMiddleware(log_fn=logs.append)
        req = _make_request()
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = _make_response("ok", usage=usage)
        mw.after(req, resp)
        assert any("15" in log for log in logs)

    def test_on_error_logs(self):
        logs = []
        mw = LoggingMiddleware(log_fn=logs.append)
        req = _make_request()
        err = Exception("boom")
        result = mw.on_error(req, err)
        assert result is err
        assert any("boom" in log for log in logs)

    def test_default_log_fn_is_print(self):
        mw = LoggingMiddleware()
        assert mw.log_fn is print

    def test_truncates_long_input(self):
        logs = []
        mw = LoggingMiddleware(log_fn=logs.append)
        long_input = "x" * 200
        req = _make_request(long_input)
        mw.before(req)
        # The preview is [:100] so logged line should be shorter than full input
        assert len(logs[0]) < 220


# ── MetricsMiddleware ────────────────────────────────────────────


class TestMetricsMiddleware:
    def test_initial_state(self):
        mw = MetricsMiddleware()
        assert mw.request_count == 0
        assert mw.total_tokens == 0
        assert mw.error_count == 0
        assert mw.total_latency_ms == 0.0

    def test_before_increments_count(self):
        mw = MetricsMiddleware()
        mw.before(_make_request())
        mw.before(_make_request())
        assert mw.request_count == 2

    def test_after_accumulates_tokens(self):
        mw = MetricsMiddleware()
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mw.after(_make_request(), _make_response(usage=usage))
        assert mw.total_tokens == 15

    def test_after_accumulates_latency(self):
        mw = MetricsMiddleware()
        mw.after(_make_request(), _make_response(latency_ms=100.0))
        mw.after(_make_request(), _make_response(latency_ms=200.0))
        assert mw.total_latency_ms == 300.0

    def test_after_no_usage(self):
        mw = MetricsMiddleware()
        mw.after(_make_request(), _make_response())
        assert mw.total_tokens == 0

    def test_on_error_increments(self):
        mw = MetricsMiddleware()
        mw.on_error(_make_request(), Exception("e"))
        assert mw.error_count == 1

    def test_stats(self):
        mw = MetricsMiddleware()
        mw.before(_make_request())
        mw.before(_make_request())
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=100)
        mw.after(_make_request(), _make_response(usage=usage, latency_ms=50.0))
        mw.after(_make_request(), _make_response(usage=usage, latency_ms=150.0))
        mw.on_error(_make_request(), Exception("e"))

        stats = mw.stats()
        assert stats["request_count"] == 2
        assert stats["total_tokens"] == 200
        assert stats["error_count"] == 1
        assert stats["total_latency_ms"] == 200.0
        assert stats["avg_latency_ms"] == 100.0

    def test_stats_avg_zero_requests(self):
        mw = MetricsMiddleware()
        assert mw.stats()["avg_latency_ms"] == 0


# ── RedactionMiddleware ──────────────────────────────────────────


class TestRedactionMiddleware:
    def test_default_patterns(self):
        mw = RedactionMiddleware()
        assert len(mw.patterns) == 4
        assert len(mw._compiled) == 4

    def test_redacts_openai_key(self):
        mw = RedactionMiddleware()
        text = "key is sk-abcdefghijklmnopqrstuvwxyz"
        assert "[REDACTED]" in mw._redact(text)

    def test_redacts_email(self):
        mw = RedactionMiddleware()
        text = "contact user@example.com for info"
        assert "[REDACTED]" in mw._redact(text)

    def test_redacts_ssn(self):
        mw = RedactionMiddleware()
        text = "SSN is 123-45-6789"
        assert "[REDACTED]" in mw._redact(text)

    def test_redacts_credit_card(self):
        mw = RedactionMiddleware()
        text = "card 1234567890123456 on file"
        assert "[REDACTED]" in mw._redact(text)

    def test_custom_patterns(self):
        mw = RedactionMiddleware(patterns=[r"secret-\w+"])
        text = "the secret-password123 is here"
        assert "[REDACTED]" in mw._redact(text)

    def test_before_returns_request(self):
        mw = RedactionMiddleware()
        req = _make_request()
        assert mw.before(req) is req

    def test_after_returns_response(self):
        mw = RedactionMiddleware()
        resp = _make_response()
        assert mw.after(_make_request(), resp) is resp


# ── RetryPolicyMiddleware ────────────────────────────────────────


class TestRetryPolicyMiddleware:
    def test_defaults(self):
        mw = RetryPolicyMiddleware()
        assert mw.max_retries == 3
        assert RateLimitError in mw.retryable_errors
        assert ProviderError in mw.retryable_errors
        assert AgentTimeoutError in mw.retryable_errors

    def test_on_error_tracks_retryable(self):
        mw = RetryPolicyMiddleware()
        err = RateLimitError("limited")
        result = mw.on_error(_make_request(), err)
        assert result is err
        assert mw._retry_count == 1

    def test_on_error_ignores_non_retryable(self):
        mw = RetryPolicyMiddleware()
        err = ValueError("bad")
        mw.on_error(_make_request(), err)
        assert mw._retry_count == 0

    def test_custom_retryable_errors(self):
        mw = RetryPolicyMiddleware(retryable_errors=(ValueError,))
        mw.on_error(_make_request(), ValueError("bad"))
        assert mw._retry_count == 1


# ── MiddlewareChain ──────────────────────────────────────────────


class TestMiddlewareChain:
    def test_empty_chain(self):
        chain = MiddlewareChain()
        req = _make_request()
        assert chain.run_before(req) is req

    def test_add_returns_self(self):
        chain = MiddlewareChain()
        result = chain.add(Middleware())
        assert result is chain

    def test_before_runs_in_order(self):
        order = []

        class MW1(Middleware):
            def before(self, request):
                order.append("mw1")
                return request

        class MW2(Middleware):
            def before(self, request):
                order.append("mw2")
                return request

        chain = MiddlewareChain([MW1(), MW2()])
        chain.run_before(_make_request())
        assert order == ["mw1", "mw2"]

    def test_after_runs_in_reverse(self):
        order = []

        class MW1(Middleware):
            def after(self, request, response):
                order.append("mw1")
                return response

        class MW2(Middleware):
            def after(self, request, response):
                order.append("mw2")
                return response

        chain = MiddlewareChain([MW1(), MW2()])
        chain.run_after(_make_request(), _make_response())
        assert order == ["mw2", "mw1"]

    def test_before_modifies_request(self):
        class AddSystem(Middleware):
            def before(self, request):
                request.system = "injected"
                return request

        chain = MiddlewareChain([AddSystem()])
        req = _make_request()
        result = chain.run_before(req)
        assert result.system == "injected"

    def test_on_error_suppresses(self):
        class SuppressAll(Middleware):
            def on_error(self, request, error):
                return None

        chain = MiddlewareChain([SuppressAll()])
        result = chain.run_on_error(_make_request(), Exception("boom"))
        assert result is None

    def test_on_error_passes_through(self):
        chain = MiddlewareChain([Middleware()])
        err = Exception("boom")
        result = chain.run_on_error(_make_request(), err)
        assert result is err

    def test_on_error_chain_stops_on_suppress(self):
        call_count = 0

        class Counter(Middleware):
            def on_error(self, request, error):
                nonlocal call_count
                call_count += 1
                return error

        class Suppressor(Middleware):
            def on_error(self, request, error):
                return None

        chain = MiddlewareChain([Suppressor(), Counter()])
        result = chain.run_on_error(_make_request(), Exception("e"))
        assert result is None
        assert call_count == 0  # Counter never called because Suppressor is first

    def test_on_error_modifies_error(self):
        class Wrapper(Middleware):
            def on_error(self, request, error):
                return ValueError(f"wrapped: {error}")

        chain = MiddlewareChain([Wrapper()])
        result = chain.run_on_error(_make_request(), Exception("original"))
        assert isinstance(result, ValueError)
        assert "wrapped" in str(result)

    def test_chain_with_multiple_middlewares(self):
        metrics = MetricsMiddleware()
        logs = []
        logging = LoggingMiddleware(log_fn=logs.append)

        chain = MiddlewareChain([metrics, logging])
        req = _make_request("hello")
        chain.run_before(req)

        assert metrics.request_count == 1
        assert len(logs) == 1
