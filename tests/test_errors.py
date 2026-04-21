"""Tests for agent error types."""

import pytest

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


class TestAgentError:
    def test_message_attr(self):
        e = AgentError("something broke")
        assert e.message == "something broke"
        assert str(e) == "something broke"

    def test_raw_default_none(self):
        e = AgentError("err")
        assert e.raw is None

    def test_raw_passthrough(self):
        raw = {"code": 500}
        e = AgentError("err", raw=raw)
        assert e.raw is raw

    def test_is_exception(self):
        assert issubclass(AgentError, Exception)


class TestAuthenticationError:
    def test_inherits_agent_error(self):
        e = AuthenticationError("bad key")
        assert isinstance(e, AgentError)
        assert e.message == "bad key"

    def test_raw_passthrough(self):
        e = AuthenticationError("bad", raw="detail")
        assert e.raw == "detail"


class TestProviderError:
    def test_attrs(self):
        e = ProviderError("fail", provider="openai", status_code=500, raw="x")
        assert e.message == "fail"
        assert e.provider == "openai"
        assert e.status_code == 500
        assert e.raw == "x"

    def test_defaults(self):
        e = ProviderError("fail")
        assert e.provider is None
        assert e.status_code is None

    def test_inherits_agent_error(self):
        assert issubclass(ProviderError, AgentError)


class TestRateLimitError:
    def test_attrs(self):
        e = RateLimitError("rate limited", provider="openai", retry_after=30.0, raw="r")
        assert e.retry_after == 30.0
        assert e.provider == "openai"
        assert e.raw == "r"

    def test_defaults(self):
        e = RateLimitError("limited")
        assert e.retry_after is None

    def test_inherits_provider_error(self):
        e = RateLimitError("limited")
        assert isinstance(e, ProviderError)
        assert isinstance(e, AgentError)


class TestTimeoutError:
    def test_attrs(self):
        e = TimeoutError("timed out", timeout=30.0, raw="t")
        assert e.timeout == 30.0
        assert e.raw == "t"

    def test_defaults(self):
        e = TimeoutError("timed out")
        assert e.timeout is None

    def test_inherits_agent_error(self):
        assert issubclass(TimeoutError, AgentError)

    def test_not_builtin_timeout(self):
        assert TimeoutError is not builtins_timeout()


class TestToolExecutionError:
    def test_attrs(self):
        e = ToolExecutionError("tool failed", tool_name="search", raw="x")
        assert e.tool_name == "search"
        assert e.raw == "x"

    def test_defaults(self):
        e = ToolExecutionError("failed")
        assert e.tool_name is None

    def test_inherits_agent_error(self):
        assert issubclass(ToolExecutionError, AgentError)


class TestSchemaValidationError:
    def test_attrs(self):
        e = SchemaValidationError("bad schema", schema={"type": "object"}, output="junk", raw="r")
        assert e.schema == {"type": "object"}
        assert e.output == "junk"
        assert e.raw == "r"

    def test_defaults(self):
        e = SchemaValidationError("bad")
        assert e.schema is None
        assert e.output is None

    def test_inherits_agent_error(self):
        assert issubclass(SchemaValidationError, AgentError)


class TestUnsupportedFeatureError:
    def test_attrs(self):
        e = UnsupportedFeatureError("no tools", feature="tools", provider="gemini", raw="u")
        assert e.feature == "tools"
        assert e.provider == "gemini"
        assert e.raw == "u"

    def test_defaults(self):
        e = UnsupportedFeatureError("unsupported")
        assert e.feature is None
        assert e.provider is None

    def test_inherits_agent_error(self):
        assert issubclass(UnsupportedFeatureError, AgentError)


class TestRoutingError:
    def test_attrs(self):
        errs = [Exception("a"), Exception("b")]
        e = RoutingError("all failed", errors=errs, raw="r")
        assert e.errors == errs
        assert len(e.errors) == 2
        assert e.raw == "r"

    def test_defaults_empty_list(self):
        e = RoutingError("failed")
        assert e.errors == []

    def test_none_errors_becomes_empty_list(self):
        e = RoutingError("failed", errors=None)
        assert e.errors == []

    def test_inherits_agent_error(self):
        assert issubclass(RoutingError, AgentError)


class TestInheritanceHierarchy:
    def test_rate_limit_is_provider_error(self):
        assert issubclass(RateLimitError, ProviderError)

    def test_authentication_is_not_provider_error(self):
        assert not issubclass(AuthenticationError, ProviderError)

    def test_timeout_is_not_provider_error(self):
        assert not issubclass(TimeoutError, ProviderError)

    def test_all_are_agent_errors(self):
        for cls in [
            AuthenticationError,
            ProviderError,
            RateLimitError,
            TimeoutError,
            ToolExecutionError,
            SchemaValidationError,
            UnsupportedFeatureError,
            RoutingError,
        ]:
            assert issubclass(cls, AgentError)

    def test_catch_agent_error_catches_all(self):
        for cls in [
            AuthenticationError,
            ProviderError,
            RateLimitError,
            TimeoutError,
            ToolExecutionError,
            SchemaValidationError,
            UnsupportedFeatureError,
            RoutingError,
        ]:
            with pytest.raises(AgentError):
                raise cls("test")


def builtins_timeout():
    import builtins

    return builtins.TimeoutError
