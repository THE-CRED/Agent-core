"""Tests for agent.cli.main module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.cli.main import cmd_doctor, cmd_providers, cmd_run, main

# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_no_args_prints_help_and_returns_zero(self, capsys):
        """Calling main with no arguments prints help and returns 0."""
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "Agent - Multi-provider LLM runtime" in captured.out

    def test_version_flag_exits_with_version(self):
        """--version triggers SystemExit and prints version string."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_version_output(self, capsys):
        """--version prints the version string to stdout."""
        with pytest.raises(SystemExit):
            main(["--version"])
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out

    def test_unknown_command_exits_nonzero(self):
        """An unrecognized subcommand causes argparse to exit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent_command"])
        assert exc_info.value.code == 2

    def test_run_subcommand_dispatches_to_cmd_run(self):
        """main(['run', ...]) calls cmd_run and returns its result."""
        with patch("agent.cli.main.cmd_run", return_value=0) as mock_cmd:
            result = main(["run", "hello", "-p", "openai"])
        assert result == 0
        mock_cmd.assert_called_once()
        parsed_args = mock_cmd.call_args[0][0]
        assert parsed_args.prompt == "hello"
        assert parsed_args.provider == "openai"

    def test_chat_subcommand_dispatches_to_cmd_chat(self):
        """main(['chat', ...]) calls cmd_chat and returns its result."""
        with patch("agent.cli.main.cmd_chat", return_value=0) as mock_cmd:
            result = main(["chat", "-p", "anthropic"])
        assert result == 0
        mock_cmd.assert_called_once()
        parsed_args = mock_cmd.call_args[0][0]
        assert parsed_args.provider == "anthropic"

    def test_providers_subcommand_dispatches_to_cmd_providers(self):
        """main(['providers']) calls cmd_providers and returns its result."""
        with patch("agent.cli.main.cmd_providers", return_value=0) as mock_cmd:
            result = main(["providers"])
        assert result == 0
        mock_cmd.assert_called_once()

    def test_doctor_subcommand_dispatches_to_cmd_doctor(self):
        """main(['doctor']) calls cmd_doctor and returns its result."""
        with patch("agent.cli.main.cmd_doctor", return_value=0) as mock_cmd:
            result = main(["doctor"])
        assert result == 0
        mock_cmd.assert_called_once()

    def test_run_parser_defaults(self):
        """Run subcommand has correct default values for optional args."""
        with patch("agent.cli.main.cmd_run", return_value=0) as mock_cmd:
            main(["run", "test prompt"])
        parsed_args = mock_cmd.call_args[0][0]
        assert parsed_args.provider == "openai"
        assert parsed_args.model is None
        assert parsed_args.system is None
        assert parsed_args.temperature is None
        assert parsed_args.stream is False
        assert parsed_args.json is False

    def test_run_parser_all_flags(self):
        """Run subcommand correctly parses all flags."""
        with patch("agent.cli.main.cmd_run", return_value=0) as mock_cmd:
            main([
                "run", "prompt text",
                "-p", "anthropic",
                "-m", "claude-sonnet-4-20250514",
                "-s", "You are helpful",
                "-t", "0.7",
                "--stream",
                "--json",
            ])
        parsed_args = mock_cmd.call_args[0][0]
        assert parsed_args.prompt == "prompt text"
        assert parsed_args.provider == "anthropic"
        assert parsed_args.model == "claude-sonnet-4-20250514"
        assert parsed_args.system == "You are helpful"
        assert parsed_args.temperature == 0.7
        assert parsed_args.stream is True
        assert parsed_args.json is True

    def test_chat_parser_defaults(self):
        """Chat subcommand has correct default values."""
        with patch("agent.cli.main.cmd_chat", return_value=0) as mock_cmd:
            main(["chat"])
        parsed_args = mock_cmd.call_args[0][0]
        assert parsed_args.provider == "openai"
        assert parsed_args.model is None
        assert parsed_args.system is None


# ---------------------------------------------------------------------------
# cmd_run()
# ---------------------------------------------------------------------------


class TestCmdRun:
    """Tests for cmd_run."""

    def _make_args(self, **overrides):
        defaults = {
            "provider": "openai",
            "model": None,
            "system": None,
            "temperature": None,
            "stream": False,
            "json": False,
            "prompt": "Say hello",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @patch("agent.cli.main.Agent", create=True)
    def test_run_normal_response(self, mock_agent_cls, capsys):
        """cmd_run prints response text for non-streaming call."""
        mock_response = MagicMock()
        mock_response.text = "Hello there!"
        mock_response.usage = None
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = mock_response

        # Patch the import inside cmd_run
        with patch.dict("sys.modules", {"agent": MagicMock(Agent=MagicMock(return_value=mock_agent_instance))}):
            args = self._make_args()
            result = cmd_run(args)

        assert result == 0

    @patch("agent.cli.main.Agent", create=True)
    def test_run_with_explicit_model(self, mock_agent_cls, capsys):
        """cmd_run uses explicit model when provided."""
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_response.usage = None
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_response

        with patch.dict("sys.modules", {"agent": MagicMock(Agent=MagicMock(return_value=mock_agent))}):
            args = self._make_args(model="gpt-4o-mini")
            result = cmd_run(args)

        assert result == 0

    def test_run_default_model_per_provider(self):
        """cmd_run sets default model based on provider when model is None."""
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "gemini": "gemini-1.5-pro",
            "deepseek": "deepseek-chat",
        }
        for provider, expected_model in defaults.items():
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "ok"
            mock_response.usage = None
            mock_agent.run.return_value = mock_response
            mock_agent_cls = MagicMock(return_value=mock_agent)

            with patch.dict(
                "sys.modules",
                {"agent": MagicMock(Agent=mock_agent_cls)},
            ):
                args = self._make_args(provider=provider, model=None)
                cmd_run(args)

            # Agent was called with the expected default model
            call_kwargs = mock_agent_cls.call_args[1]
            assert call_kwargs["model"] == expected_model, (
                f"Provider {provider}: expected {expected_model}, got {call_kwargs['model']}"
            )

    def test_run_unknown_provider_uses_fallback_model(self):
        """cmd_run falls back to gpt-4o for unknown providers."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage = None
        mock_agent.run.return_value = mock_response
        mock_agent_cls = MagicMock(return_value=mock_agent)

        with patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=mock_agent_cls)},
        ):
            args = self._make_args(provider="unknown_provider", model=None)
            cmd_run(args)

        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_run_with_usage_info(self, capsys):
        """cmd_run prints token usage to stderr when available."""
        mock_usage = MagicMock()
        mock_usage.total_tokens = 100
        mock_usage.prompt_tokens = 80
        mock_usage.completion_tokens = 20
        mock_response = MagicMock()
        mock_response.text = "Hello"
        mock_response.usage = mock_usage
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=MagicMock(return_value=mock_agent))},
        ):
            args = self._make_args()
            result = cmd_run(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "Tokens: 100" in captured.err
        assert "prompt: 80" in captured.err
        assert "completion: 20" in captured.err

    def test_run_streaming(self, capsys):
        """cmd_run streams text deltas when --stream is set."""
        event1 = MagicMock()
        event1.type = "text_delta"
        event1.text = "Hello "
        event2 = MagicMock()
        event2.type = "text_delta"
        event2.text = "world"

        mock_agent = MagicMock()
        mock_agent.stream.return_value = [event1, event2]

        with patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=MagicMock(return_value=mock_agent))},
        ):
            args = self._make_args(stream=True)
            result = cmd_run(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Hello " in captured.out
        assert "world" in captured.out

    def test_run_with_temperature(self):
        """cmd_run passes temperature kwarg when set."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage = None
        mock_agent.run.return_value = mock_response
        mock_agent_cls = MagicMock(return_value=mock_agent)

        with patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=mock_agent_cls)},
        ):
            args = self._make_args(temperature=0.5)
            cmd_run(args)

        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_run_agent_exception_returns_one(self, capsys):
        """cmd_run returns 1 when Agent raises an exception."""
        mock_agent_cls = MagicMock(side_effect=RuntimeError("API failure"))

        with patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=mock_agent_cls)},
        ):
            args = self._make_args()
            result = cmd_run(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "API failure" in captured.err

    def test_run_passes_system_prompt(self):
        """cmd_run passes system prompt to agent.run."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage = None
        mock_agent.run.return_value = mock_response

        with patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=MagicMock(return_value=mock_agent))},
        ):
            args = self._make_args(system="Be concise")
            cmd_run(args)

        mock_agent.run.assert_called_once_with(
            "Say hello", system="Be concise"
        )


# ---------------------------------------------------------------------------
# cmd_providers()
# ---------------------------------------------------------------------------


class TestCmdProviders:
    """Tests for cmd_providers."""

    @patch("agent.cli.main._ensure_providers_loaded", create=True)
    @patch("agent.cli.main.ProviderRegistry", create=True)
    def test_providers_returns_zero(self, mock_registry, mock_ensure, capsys):
        """cmd_providers returns 0 and prints header."""
        mock_registry.list_providers.return_value = []
        # Need to also patch the import inside the function
        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry"
        ) as inner_registry:
            inner_registry.list_providers.return_value = []
            result = cmd_providers(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "Available providers:" in captured.out

    def test_providers_lists_features(self, capsys):
        """cmd_providers lists provider features correctly."""
        mock_caps = MagicMock()
        mock_caps.streaming = True
        mock_caps.tools = True
        mock_caps.structured_output = False
        mock_caps.vision = True

        mock_provider_cls = MagicMock()
        mock_provider_cls.capabilities = mock_caps

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["test_provider"]
        mock_registry.get_class.return_value = mock_provider_cls

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ):
            result = cmd_providers(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "test_provider" in captured.out
        assert "streaming" in captured.out
        assert "tools" in captured.out
        assert "vision" in captured.out
        assert "structured" not in captured.out

    def test_providers_handles_error_loading(self, capsys):
        """cmd_providers gracefully handles errors when loading a provider."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["broken"]
        mock_registry.get_class.side_effect = RuntimeError("bad import")

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ):
            result = cmd_providers(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "broken" in captured.out
        assert "error loading" in captured.out


# ---------------------------------------------------------------------------
# cmd_doctor()
# ---------------------------------------------------------------------------


class TestCmdDoctor:
    """Tests for cmd_doctor."""

    def test_doctor_returns_zero(self, capsys):
        """cmd_doctor always returns 0."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", {}
        ), patch.dict(
            "os.environ", {}, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "Agent Doctor" in captured.out
        assert "Doctor check complete." in captured.out

    def test_doctor_shows_set_api_keys(self, capsys):
        """cmd_doctor shows masked API keys for set env vars."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []

        env_vars = {"openai": "OPENAI_API_KEY"}
        fake_env = {"OPENAI_API_KEY": "sk-1234567890abcdefghijklmn"}

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", env_vars
        ), patch.dict(
            "os.environ", fake_env, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.out
        assert "sk-12345..." in captured.out
        assert "klmn" in captured.out
        # Full key should NOT appear
        assert "sk-1234567890abcdefghijklmn" not in captured.out

    def test_doctor_shows_unset_api_keys(self, capsys):
        """cmd_doctor shows '(not set)' for missing env vars."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []

        env_vars = {"anthropic": "ANTHROPIC_API_KEY"}

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", env_vars
        ), patch.dict(
            "os.environ", {}, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY" in captured.out
        assert "(not set)" in captured.out

    def test_doctor_shows_provider_status_ok(self, capsys):
        """cmd_doctor shows OK for providers that load fine."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["openai"]
        mock_registry.get_class.return_value = MagicMock()

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", {}
        ), patch.dict(
            "os.environ", {}, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "openai: OK" in captured.out

    def test_doctor_shows_provider_import_error(self, capsys):
        """cmd_doctor shows missing dependency for ImportError."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["broken"]
        mock_registry.get_class.side_effect = ImportError("no module named x")

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", {}
        ), patch.dict(
            "os.environ", {}, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "broken: Missing dependency" in captured.out

    def test_doctor_shows_provider_generic_error(self, capsys):
        """cmd_doctor shows Error for non-ImportError exceptions."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["bad"]
        mock_registry.get_class.side_effect = RuntimeError("something broke")

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", {}
        ), patch.dict(
            "os.environ", {}, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "bad: Error" in captured.out

    def test_doctor_skips_connection_when_no_key(self, capsys):
        """cmd_doctor skips connection test when API key is not set."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["openai"]
        mock_registry.get_class.return_value = MagicMock()

        env_vars = {"openai": "OPENAI_API_KEY"}

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", env_vars
        ), patch.dict(
            "os.environ", {}, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "Skipped (no API key)" in captured.out

    def test_doctor_connection_test_success(self, capsys):
        """cmd_doctor shows success when connection test passes."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["openai"]
        mock_registry.get_class.return_value = MagicMock()

        env_vars = {"openai": "OPENAI_API_KEY"}
        fake_env = {"OPENAI_API_KEY": "sk-testkey1234567890"}

        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_response

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", env_vars
        ), patch.dict(
            "os.environ", fake_env, clear=True
        ), patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=MagicMock(return_value=mock_agent))},
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "openai: OK" in captured.out

    def test_doctor_connection_test_failure(self, capsys):
        """cmd_doctor shows failure when connection test raises."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["openai"]
        mock_registry.get_class.return_value = MagicMock()

        env_vars = {"openai": "OPENAI_API_KEY"}
        fake_env = {"OPENAI_API_KEY": "sk-testkey1234567890"}

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", env_vars
        ), patch.dict(
            "os.environ", fake_env, clear=True
        ), patch.dict(
            "sys.modules",
            {"agent": MagicMock(Agent=MagicMock(side_effect=RuntimeError("conn refused")))},
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "Failed" in captured.out

    def test_doctor_short_api_key_fully_masked(self, capsys):
        """cmd_doctor masks short API keys entirely."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []

        env_vars = {"openai": "OPENAI_API_KEY"}
        fake_env = {"OPENAI_API_KEY": "short"}

        with patch(
            "agent.providers.registry._ensure_providers_loaded"
        ), patch(
            "agent.providers.registry.ProviderRegistry", mock_registry
        ), patch(
            "agent.config.ENV_VARS", env_vars
        ), patch.dict(
            "os.environ", fake_env, clear=True
        ):
            result = cmd_doctor(SimpleNamespace())

        assert result == 0
        captured = capsys.readouterr()
        assert "***" in captured.out
        assert "short" not in captured.out
