"""
Agent CLI.

Provides commands for quick experimentation and development.
"""

import argparse
import sys
from typing import Any


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agent - Multi-provider LLM runtime",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single prompt")
    run_parser.add_argument("prompt", help="The prompt to send")
    run_parser.add_argument(
        "-p", "--provider",
        default="openai",
        help="Provider to use (default: openai)",
    )
    run_parser.add_argument(
        "-m", "--model",
        help="Model to use (default: provider default)",
    )
    run_parser.add_argument(
        "-s", "--system",
        help="System prompt",
    )
    run_parser.add_argument(
        "-t", "--temperature",
        type=float,
        help="Sampling temperature",
    )
    run_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Request JSON output",
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "-p", "--provider",
        default="openai",
        help="Provider to use (default: openai)",
    )
    chat_parser.add_argument(
        "-m", "--model",
        help="Model to use",
    )
    chat_parser.add_argument(
        "-s", "--system",
        help="System prompt",
    )

    # Providers command
    subparsers.add_parser("providers", help="List available providers")

    # Doctor command
    subparsers.add_parser("doctor", help="Check configuration and credentials")

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    if parsed.command == "run":
        return cmd_run(parsed)
    elif parsed.command == "chat":
        return cmd_chat(parsed)
    elif parsed.command == "providers":
        return cmd_providers(parsed)
    elif parsed.command == "doctor":
        return cmd_doctor(parsed)

    return 0


def cmd_run(args: Any) -> int:
    """Execute a single prompt."""
    try:
        from agent import Agent
    except ImportError as e:
        print(f"Error importing agent: {e}", file=sys.stderr)
        return 1

    # Build kwargs
    kwargs: dict[str, Any] = {"provider": args.provider}
    if args.model:
        kwargs["model"] = args.model
    else:
        # Default models per provider
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "gemini": "gemini-1.5-pro",
            "deepseek": "deepseek-chat",
        }
        kwargs["model"] = defaults.get(args.provider, "gpt-4o")

    if args.temperature:
        kwargs["temperature"] = args.temperature

    try:
        agent = Agent(**kwargs)

        if args.stream:
            # Stream response
            for event in agent.stream(args.prompt, system=args.system):
                if event.type == "text_delta" and event.text:
                    print(event.text, end="", flush=True)
            print()  # Newline at end
        else:
            # Normal response
            response = agent.run(args.prompt, system=args.system)
            print(response.text)

            # Show usage if available
            if response.usage:
                print(
                    f"\n[Tokens: {response.usage.total_tokens} "
                    f"(prompt: {response.usage.prompt_tokens}, "
                    f"completion: {response.usage.completion_tokens})]",
                    file=sys.stderr,
                )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_chat(args: Any) -> int:
    """Start an interactive chat session."""
    try:
        from agent import Agent
    except ImportError as e:
        print(f"Error importing agent: {e}", file=sys.stderr)
        return 1

    # Build kwargs
    kwargs: dict[str, Any] = {"provider": args.provider}
    if args.model:
        kwargs["model"] = args.model
    else:
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "gemini": "gemini-1.5-pro",
            "deepseek": "deepseek-chat",
        }
        kwargs["model"] = defaults.get(args.provider, "gpt-4o")

    try:
        agent = Agent(**kwargs)
        session = agent.session(system=args.system)

        print(f"Agent Chat ({args.provider}/{kwargs['model']})")
        print("Type 'quit' or 'exit' to end the session.")
        print("Type 'clear' to clear history.")
        print("-" * 40)

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                session.clear()
                print("[History cleared]")
                continue

            try:
                # Stream the response
                print("\nAssistant: ", end="", flush=True)
                stream = session.stream(user_input)
                for event in stream:
                    if event.type == "text_delta" and event.text:
                        print(event.text, end="", flush=True)
                print()

            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_providers(args: Any) -> int:
    """List available providers."""
    from agent.providers.registry import ProviderRegistry, _ensure_providers_loaded

    _ensure_providers_loaded()

    providers = ProviderRegistry.list_providers()

    print("Available providers:")
    print("-" * 40)

    for name in providers:
        try:
            provider_class = ProviderRegistry.get_class(name)
            caps = provider_class.capabilities

            features = []
            if caps.streaming:
                features.append("streaming")
            if caps.tools:
                features.append("tools")
            if caps.structured_output:
                features.append("structured")
            if caps.vision:
                features.append("vision")

            print(f"  {name}: {', '.join(features)}")
        except Exception as e:
            print(f"  {name}: (error loading: {e})")

    return 0


def cmd_doctor(args: Any) -> int:
    """Check configuration and credentials."""
    import os

    from agent.config import ENV_VARS
    from agent.providers.registry import ProviderRegistry, _ensure_providers_loaded

    _ensure_providers_loaded()

    print("Agent Doctor")
    print("=" * 40)

    # Check environment variables
    print("\nAPI Keys:")
    print("-" * 40)

    for provider, env_var in ENV_VARS.items():
        value = os.environ.get(env_var)
        if value:
            # Show masked key
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  {provider}: {env_var} = {masked}")
        else:
            print(f"  {provider}: {env_var} = (not set)")

    # Check provider imports
    print("\nProvider Status:")
    print("-" * 40)

    providers = ProviderRegistry.list_providers()

    for name in providers:
        try:
            ProviderRegistry.get_class(name)
            print(f"  {name}: OK")
        except ImportError as e:
            print(f"  {name}: Missing dependency ({e})")
        except Exception as e:
            print(f"  {name}: Error ({e})")

    # Test connections
    print("\nConnection Tests:")
    print("-" * 40)

    for name in providers:
        env_var = ENV_VARS.get(name)
        if not env_var or not os.environ.get(env_var):
            print(f"  {name}: Skipped (no API key)")
            continue

        try:
            from agent import Agent

            # Get default model
            defaults = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-haiku-20241022",
                "gemini": "gemini-1.5-flash",
                "deepseek": "deepseek-chat",
            }
            model = defaults.get(name, "gpt-4o-mini")

            agent = Agent(provider=name, model=model)
            response = agent.run("Say 'OK' and nothing else.", max_tokens=10)

            if response.text:
                print(f"  {name}: OK (response: {response.text[:20]}...)")
            else:
                print(f"  {name}: OK (empty response)")

        except Exception as e:
            print(f"  {name}: Failed ({e})")

    print("\n" + "=" * 40)
    print("Doctor check complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
