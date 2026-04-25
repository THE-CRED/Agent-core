"""
Provider registry.

Manages registration and instantiation of provider adapters.
"""

import threading
from typing import Any

from agent.errors import ProviderError
from agent.providers.base import BaseProvider

_registry_lock = threading.Lock()


class ProviderRegistry:
    """Registry for provider adapters."""

    _providers: dict[str, type[BaseProvider]] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_class: type[BaseProvider],
        aliases: list[str] | None = None,
    ) -> None:
        """
        Register a provider adapter.

        Args:
            name: Provider name (e.g., "openai")
            provider_class: Provider class
            aliases: Optional list of aliases
        """
        cls._providers[name] = provider_class
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name

    @classmethod
    def get_class(cls, name: str) -> type[BaseProvider]:
        """
        Get a provider class by name.

        Args:
            name: Provider name or alias

        Returns:
            Provider class

        Raises:
            ProviderError: If provider not found
        """
        # Resolve alias
        resolved_name = cls._aliases.get(name, name)

        if resolved_name not in cls._providers:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ProviderError(
                f"Provider '{name}' not found. Available: {available}",
                provider=name,
            )

        return cls._providers[resolved_name]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseProvider:
        """
        Create a provider instance.

        Args:
            name: Provider name or alias
            **kwargs: Provider configuration

        Returns:
            Provider instance
        """
        provider_class = cls.get_class(name)
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return sorted(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered."""
        resolved_name = cls._aliases.get(name, name)
        return resolved_name in cls._providers


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """
    Get a provider instance.

    This is the main entry point for getting providers.
    Automatically loads provider modules on first access.

    Args:
        name: Provider name (e.g., "openai", "anthropic")
        **kwargs: Provider configuration

    Returns:
        Provider instance
    """
    # Lazy load providers
    _ensure_providers_loaded()

    return ProviderRegistry.create(name, **kwargs)


_providers_loaded = False


def _ensure_providers_loaded() -> None:
    """Ensure all built-in providers are loaded."""
    global _providers_loaded
    if _providers_loaded:
        return

    with _registry_lock:
        # Double-check after acquiring lock
        if _providers_loaded:
            return

        # Import provider modules to trigger registration
        import contextlib

        with contextlib.suppress(ImportError):
            from agent.providers import openai as _  # noqa: F401

        with contextlib.suppress(ImportError):
            from agent.providers import anthropic as _  # noqa: F401

        with contextlib.suppress(ImportError):
            from agent.providers import gemini as _  # noqa: F401

        with contextlib.suppress(ImportError):
            from agent.providers import deepseek as _  # noqa: F401

        _providers_loaded = True
