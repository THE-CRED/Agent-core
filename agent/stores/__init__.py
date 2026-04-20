"""
Session storage backends.
"""

from agent.stores.base import SessionStore
from agent.stores.memory import InMemoryStore

__all__ = [
    "SessionStore",
    "InMemoryStore",
]

# Optional stores (may not be available)
try:
    from agent.stores.sqlite import SQLiteStore  # noqa: F401
    __all__.append("SQLiteStore")
except ImportError:
    pass
