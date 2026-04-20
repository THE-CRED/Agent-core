"""
In-memory session store.
"""

from typing import Any

from agent.stores.base import SessionStore


class InMemoryStore(SessionStore):
    """
    In-memory session store.

    Stores sessions in memory. Data is lost when the process ends.
    Useful for development and testing.

    Example:
        ```python
        store = InMemoryStore()

        # Save session
        store.save("session-123", {"messages": [...], "system": "..."})

        # Load session
        data = store.load("session-123")
        ```
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Save session data."""
        self._sessions[session_id] = data

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session data."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete session data."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        return list(self._sessions.keys())

    def clear(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()

    def __len__(self) -> int:
        """Return number of stored sessions."""
        return len(self._sessions)
