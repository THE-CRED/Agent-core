"""
Base session store interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class SessionStore(ABC):
    """
    Abstract base class for session storage.

    Session stores persist conversation history across sessions.
    """

    @abstractmethod
    def save(self, session_id: str, data: dict[str, Any]) -> None:
        """
        Save session data.

        Args:
            session_id: Unique session identifier
            data: Session data to save
        """
        ...

    @abstractmethod
    def load(self, session_id: str) -> dict[str, Any] | None:
        """
        Load session data.

        Args:
            session_id: Unique session identifier

        Returns:
            Session data if found, None otherwise
        """
        ...

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """
        Delete session data.

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """
        List all session IDs.

        Returns:
            List of session identifiers
        """
        ...

    def exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Unique session identifier

        Returns:
            True if session exists
        """
        return self.load(session_id) is not None
