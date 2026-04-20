"""
SQLite session store.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent.stores.base import SessionStore


class SQLiteStore(SessionStore):
    """
    SQLite-based session store.

    Persists sessions to a SQLite database file.

    Example:
        ```python
        store = SQLiteStore("sessions.db")

        # Save session
        store.save("session-123", {"messages": [...], "system": "..."})

        # Load session
        data = store.load("session-123")
        ```
    """

    def __init__(self, db_path: str | Path = "agent_sessions.db") -> None:
        """
        Initialize SQLite store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Save session data."""
        json_data = json.dumps(data)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, json_data),
            )
            conn.commit()

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session data."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def delete(self, session_id: str) -> bool:
        """Delete session data."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC")
            return [row[0] for row in cursor.fetchall()]

    def clear(self) -> None:
        """Clear all sessions."""
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions")
            conn.commit()

    def vacuum(self) -> None:
        """Reclaim disk space after deletions."""
        with self._connect() as conn:
            conn.execute("VACUUM")
