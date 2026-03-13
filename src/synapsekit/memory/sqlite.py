"""Persistent conversation memory backed by SQLite."""

from __future__ import annotations

import json
import sqlite3


class SQLiteConversationMemory:
    """Persistent conversation memory using SQLite.

    Messages survive process restarts. Supports multiple conversations
    via ``conversation_id``.

    Usage::

        memory = SQLiteConversationMemory(db_path="chat.db", conversation_id="user-1")
        memory.add("user", "Hello!")
        memory.add("assistant", "Hi there!")
        messages = memory.get_messages()  # persisted to disk

    """

    def __init__(
        self,
        db_path: str = "conversations.db",
        conversation_id: str = "default",
        window: int | None = None,
    ) -> None:
        self._db_path = db_path
        self._conversation_id = conversation_id
        self._window = window
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  conversation_id TEXT NOT NULL,"
            "  role TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  metadata TEXT"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_id ON messages(conversation_id)")
        self._conn.commit()

    def add(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Append a message to the conversation."""
        meta_json = json.dumps(metadata) if metadata else None
        self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (self._conversation_id, role, content, meta_json),
        )
        self._conn.commit()

        # Apply window if set
        if self._window is not None:
            max_messages = self._window * 2
            count = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (self._conversation_id,),
            ).fetchone()[0]
            if count > max_messages:
                self._conn.execute(
                    "DELETE FROM messages WHERE id IN ("
                    "  SELECT id FROM messages WHERE conversation_id = ? "
                    "  ORDER BY id ASC LIMIT ?"
                    ")",
                    (self._conversation_id, count - max_messages),
                )
                self._conn.commit()

    def get_messages(self) -> list[dict]:
        """Return all messages for this conversation."""
        rows = self._conn.execute(
            "SELECT role, content, metadata FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (self._conversation_id,),
        ).fetchall()
        messages = []
        for role, content, meta_json in rows:
            msg: dict = {"role": role, "content": content}
            if meta_json:
                msg["metadata"] = json.loads(meta_json)
            messages.append(msg)
        return messages

    def format_context(self) -> str:
        """Flatten history to a plain string for prompt injection."""
        parts = []
        for m in self.get_messages():
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Delete all messages for this conversation."""
        self._conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (self._conversation_id,),
        )
        self._conn.commit()

    def list_conversations(self) -> list[str]:
        """Return all conversation IDs in the database."""
        rows = self._conn.execute(
            "SELECT DISTINCT conversation_id FROM messages ORDER BY conversation_id"
        ).fetchall()
        return [r[0] for r in rows]

    def __len__(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (self._conversation_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
