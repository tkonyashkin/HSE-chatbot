from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    user_query TEXT NOT NULL,
    assistant_reply TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_user_ts ON turns(user_id, ts DESC);
"""

HISTORY_REPLY_CHAR_LIMIT = 1500


@dataclass(frozen=True)
class Turn:
    user_query: str
    assistant_reply: str
    timestamp: datetime


class ConversationStore:
    def __init__(
        self,
        db_path: Path,
        max_messages: int = 5,
        ttl: timedelta = timedelta(minutes=30),
    ) -> None:
        self._db_path = Path(db_path)
        self._max_messages = max_messages
        self._ttl = ttl
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)

    def add(self, user_id: int, user_query: str, assistant_reply: str) -> None:
        now = datetime.now(UTC).isoformat()
        insert = "INSERT INTO turns (user_id, user_query, assistant_reply, ts) VALUES (?, ?, ?, ?)"
        trim = """
            DELETE FROM turns
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id FROM turns
                  WHERE user_id = ?
                  ORDER BY ts DESC, id DESC
                  LIMIT ?
              )
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(insert, (user_id, user_query, assistant_reply, now))
            conn.execute(trim, (user_id, user_id, self._max_messages))

    def get_recent(self, user_id: int) -> list[Turn]:
        cutoff = (datetime.now(UTC) - self._ttl).isoformat()
        select = (
            "SELECT user_query, assistant_reply, ts FROM turns WHERE user_id = ? ORDER BY ts ASC"
        )
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM turns WHERE user_id = ? AND ts < ?",
                (user_id, cutoff),
            )
            rows = conn.execute(select, (user_id,)).fetchall()
        return [
            Turn(
                user_query=row[0],
                assistant_reply=row[1],
                timestamp=datetime.fromisoformat(row[2]),
            )
            for row in rows
        ]

    def clear(self, user_id: int) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM turns WHERE user_id = ?", (user_id,))

    def render_context(self, user_id: int) -> str:
        turns = self.get_recent(user_id)
        if not turns:
            return ""
        lines = ["История диалога (новые сверху -> старые внизу):"]
        for t in reversed(turns):
            lines.append(f"[USER]: {t.user_query}")
            reply = t.assistant_reply.strip()
            if len(reply) > HISTORY_REPLY_CHAR_LIMIT:
                reply = reply[:HISTORY_REPLY_CHAR_LIMIT] + "…"
            lines.append(f"[ASSISTANT]: {reply}")
            lines.append("")
        return "\n".join(lines)
