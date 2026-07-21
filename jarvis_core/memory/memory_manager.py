import os
import sqlite3
import json
import time
from typing import Dict, Any, List, Optional

class ExperienceMemory:
    """SQLite-backed Experience Memory storing conversations, decisions, commands, errors, and fixes."""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                goal TEXT,
                decision TEXT,
                reason TEXT,
                files_modified TEXT,
                commands_executed TEXT,
                mistakes TEXT,
                fix_description TEXT,
                benchmark_data TEXT,
                timestamp REAL
            );
            """)
            conn.commit()

    def add_conversation(self, session_id: str, role: str, content: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time())
            )
            conn.commit()

    def log_experience(
        self,
        task_id: str,
        goal: str,
        decision: str,
        reason: str,
        files_modified: Optional[List[str]] = None,
        commands_executed: Optional[List[str]] = None,
        mistakes: Optional[str] = None,
        fix_description: Optional[str] = None,
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiences (
                    task_id, goal, decision, reason, files_modified, commands_executed, mistakes, fix_description, benchmark_data, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    goal,
                    decision,
                    reason,
                    json.dumps(files_modified or []),
                    json.dumps(commands_executed or []),
                    mistakes or "",
                    fix_description or "",
                    json.dumps(benchmark_data or {}),
                    time.time()
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_experiences(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
