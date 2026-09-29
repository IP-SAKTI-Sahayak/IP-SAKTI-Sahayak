"""SQLite persistence for authenticated users and facilitator escalations."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("IP_SAKTI_DATABASE", ROOT / "data" / "runtime" / "ip_sakti.sqlite3"))
VALID_ROLES = {"user", "facilitator", "admin"}
VALID_STATUSES = {"new", "in_progress", "answered"}


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK(role IN ('user', 'facilitator', 'admin')),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                question TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                contact_email TEXT NOT NULL DEFAULT '',
                assistant_response TEXT NOT NULL DEFAULT '',
                jurisdiction TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new'
                    CHECK(status IN ('new', 'in_progress', 'answered')),
                facilitator_reply TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS escalations_user_id_idx ON escalations(user_id);
            CREATE INDEX IF NOT EXISTS escalations_status_idx ON escalations(status);
            """
        )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def public_user(row):
    if row is None:
        return None
    return {"id": row["id"], "email": row["email"], "role": row["role"], "created_at": row["created_at"]}


def create_user(email, password_hash, role="user"):
    if role not in VALID_ROLES:
        raise ValueError("Invalid role")
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO users(email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (email.strip().lower(), password_hash, role, utc_now()),
        )
        row = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return public_user(row)


def update_user_credentials(email, password_hash, role):
    """Used only by the local interactive seed command."""
    if role not in VALID_ROLES:
        raise ValueError("Invalid role")
    with connect() as connection:
        row = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            cursor = connection.execute(
                "INSERT INTO users(email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (email.strip().lower(), password_hash, role, utc_now()),
            )
            row_id = cursor.lastrowid
        else:
            row_id = row["id"]
            connection.execute(
                "UPDATE users SET password_hash = ?, role = ? WHERE id = ?",
                (password_hash, role, row_id),
            )
        return public_user(connection.execute("SELECT * FROM users WHERE id = ?", (row_id,)).fetchone())


def get_user_by_email(email):
    with connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id):
    with connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_escalation(*, user_id, question, details, contact_email, assistant_response, jurisdiction, timestamp=None):
    now = timestamp or utc_now()
    with connect() as connection:
        cursor = connection.execute(
            """INSERT INTO escalations
               (user_id, question, details, contact_email, assistant_response, jurisdiction, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?)""",
            (user_id, question, details, contact_email, assistant_response, jurisdiction, now, now),
        )
        return cursor.lastrowid


def _escalation_query(where="", values=()):
    with connect() as connection:
        rows = connection.execute(
            "SELECT e.*, u.email AS account_email FROM escalations e JOIN users u ON u.id=e.user_id "
            + where + " ORDER BY CASE e.status WHEN 'new' THEN 0 WHEN 'in_progress' THEN 1 ELSE 2 END, e.created_at DESC",
            values,
        ).fetchall()
        return [dict(row) for row in rows]


def list_escalations():
    return _escalation_query()


def list_user_escalations(user_id):
    return _escalation_query("WHERE e.user_id = ?", (user_id,))


def get_escalation(escalation_id):
    rows = _escalation_query("WHERE e.id = ?", (escalation_id,))
    return rows[0] if rows else None


def update_escalation_status(escalation_id, status):
    if status not in VALID_STATUSES:
        raise ValueError("Invalid status")
    with connect() as connection:
        connection.execute(
            "UPDATE escalations SET status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now(), escalation_id),
        )
    return get_escalation(escalation_id)


def answer_escalation(escalation_id, reply):
    with connect() as connection:
        connection.execute(
            "UPDATE escalations SET facilitator_reply = ?, status = 'answered', updated_at = ? WHERE id = ?",
            (reply, utc_now(), escalation_id),
        )
    return get_escalation(escalation_id)
