from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from .models import Message
from .utils import ensure_dir


class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        ensure_dir(os.path.dirname(db_path) or ".")
        self._lock = threading.Lock()
        with self._conn_ctx() as con:
            con.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS templates (
                    name TEXT PRIMARY KEY,
                    template TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    system TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    convo_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta_json TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (convo_id, seq),
                    FOREIGN KEY (convo_id) REFERENCES conversations(id)
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    meta_json TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector_json TEXT,
                    meta_json TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                """
            )

    @contextmanager
    def _conn_ctx(self) -> sqlite3.Connection:
        """Open a connection, guarantee commit/rollback and CLOSE."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            try:
                con.commit()
            except Exception:
                # Some read-only paths won't need commit; ignore
                pass
        except Exception:
            try:
                con.rollback()
            finally:
                con.close()
            raise
        else:
            con.close()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def upsert_template(self, name: str, template: str) -> None:
        now = time.time()
        with self._lock, self._conn_ctx() as con:
            con.execute(
                """
                INSERT INTO templates(name, template, created_at, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    template=excluded.template,
                    updated_at=excluded.updated_at
                """,
                (name, template, now, now),
            )

    def get_template(self, name: str) -> Optional[str]:
        with self._lock, self._conn_ctx() as con:
            row = con.execute("SELECT template FROM templates WHERE name=?", (name,)).fetchone()
            return row[0] if row else None

    def list_templates(self) -> Dict[str, str]:
        with self._lock, self._conn_ctx() as con:
            rows = con.execute("SELECT name, template FROM templates ORDER BY name").fetchall()
            return {r[0]: r[1] for r in rows}

    def get_embedding(self, key: str) -> Optional[List[float]]:
        with self._lock, self._conn_ctx() as con:
            row = con.execute("SELECT vector_json FROM embeddings WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def put_embedding(self, key: str, model: str, text_hash: str, vector: List[float]) -> None:
        now = time.time()
        with self._lock, self._conn_ctx() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO embeddings(key, model, text_hash, vector_json, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (key, model, text_hash, json.dumps(vector), now),
            )

    def new_conversation(self, name: Optional[str], system: Optional[str]) -> str:
        cid = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._conn_ctx() as con:
            con.execute(
                "INSERT INTO conversations(id, name, system, created_at) VALUES (?, ?, ?, ?)",
                (cid, name, system, now),
            )
        return cid

    def append_message(
        self,
        convo_id: str,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta_json = json.dumps(meta or {})
        now = time.time()
        with self._lock, self._conn_ctx() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(seq), -1) AS m FROM messages WHERE convo_id=?",
                (convo_id,),
            ).fetchone()
            next_seq = int(row[0]) + 1
            con.execute(
                """
                INSERT INTO messages(convo_id, seq, role, content, meta_json, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (convo_id, next_seq, role, content, meta_json, now),
            )
        return next_seq

    def get_history(self, convo_id: str) -> List[Message]:
        with self._lock, self._conn_ctx() as con:
            rows = con.execute(
                "SELECT role, content FROM messages WHERE convo_id=? ORDER BY seq ASC",
                (convo_id,),
            ).fetchall()
        return [Message(role=r[0], content=r[1]) for r in rows]

    def get_conversation_system(self, convo_id: str) -> Optional[str]:
        with self._lock, self._conn_ctx() as con:
            row = con.execute("SELECT system FROM conversations WHERE id=?", (convo_id,)).fetchone()
            return row[0] if row else None

    def upsert_document(self, source: str, meta: Optional[Dict[str, Any]] = None) -> str:
        did = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._conn_ctx() as con:
            con.execute(
                "INSERT INTO documents(id, source, meta_json, created_at) VALUES (?, ?, ?, ?)",
                (did, source, json.dumps(meta or {}), now),
            )
        return did

    def insert_chunk(
        self,
        doc_id: str,
        idx: int,
        text: str,
        vector: Optional[List[float]],
        meta: Optional[Dict[str, Any]],
    ):
        cid = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._conn_ctx() as con:
            con.execute(
                """
                INSERT INTO chunks(id, doc_id, idx, text, vector_json, meta_json, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    doc_id,
                    idx,
                    text,
                    json.dumps(vector) if vector is not None else None,
                    json.dumps(meta or {}),
                    now,
                ),
            )
        return cid

    def iter_chunks(self) -> Iterable[sqlite3.Row]:
        # Read fully, then yield after the connection is closed so we don't leak
        # a live connection if the caller breaks iteration early.
        with self._lock, self._conn_ctx() as con:
            rows = con.execute(
                "SELECT id, doc_id, idx, text, vector_json, meta_json FROM chunks"
            ).fetchall()
        for row in rows:
            yield row

    def update_chunk_vector(self, chunk_id: str, vector: List[float]) -> None:
        with self._lock, self._conn_ctx() as con:
            con.execute(
                "UPDATE chunks SET vector_json=? WHERE id=?",
                (json.dumps(vector), chunk_id),
            )

    # helpers for re-ingest/versioning
    def get_latest_document_by_source(self, source: str):
        with self._lock, self._conn_ctx() as con:
            row = con.execute(
                "SELECT id, meta_json FROM documents WHERE source=? " \
                "ORDER BY created_at DESC LIMIT 1",
                (source,),
            ).fetchone()
        if not row:
            return None
        meta = json.loads(row[1] or "{}")
        return row[0], meta

    def delete_chunks_for_doc(self, doc_id: str) -> None:
        with self._lock, self._conn_ctx() as con:
            con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

    def update_document_meta(self, doc_id: str, meta: dict) -> None:
        with self._lock, self._conn_ctx() as con:
            con.execute(
                "UPDATE documents SET meta_json=? WHERE id=?",
                (json.dumps(meta), doc_id),
            )

    def get_chunk_hash_map(self, doc_id: str) -> dict:
        out = {}
        with self._lock, self._conn_ctx() as con:
            for r in con.execute("SELECT id, meta_json FROM chunks WHERE doc_id=?", (doc_id,)):
                m = json.loads(r[1] or "{}")
                ch = m.get("chunk_hash")
                if ch:
                    out[ch] = r[0]
        return out
