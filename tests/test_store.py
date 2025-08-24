import json
import sqlite3

import pytest

import corpusloom.store as store_mod
from corpusloom.store import Store


def test_template_roundtrip(tmp_path):
    db = tmp_path / "db.sqlite"
    s = Store(str(db))
    s.upsert_template("t", "Hello {x}")
    assert s.get_template("t") == "Hello {x}"
    all_names = s.list_templates()
    assert "t" in all_names


def test__conn_ctx_rollback_branch(tmp_path):
    """Exercise the EXCEPT path in _conn_ctx: rollback + close + re-raise."""
    db = tmp_path / "db.sqlite"
    s = Store(str(db))
    # Raise inside the with-block so the context manager handles it.
    try:
        with s._conn_ctx() as con:  # noqa: SLF001 (private OK for white-box test)
            con.execute("THIS IS NOT VALID SQL")  # triggers sqlite3.OperationalError
    except sqlite3.OperationalError:
        pass
    else:
        raise AssertionError("Expected sqlite3.OperationalError not raised")

    # After the exception, a fresh context must work (prior connection closed).
    with s._conn_ctx() as con:
        con.execute("SELECT 1")


def test__conn_ctx_commit_exception_is_suppressed(tmp_path):
    """Exercise the inner commit() except path (commit raises, but is swallowed)."""
    db = tmp_path / "db.sqlite"
    s = Store(str(db))
    # Close the connection before context exit to make commit() raise.
    with s._conn_ctx() as con:  # noqa: SLF001
        con.execute("SELECT 1")
        con.close()  # forces commit() to raise on exit; should be swallowed
    # If we got here without an exception, the branch is covered.


def test_get_latest_document_by_source_none(tmp_path):
    """Hit the 'no row' branch returning None."""
    db = tmp_path / "db.sqlite"
    s = Store(str(db))
    assert s.get_latest_document_by_source("missing-source") is None


def test_get_chunk_hash_map_has_and_missing_keys(tmp_path):
    """Cover both sides of 'if ch:' inside the loop."""
    db = tmp_path / "db.sqlite"
    s = Store(str(db))
    doc_id = s.upsert_document("src.txt", meta={"v": 1})

    # Chunk without a chunk_hash (should be skipped)
    s.insert_chunk(doc_id, idx=0, text="alpha", vector=None, meta={"note": "nohash"})

    # Chunk with a chunk_hash (should be included)
    s.insert_chunk(doc_id, idx=1, text="beta", vector=None, meta={"chunk_hash": "H123"})

    cmap = s.get_chunk_hash_map(doc_id)
    # Only the hashed one should appear
    assert "H123" in cmap and isinstance(cmap["H123"], str)
    assert len(cmap) == 1


def test__conn_ctx_commit_raises_is_swallowed_and_closes(monkeypatch, tmp_path):
    """Force commit() to raise to cover the inner except and the final close() in the else branch."""
    # Build a real Store first so __init__ can create schema using the real sqlite3.
    s = Store(str((tmp_path / "db.sqlite").resolve()))

    # Fake connection object returned only for THIS test's _conn_ctx invocation.
    class FakeConn:
        def __init__(self):
            self.row_factory = None
            self.closed = False
            self.commit_called = False
            self.rolled_back = False

        # Minimal API used by _conn_ctx
        def execute(self, *a, **k):  # not used here, but harmless
            return None

        def commit(self):
            self.commit_called = True
            raise RuntimeError("commit boom")

        def rollback(self):
            self.rolled_back = True  # should NOT be called in this path

        def close(self):
            self.closed = True

    fake = FakeConn()

    # Monkeypatch connect INSIDE the corpusloom.store module
    monkeypatch.setattr(store_mod.sqlite3, "connect", lambda path: fake)

    # Now run the context manager: commit() will raise, inner except swallows, then else: close()
    with s._conn_ctx() as con:
        pass

    assert fake.commit_called is True
    assert fake.rolled_back is False
    assert fake.closed is True


def test_conn_ctx_normal_exit_executes_close(tmp_path):
    """
    Exercise the 'normal' path of _conn_ctx:
      - yield returns
      - commit() succeeds
      - outer 'else:' block runs and close() is called
    We don't assert close() directly (sqlite doesn't expose it), but coverage
    will mark the 'else: con.close()' lines as executed.
    """
    s = Store(str((tmp_path / "db.sqlite").resolve()))
    # Do a harmless statement to ensure there's a transaction to commit.
    with s._conn_ctx() as con:  # noqa: SLF001 (white-box test)
        con.execute("SELECT 1")


def test_conn_ctx_body_exception_triggers_rollback_and_close(monkeypatch, tmp_path):
    """
    Raise from inside the with-block:
      - hit outer 'except' path (lines ~96-98): rollback(); close(); re-raise
    """
    s = Store(str((tmp_path / "db3.sqlite").resolve()))

    class FakeConn:
        def __init__(self):
            self.row_factory = None
            self.did_commit = False
            self.did_rollback = False
            self.did_close = False

        def execute(self, *a, **k):
            return None

        def commit(self):
            self.did_commit = True  # should NOT be reached in this path

        def rollback(self):
            self.did_rollback = True

        def close(self):
            self.did_close = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            # mimic sqlite behavior: rollback on exception, else commit
            if exc_type is not None:
                self.rollback()
            else:
                try:
                    self.commit()
                except Exception:
                    pass
            self.close()
            # return False to propagate exceptions (like sqlite3.Connection)
            return False

    fake = FakeConn()
    monkeypatch.setattr("corpusloom.store.sqlite3.connect", lambda path: fake)

    with pytest.raises(RuntimeError, match="boom"):
        with s._conn() as _:
            raise RuntimeError("boom")

    assert fake.did_rollback is True
    assert fake.did_close is True
    assert fake.did_commit is False


def test_update_chunk_vector_roundtrip(tmp_path):
    db = tmp_path / "db.sqlite"
    s = Store(str(db))

    # Create a doc + one chunk with no vector yet
    doc_id = s.upsert_document("src.txt", meta={"v": 1})
    chunk_id = s.insert_chunk(doc_id, idx=0, text="hello world", vector=None, meta=None)

    # Before: vector_json should be NULL
    rows = list(s.iter_chunks())
    pre = next(r for r in rows if r["id"] == chunk_id)
    assert pre["vector_json"] is None

    # Update vector
    vec = [0.1, 0.2, 0.3]
    s.update_chunk_vector(chunk_id, vec)

    # After: vector_json should be the JSON-encoded list
    rows2 = list(s.iter_chunks())
    post = next(r for r in rows2 if r["id"] == chunk_id)
    assert json.loads(post["vector_json"]) == vec
