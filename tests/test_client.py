import builtins
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from corpusloom import OllamaClient


# ---- Helpers to stub network ----
def stub_post(self, path, payload):
    # Embeddings
    if path.endswith("/api/embeddings"):
        txt = payload.get("prompt", "")
        # deterministic tiny vector from length + bytes
        s = float(len(txt) % 10)
        return {"embedding": [s, s + 1, s + 2, s + 3]}
    # Generate
    if path.endswith("/api/generate"):
        return {
            "model": self.model,
            "response": "ok",
            "eval_count": 10,
            "eval_duration": 1,
            "context": [1, 2, 3],
        }
    # Chat (non-stream)
    if path.endswith("/api/chat"):
        # echo last user content as reply
        msgs = payload.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        return {
            "model": self.model,
            "message": {"content": last.upper()},
            "eval_count": 5,
            "eval_duration": 1,
        }
    raise AssertionError("unexpected path " + path)

def stub_post_stream(self, path, payload):
    # Stream two chunks and a done line
    if path.endswith("/api/generate"):
        yield {"response": "A"}
        yield {"response": "B"}
        yield {
            "done": True,
            "model": self.model,
            "eval_count": 2,
            "eval_duration": 1,
            "context": [9, 9],
        }
        return
    if path.endswith("/api/chat"):
        yield {"message": {"content": "A"}}
        yield {"message": {"content": "B"}}
        yield {"done": True, "model": self.model, "eval_count": 2, "eval_duration": 1}
        return
    raise AssertionError("unexpected path " + path)

@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "db.sqlite"
    c = OllamaClient(cache_db_path=str(db))
    monkeypatch.setattr(OllamaClient, "_post", stub_post, raising=False)
    monkeypatch.setattr(OllamaClient, "_post_stream", stub_post_stream, raising=False)
    return c

def test_embed_cache_and_fallback(client):
    v1 = client.embed_texts(["hello"], cache=True)
    v2 = client.embed_texts(["hello"], cache=True)  # cache hit
    assert v1 == v2
    v3 = client.embed_texts(["hello"], cache=False)  # bypass cache
    assert v3 != []

def test_generate_stream_and_nonstream(client, capsys):
    # non-stream path
    res = client.generate("x", stream=False)
    assert res.response_text == "ok"
    # stream path
    gen = client.generate("x", stream=True)
    tokens = "".join(list(gen))
    assert tokens == "AB"

def test_chat_stream_and_nonstream(client):
    cid = client.new_conversation(system="SYS")
    # non-stream
    r1 = client.chat(cid, "hello")
    assert r1.reply.content == "HELLO"
    # stream
    gen = client.chat(cid, "hello again", stream=True)
    out = "".join(list(gen))
    assert out == "AB"

def write(tmpdir: Path, name: str, content: str) -> Path:
    p = tmpdir / name
    p.write_text(content, encoding="utf-8")
    return p

def count_docs(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    n = con.execute("SELECT count(*) FROM documents").fetchone()[0]
    con.close()
    return int(n)

def test_add_text_incremental(client, tmp_path):
    db = str(tmp_path / "db.sqlite")
    client.store.db_path = db  # ensure fresh DB
    text1 = "Para A\\n\\nPara B"
    d1, chunks1 = client.add_text(text1, source="inline1")
    # reuse doc_id, only one new chunk should be added if one paragraph changes
    text2 = "Para A\\n\\nPara C"  # share first chunk hash, second differs
    d2, chunks2 = client.add_text(text2, source="inline1", doc_id=d1, reuse_incremental=True)
    assert d1 == d2
    # Should have inserted only one additional chunk
    assert len(chunks2) == 1

def test_add_files_strategies_auto_replace_skip(client, tmp_path):
    # create two versions of a file
    f1 = write(tmp_path, "a.md", "# A\\n")
    f2 = write(tmp_path, "b.txt", "B1")
    # ingest initial
    out1 = client.add_files([str(f1), str(f2)], strategy="auto")
    assert len(out1) == 2
    db = client.store.db_path
    assert count_docs(db) == 2

    # unchanged file with auto -> skip
    out2 = client.add_files([str(f1)], strategy="auto")
    assert out2 == []

    # modify file and auto -> replace chunks within same doc id (doc count unchanged)
    f1.write_text("# A changed\\n", encoding="utf-8")
    out3 = client.add_files([str(f1)], strategy="auto")
    assert len(out3) == 1
    assert count_docs(db) == 2  # still 2 docs, replaced not added

    # skip strategy -> if doc exists, skip even if changed
    f2.write_text("B2", encoding="utf-8")
    out4 = client.add_files([str(f2)], strategy="skip")
    assert out4 == []  # skipped
    assert count_docs(db) == 2

def test_context_building_uses_topk(client, tmp_path):
    # ingest content to build context
    p = tmp_path / "c.md"
    p.write_text("Alpha\\n\\nBeta\\n\\nGamma", encoding="utf-8")
    client.add_files([str(p)], strategy="replace")
    ctx = client.build_context("Alpha", top_k=2)

    # Expect one stitched blocks
    assert ctx.count("[CTX 1") == 1

def test_context_building_uses_topk(client, tmp_path):
    # Ingest TWO docs so build_context (doc-level stitching) emits two CTX blocks
    p1 = tmp_path / "c1.md"
    p1.write_text("Alpha", encoding="utf-8")
    p2 = tmp_path / "c2.md"
    p2.write_text("Omega", encoding="utf-8")  # same length as "Alpha" → equal similarity with the test’s length-based embedding stub
    client.add_files([str(p1), str(p2)], strategy="replace")
    ctx = client.build_context("Alpha", top_k=2)

    # Expect 2 stitched blocks
    assert ctx.count("[CTX 1") == 1
    assert ctx.count("[CTX 2") == 1

class FakeResp:
    """Context-manager response for _post_stream tests."""
    def __init__(self, status_code=200, lines=None, text="ERR", json_obj=None):
        self.status_code = status_code
        self._lines = lines or []
        self.text = text
        self._json_obj = json_obj

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self, decode_unicode=True):
        for ln in self._lines:
            yield ln

    def json(self):
        if self._json_obj is not None:
            return self._json_obj
        return {}

def test_render_template_missing_and_missing_var(tmp_path):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    # Missing template -> KeyError
    with pytest.raises(KeyError):
        c.render_template("nope")

    # Missing variable -> ValueError
    c.register_template("greet", "Hello {name}")
    with pytest.raises(ValueError):
        c.render_template("greet")  # no 'name' provided

def test_embed_texts_progress_cache_and_bad_response(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    calls = []
    def on_prog(i, total):
        calls.append((i, total))

    # 1) Good embedding (populates cache)
    def ok_post(self, path, payload):
        return {"embedding": [0.0, 1.0, 2.0]}
    monkeypatch.setattr(OllamaClient, "_post", ok_post, raising=False)
    v1 = c.embed_texts(["hello"], cache=True, on_progress=on_prog)
    assert v1 and calls == [(1, 1)]  # progress reported

    # 2) Cache hit still reports progress
    calls.clear()
    v2 = c.embed_texts(["hello"], cache=True, on_progress=on_prog)
    assert v2 and calls == [(1, 1)]

    # 3) Bad response type -> RuntimeError
    def bad_post(self, path, payload):
        return {"embedding": "not-a-list"}
    monkeypatch.setattr(OllamaClient, "_post", bad_post, raising=False)
    with pytest.raises(RuntimeError):
        c.embed_texts(["new"], cache=False, on_progress=on_prog)

def test_generate_stream_on_token(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    def stub_stream(self, path, payload):
        yield {"response": "X"}
        yield {"response": "Y"}
        yield {"done": True, "model": self.model, "eval_count": 1, "eval_duration": 1, "context": []}
    monkeypatch.setattr(OllamaClient, "_post_stream", stub_stream, raising=False)

    seen = []
    gen = c.generate("p", stream=True, on_token=lambda t: seen.append(t))
    text = "".join(gen)
    assert text == "XY"
    assert seen == ["X", "Y"]

def test_chat_stream_on_token(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))
    cid = c.new_conversation(system="SYS")

    def stub_stream(self, path, payload):
        yield {"message": {"content": "A"}}
        yield {"message": {"content": "B"}}
        yield {"done": True, "model": self.model, "eval_count": 2, "eval_duration": 1}
    monkeypatch.setattr(OllamaClient, "_post_stream", stub_stream, raising=False)

    seen = []
    gen = c.chat(cid, "hi", stream=True, on_token=lambda t: seen.append(t))
    out = "".join(gen)
    assert out == "AB"
    assert seen == ["A", "B"]

def test_json_mode_delegation_generate(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))
    class JM:
        def generate_json(self, prompt, schema, options, retries):
            return {"ok": True, "prompt": prompt, "r": retries}
    c.json_mode = JM()
    out = c.generate_json("P", schema=dict, options={"x": 1}, retries=3)
    assert out["ok"] and out["prompt"] == "P" and out["r"] == 3


def test_json_mode_delegation_chat(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))
    class JM:
        def chat_json(self, cid, msg, schema, options, retries):
            return {"cid": cid, "msg": msg, "r": retries}
    c.json_mode = JM()
    out = c.chat_json("CID", "hello", dict, options=None, retries=1)
    assert out == {"cid": "CID", "msg": "hello", "r": 1}

def test_add_text_updates_metadata_when_doc_id(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    # Make embeddings deterministic
    def emb_post(self, path, payload):
        return {"embedding": [1.0, 2.0, 3.0]}
    monkeypatch.setattr(OllamaClient, "_post", emb_post, raising=False)

    # Create a doc_id first
    did = c.store.upsert_document(source="src1", meta={"a": 1})

    called = {}
    def fake_update(doc_id, meta):
        called["doc_id"] = doc_id
        called["meta"] = meta
    monkeypatch.setattr(c.store, "update_document_meta", fake_update, raising=True)

    c.add_text("Alpha\n\nBeta", source="src1", doc_id=did, metadata={"foo": "bar"})
    assert called["doc_id"] == did
    assert called["meta"]["foo"] == "bar"

def test_add_files_nonmatching_ext_and_open_failure(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    # 1) Non-matching extension -> ignored
    junk = tmp_path / "x.bin"
    junk.write_text("zzz", encoding="utf-8")
    out = c.add_files([str(junk)])
    assert out == []

    # 2) open() failure -> continue (skips file)
    bad = tmp_path / "bad.txt"
    bad.write_text("won't open", encoding="utf-8")

    orig_open = builtins.open
    def failing_open(path, *a, **kw):
        if str(path) == str(bad):
            raise OSError("nope")
        return orig_open(path, *a, **kw)

    # Stub embeddings to avoid network if anything gets through
    def emb_post(self, path, payload):
        return {"embedding": [0.1, 0.2]}
    monkeypatch.setattr(OllamaClient, "_post", emb_post, raising=False)

    monkeypatch.setattr(builtins, "open", failing_open)
    try:
        out2 = c.add_files([str(bad)])
        assert out2 == []
    finally:
        monkeypatch.setattr(builtins, "open", orig_open)

def test_add_files_meta_callback_and_filehash_exception(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    # Make sure embeddings are stubbed
    def emb_post(self, path, payload):
        return {"embedding": [0.0, 0.0]}
    monkeypatch.setattr(OllamaClient, "_post", emb_post, raising=False)

    p = tmp_path / "ok.md"
    p.write_text("Alpha\n\nBeta", encoding="utf-8")

    # Force _file_hash to raise -> branch sets file_hash=None
    def boom(*a, **k):
        raise RuntimeError("hash fail")
    monkeypatch.setattr(OllamaClient, "_file_hash", boom, raising=True)

    def pmeta(path):
        return {"custom": "yes", "path": "WRONG"}  # ensure we can see replacement
    out = c.add_files([str(p)], per_file_metadata=pmeta, strategy="auto")
    assert len(out) == 1

    # Verify metadata stored respects callback + no content_hash when hash failed
    did, _ = out[0]
    latest = c.store.get_latest_document_by_source(str(p))
    assert latest is not None
    doc_id, meta = latest
    assert doc_id == did
    assert meta.get("custom") == "yes"
    # content_hash omitted when hashing failed
    assert "content_hash" not in meta

def test_add_files_replace_hits_reuse_doc_id(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    def emb_post(self, path, payload):
        return {"embedding": [0.1, 0.2]}
    monkeypatch.setattr(OllamaClient, "_post", emb_post, raising=False)

    f = tmp_path / "a.md"
    f.write_text("First", encoding="utf-8")
    out1 = c.add_files([str(f)], strategy="auto")
    assert len(out1) == 1
    # Now replace explicitly -> ensures the 'replace' branch runs, reusing doc_id
    f.write_text("Changed", encoding="utf-8")
    out2 = c.add_files([str(f)], strategy="replace")
    assert len(out2) == 1

def test__post_non_200_raises(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    class R:
        status_code = 500
        text = "bad"
        def json(self): return {}
    def fake_post(url, json=None, timeout=None, stream=False):
        return R()

    import corpusloom.client as client_mod
    monkeypatch.setattr(client_mod.requests, "post", fake_post, raising=True)

    with pytest.raises(RuntimeError):
        c._post("/api/generate", {"x": 1})

def test__post_stream_error_and_line_parsing(tmp_path, monkeypatch):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))

    # 1) Non-200 in stream -> raise
    def post_err(url, json=None, stream=False, timeout=None):
        return FakeResp(status_code=500, text="oops")
    import corpusloom.client as client_mod
    monkeypatch.setattr(client_mod.requests, "post", post_err, raising=True)
    with pytest.raises(RuntimeError):
        list(c._post_stream("/api/generate", {"x": 1}))

    # 2) 200 but with empty and invalid JSON lines - they should be skipped;
    #    valid ones should be yielded.
    lines = [
        "",                # skipped
        "not json",        # skipped
        json.dumps({"a": 1}),
        "   ",             # skipped
        json.dumps({"b": 2}),
    ]
    def post_ok(url, json=None, stream=False, timeout=None):
        return FakeResp(status_code=200, lines=lines)
    monkeypatch.setattr(client_mod.requests, "post", post_ok, raising=True)
    got = list(c._post_stream("/api/generate", {"x": 2}))
    assert got == [{"a": 1}, {"b": 2}]

def test__file_hash_reads_and_hashes(tmp_path):
    c = OllamaClient(cache_db_path=str(tmp_path / "db.sqlite"))
    p = tmp_path / "big.txt"
    # write > 1 chunk’s worth of data? Not needed; just ensure deterministic hash.
    p.write_bytes(b"hello world")
    h1 = c._file_hash(str(p))
    # compute independently
    h2 = hashlib.sha256(b"hello world").hexdigest()
    assert h1 == h2
