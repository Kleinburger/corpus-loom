import os, json, tempfile, sqlite3, time
from pathlib import Path
import pytest

from corpusloom import OllamaClient

# ---- Helpers to stub network ----
def stub_post(self, path, payload):
    # Embeddings
    if path.endswith("/api/embeddings"):
        txt = payload.get("prompt","")
        # deterministic tiny vector from length + bytes
        s = float(len(txt) % 10)
        return {"embedding": [s, s+1, s+2, s+3]}
    # Generate
    if path.endswith("/api/generate"):
        return {"model": self.model, "response": "ok", "eval_count": 10, "eval_duration": 1, "context": [1,2,3]}
    # Chat (non-stream)
    if path.endswith("/api/chat"):
        # echo last user content as reply
        msgs = payload.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        return {"model": self.model, "message": {"content": last.upper()}, "eval_count": 5, "eval_duration": 1}
    raise AssertionError("unexpected path " + path)

def stub_post_stream(self, path, payload):
    # Stream two chunks and a done line
    if path.endswith("/api/generate"):
        yield {"response": "A"}
        yield {"response": "B"}
        yield {"done": True, "model": self.model, "eval_count": 2, "eval_duration": 1, "context": [9,9]}
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
    v3 = client.embed_texts(["hello"], cache=False) # bypass cache
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
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    n = con.execute("SELECT count(*) FROM documents").fetchone()[0]
    con.close()
    return int(n)

def test_add_text_incremental(client, tmp_path):
    db = str(tmp_path / "db.sqlite")
    client.store.db_path = db  # ensure fresh DB
    text1 = "Para A\\n\\nPara B"
    d1, chunks1 = client.add_text(text1, source="inline1")
    # reuse doc_id, only one new chunk should be added if one paragraph changes
    text2 = "Para A\\n\\nPara C"   # share first chunk hash, second differs
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
    p = tmp_path / "c.md"; p.write_text("Alpha\\n\\nBeta\\n\\nGamma", encoding="utf-8")
    client.add_files([str(p)], strategy="replace")
    ctx = client.build_context("Alpha", top_k=2)

    # Expect one stitched blocks
    assert ctx.count("[CTX 1") == 1

def test_context_building_uses_topk(client, tmp_path):
    # Ingest TWO docs so build_context (doc-level stitching) emits two CTX blocks
    p1 = tmp_path / "c1.md"; p1.write_text("Alpha", encoding="utf-8")
    p2 = tmp_path / "c2.md"; p2.write_text("Omega", encoding="utf-8")  # same length as "Alpha" → equal similarity with the test’s length-based embedding stub
    client.add_files([str(p1), str(p2)], strategy="replace")
    ctx = client.build_context("Alpha", top_k=2)

    #Expect 2 stitched blocks
    assert ctx.count("[CTX 1") == 1
    assert ctx.count("[CTX 2") == 1
