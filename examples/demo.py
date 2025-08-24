"""
corpus-loom demo: end-to-end usage in one file.

What this script shows:
  1) Construct an OllamaClient with defaults
  2) Ingest a small markdown file (chunk + embed + store)
  3) Search & build a stitched context
  4) One-shot generation (non-stream + stream)
  5) Simple chat with a conversation
  6) Templates: register, list, render
  7) JSON mode generation with Pydantic schema (fallback if pydantic missing)

Requires: an Ollama server at http://localhost:11434 and a model (default gpt-oss:20b).
You can override host/model/db via environment variables:
  OCP_HOST, OCP_MODEL, OCP_DB, OCP_KEEP_ALIVE, OCP_CPM
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from corpusloom import OllamaClient
from corpusloom.utils import extract_json_str

HOST = os.getenv("OCP_HOST", "http://localhost:11434")
MODEL = os.getenv("OCP_MODEL", "gpt-oss:20b")
DB    = os.getenv("OCP_DB", "./.ollama_client/cache.sqlite")
KEEP  = os.getenv("OCP_KEEP_ALIVE", "10m")
CPM   = int(os.getenv("OCP_CPM", "0"))

def make_client() -> OllamaClient:
    print(f"[i] Connecting to {HOST} model={MODEL} db={DB}")
    return OllamaClient(
        model=MODEL,
        host=HOST,
        cache_db_path=DB,
        default_options={},    # you can pre-seed any model options here
        keep_alive=KEEP,
        calls_per_minute=CPM,  # simple client-side rate limiter; 0=off
    )

def demo_ingest(client: OllamaClient) -> str:
    """
    Create a tiny markdown file and ingest it. Returns the file path.
    """
    fd, path = tempfile.mkstemp(suffix=".md", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Alpha\n\nBeta\n\nGamma\n\n```py\nprint('hello')\n```")
    print(f"[i] Ingesting: {path}")
    results = client.add_files([path], encoding="utf-8", embed_model="nomic-embed-text", strategy="replace")
    # results ~ [("doc_id", ["chunk_id", ...])]
    print("[ok] Ingested:", json.dumps({"ingested": [{"doc_id": d, "chunks": len(ch)} for d, ch in results]}, indent=2))
    return path

def demo_search_and_context(client: OllamaClient) -> None:
    print("\n[i] Searching for: 'Alpha'")
    hits = client.search_similar("Alpha", embed_model="nomic-embed-text", top_k=3)
    print(json.dumps(hits, indent=2, ensure_ascii=False))

    print("\n[i] Building stitched context (top_k=2) for: 'Alpha'")
    ctx = client.build_context("Alpha", top_k=2, embed_model="nomic-embed-text")
    print("[context]\n" + ctx)

def demo_generate(client: OllamaClient) -> None:
    print("\n[i] One-shot, non-stream:")
    res = client.generate("Write a single short sentence about alpacas.")
    print(res.response_text)

    print("\n[i] One-shot, stream:")
    for tok in client.generate("Stream three short words, space-separated.", stream=True):
        print(tok, end="", flush=True)
    print()

def demo_chat(client: OllamaClient) -> None:
    print("\n[i] Chat demo:")
    convo_id = client.new_conversation(name="Demo chat", system="Be concise.")
    print("  convo_id:", convo_id)

    r1 = client.chat(convo_id, "Hello! In one sentence, explain chunking vs tokenization.")
    print("assistant:", r1.reply.content)

    print("[i] Send a streaming message:")
    for tok in client.chat(convo_id, "Stream a haiku about llamas.", stream=True):
        print(tok, end="", flush=True)
    print()

    print("[i] Conversation history:")
    for m in client.history(convo_id):
        print(f"  {m.role}: {m.content}")

def demo_templates(client: OllamaClient) -> None:
    print("\n[i] Templates demo:")
    client.register_template("hello", "Hello {name2}, welcome to {place}!")
    print("  Registered 'hello'")
    tmpls = client.list_templates()
    print("  Templates:", json.dumps(tmpls, indent=2, ensure_ascii=False))
    rendered = client.render_template("hello", name2="Ethan", place="CorpusLoom")
    print("  Rendered:", rendered)

def demo_json_mode(client: OllamaClient) -> None:
    print("\n[i] JSON-mode demo:")
    # Try with Pydantic if available; otherwise, demonstrate a light fallback and skip validation.
    try:
        from pydantic import BaseModel  # type: ignore

        class Animal(BaseModel):
            name2: str
            kind: str

        obj = client.generate_json(
            prompt="Return an object like {name2:'Llama', kind:'Camelid'}",
            schema=Animal,
        )
        if hasattr(obj, "model_dump"):
            print("  Pydantic object:", obj.model_dump())
        else:
            print("  JSON object:", obj)
    except Exception as e:
        # If pydantic isn’t present, the JsonMode will raise a RuntimeError in _ensure_pydantic.
        print("  [skip] JSON-mode requires pydantic. Error:", e)

    # Alternate “strict JSON” without schema – useful if you just want JSON text:
    print("\n[i] Strict JSON via chat (no schema):")
    strict = (
        "You are a strict JSON generator. Return ONLY valid JSON. "
        "Do not include prose or code fences."
    )
    cid = client.new_conversation(system=strict)
    r = client.chat(cid, "Return an object: {city: 'Quito', country: 'Ecuador'}")
    jtxt = extract_json_str(r.reply.content) or r.reply.content.strip()
    print("  JSON text:", jtxt)
    try:
        print("  Parsed:", json.loads(jtxt))
    except Exception:
        print("  (Could not parse; model may have returned non-JSON.)")

def main() -> int:
    client = make_client()

    try:
        demo_ingest(client)
        demo_search_and_context(client)
        demo_generate(client)
        demo_chat(client)
        demo_templates(client)
        demo_json_mode(client)
    except Exception as e:
        # This is just a demo; if the Ollama server or model isn’t available,
        # we’ll fail gracefully with a single message.
        print("[error]", type(e).__name__, e)
        return 2

    print("\n[done]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())