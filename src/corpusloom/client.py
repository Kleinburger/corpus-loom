from __future__ import annotations

import glob as _glob
import hashlib
import json
import os
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional

import requests

from .chunking import Chunker
from .json_mode import JsonMode
from .models import ChatResult, GenerateResult, Message
from .retrieval import Retriever
from .store import Store
from .utils import RateLimiter, hash_key, now_ms


class OllamaClient:
    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        host: str = "http://localhost:11434",
        cache_db_path: str = "./.ollama_client/cache.sqlite",
        default_options: Optional[Dict[str, Any]] = None,
        calls_per_minute: Optional[int] = None,
        keep_alive: Optional[str] = None,
        request_timeout: int = 300,
        chunk_max_tokens: int = 800,
        chunk_overlap_tokens: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self.default_options = default_options or {"temperature": 0.2}
        self.request_timeout = request_timeout

        self.store = Store(cache_db_path)
        self.rate = RateLimiter(calls_per_minute or 0)
        self.chunker = Chunker(max_tokens=chunk_max_tokens, overlap_tokens=chunk_overlap_tokens)
        self.retriever = Retriever(self.store)
        self.json_mode = JsonMode(
            self._post,
            self.store.get_history,
            self.store.append_message,
            self.store.get_conversation_system,
            self.model,
            self.keep_alive,
            self.default_options,
        )

    def register_template(self, name: str, template: str) -> None:
        self.store.upsert_template(name, template)

    def list_templates(self) -> Dict[str, str]:
        return self.store.list_templates()

    def render_template(self, name: str, **kwargs) -> str:
        tmpl = self.store.get_template(name)
        if tmpl is None:
            raise KeyError(f"Template '{name}' not found")
        try:
            return tmpl.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: '{e.args[0]}'") from e

    def new_conversation(self, name: Optional[str] = None, system: Optional[str] = None) -> str:
        return self.store.new_conversation(name, system)

    def history(self, convo_id: str) -> List[Message]:
        return self.store.get_history(convo_id)

    def embed_texts(
        self,
        texts: Iterable[str],
        embed_model: str = "nomic-embed-text",
        cache: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        texts_list = list(texts)
        total = len(texts_list)
        for i, t in enumerate(texts_list, start=1):
            key = hash_key(embed_model, t)
            if cache:
                cached = self.store.get_embedding(key)
                if cached is not None:
                    vectors.append(cached)
                    if on_progress:
                        on_progress(i, total)
                    continue
            payload = {"model": embed_model, "prompt": t}
            res = self._post("/api/embeddings", payload)
            vec = res.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError(f"Bad embedding response for index {i}: {res}")
            if cache:
                self.store.put_embedding(
                    key, embed_model, hashlib.sha256(t.encode("utf-8")).hexdigest(), vec
                )
            vectors.append(vec)
            if on_progress:
                on_progress(i, total)
        return vectors

    def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> GenerateResult | Generator[str, None, GenerateResult]:
        opts = {**self.default_options, **(options or {})}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": opts,
            "keep_alive": self.keep_alive,
            "stream": bool(stream),
        }
        t0 = now_ms()
        if not stream:
            res = self._post("/api/generate", payload)
            t1 = now_ms()
            return GenerateResult(
                model=res.get("model", self.model),
                prompt=prompt,
                response_text=res.get("response", ""),
                total_duration_ms=(t1 - t0),
                eval_count=res.get("eval_count"),
                eval_duration_ms=res.get("eval_duration"),
                context=res.get("context"),
                raw=res,
            )

        def _gen() -> Generator[str, None, GenerateResult]:
            parts: List[str] = []
            final_raw: Dict[str, Any] = {}
            for chunk in self._post_stream("/api/generate", payload):
                if "response" in chunk:
                    tok = chunk["response"]
                    parts.append(tok)
                    if on_token:
                        on_token(tok)
                    yield tok
                if chunk.get("done"):
                    final_raw = chunk
                    break
            t1 = now_ms()
            return GenerateResult(
                model=final_raw.get("model", self.model),
                prompt=prompt,
                response_text="".join(parts),
                total_duration_ms=(t1 - t0),
                eval_count=final_raw.get("eval_count"),
                eval_duration_ms=final_raw.get("eval_duration"),
                context=final_raw.get("context"),
                raw=final_raw or None,
            )

        return _gen()

    def chat(
        self,
        convo_id: str,
        user_message: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> ChatResult | Generator[str, None, ChatResult]:
        self.store.append_message(convo_id, "user", user_message, {"kind": "chat_input"})
        system = self.store.get_conversation_system(convo_id)
        history = self.store.get_history(convo_id)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in history:
            msgs.append({"role": m.role, "content": m.content})
        opts = {**self.default_options, **(options or {})}
        payload = {
            "model": self.model,
            "messages": msgs,
            "options": opts,
            "keep_alive": self.keep_alive,
            "stream": bool(stream),
        }
        t0 = now_ms()
        if not stream:
            res = self._post("/api/chat", payload)
            t1 = now_ms()
            msg = res.get("message", {})
            assistant_text = msg.get("content", "")
            self.store.append_message(
                convo_id, "assistant", assistant_text, {"kind": "chat_output"}
            )
            full_history = self.store.get_history(convo_id)
            return ChatResult(
                model=res.get("model", self.model),
                messages=full_history,
                reply=Message("assistant", assistant_text),
                total_duration_ms=(t1 - t0),
                eval_count=res.get("eval_count"),
                eval_duration_ms=res.get("eval_duration"),
                raw=res,
            )

        def _gen() -> Generator[str, None, ChatResult]:
            parts: List[str] = []
            final_raw: Dict[str, Any] = {}
            for chunk in self._post_stream("/api/chat", payload):
                delta = chunk.get("message", {}).get("content")
                if delta:
                    parts.append(delta)
                    if on_token:
                        on_token(delta)
                    yield delta
                if chunk.get("done"):
                    final_raw = chunk
                    break
            t1 = now_ms()
            assistant_text = "".join(parts)
            self.store.append_message(
                convo_id, "assistant", assistant_text, {"kind": "chat_output"}
            )
            full_history = self.store.get_history(convo_id)
            return ChatResult(
                model=final_raw.get("model", self.model),
                messages=full_history,
                reply=Message("assistant", assistant_text),
                total_duration_ms=(t1 - t0),
                eval_count=final_raw.get("eval_count"),
                eval_duration_ms=final_raw.get("eval_duration"),
                raw=final_raw or None,
            )

        return _gen()

    def generate_json(
        self,
        prompt: str,
        schema,
        options: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ):
        return self.json_mode.generate_json(prompt, schema, options, retries)

    def chat_json(
        self,
        convo_id: str,
        user_message: str,
        schema,
        options: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ):
        return self.json_mode.chat_json(convo_id, user_message, schema, options, retries)

    def add_text(
        self,
        text: str,
        source: str = "inline",
        *,
        embed_model: str = "nomic-embed-text",
        cache_embeddings: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        reuse_incremental: bool = False,
    ) -> tuple[str, list[str]]:
        if doc_id is None:
            doc_id = self.store.upsert_document(source=source, meta=metadata)
        else:
            if metadata is not None:
                self.store.update_document_meta(doc_id, metadata)

        chunks = self.chunker.chunk_text(text)
        chunk_ids: List[str] = []

        if reuse_incremental:
            existing = self.store.get_chunk_hash_map(doc_id)
            to_embed: List[str] = []
            for i, t in enumerate(chunks):
                ch = hashlib.sha256(t.encode("utf-8")).hexdigest()
                if ch in existing:
                    continue
                to_embed.append(t)
            vectors_new = self.embed_texts(
                to_embed, embed_model=embed_model, cache=cache_embeddings
            )
            it = iter(vectors_new)
            for i, t in enumerate(chunks):
                ch = hashlib.sha256(t.encode("utf-8")).hexdigest()
                if ch in existing:
                    continue
                v = next(it)
                cid = self.store.insert_chunk(
                    doc_id, i, t, v, meta={"source": source, "chunk_hash": ch}
                )
                chunk_ids.append(cid)
        else:
            vectors = self.embed_texts(chunks, embed_model=embed_model, cache=cache_embeddings)
            for i, (t, v) in enumerate(zip(chunks, vectors)):
                ch = hashlib.sha256(t.encode("utf-8")).hexdigest()
                cid = self.store.insert_chunk(
                    doc_id, i, t, v, meta={"source": source, "chunk_hash": ch}
                )
                chunk_ids.append(cid)

        return doc_id, chunk_ids

    def add_files(
        self,
        paths_or_globs: Iterable[str],
        *,
        encoding: str = "utf-8",
        embed_model: str = "nomic-embed-text",
        cache_embeddings: bool = True,
        per_file_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
        strategy: str = "auto",
    ) -> List[tuple[str, list[str]]]:
        results: List[tuple[str, list[str]]] = []
        exts_ok = {
            ".txt",
            ".md",
            ".rst",
            ".py",
            ".cpp",
            ".hpp",
            ".c",
            ".h",
            ".json",
            ".yaml",
            ".yml",
            ".ini",
        }
        files: List[str] = []
        for p in paths_or_globs:
            files.extend(_glob.glob(p))

        for path in sorted(set(files)):
            _, ext = os.path.splitext(path)
            if ext.lower() not in exts_ok:
                continue

            try:
                file_hash = self._file_hash(path)
            except Exception:
                file_hash = None

            existing = self.store.get_latest_document_by_source(path)
            reuse_doc_id = None
            if existing:
                doc_id_existing, meta_existing = existing
                if strategy == "skip":
                    # Skip entirely if a document for this source already exists (even if changed)
                    continue
                if (
                    file_hash
                    and (strategy in ("auto", "skip"))
                    and meta_existing.get("content_hash") == file_hash
                ):
                    if strategy in ("auto", "skip"):
                        continue
                if strategy in ("auto", "replace"):
                    reuse_doc_id = doc_id_existing
                    self.store.delete_chunks_for_doc(reuse_doc_id)

            try:
                with open(path, "r", encoding=encoding, errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue

            meta = per_file_metadata(path) if per_file_metadata else {"path": path}
            if file_hash:
                meta = {
                    **meta,
                    "content_hash": file_hash,
                    "mtime": os.path.getmtime(path),
                    "size": os.path.getsize(path),
                }

            if reuse_doc_id:
                doc_id, chunk_ids = self.add_text(
                    text,
                    source=path,
                    embed_model=embed_model,
                    cache_embeddings=cache_embeddings,
                    metadata=meta,
                    doc_id=reuse_doc_id,
                    reuse_incremental=False,
                )
            else:
                doc_id, chunk_ids = self.add_text(
                    text,
                    source=path,
                    embed_model=embed_model,
                    cache_embeddings=cache_embeddings,
                    metadata=meta,
                )

            results.append((doc_id, chunk_ids))

        return results

    def search_similar(self, query: str, *, embed_model: str = "nomic-embed-text", top_k: int = 5):
        qv = self.embed_texts([query], embed_model=embed_model, cache=True)[0]
        ranked = self.retriever.rank_chunks(qv)
        return [r for _, r in ranked[:top_k]]

    def build_context(
        self, query: str, *, top_k: int = 5, embed_model: str = "nomic-embed-text"
    ) -> str:
        hits = self.search_similar(query, embed_model=embed_model, top_k=top_k)
        ctx_parts = []
        for i, h in enumerate(hits, 1):
            src = h["meta"].get("path") or h["meta"].get("source") or h["doc_id"]
            header = f"[CTX {i} | score={h['score']:.3f} | src={src}]"
            ctx_parts.append(header + "\n" + h["text"])
        return "\n\n".join(ctx_parts)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.rate.throttle()
        url = f"{self.host}{path}"
        r = requests.post(url, json=payload, timeout=self.request_timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama returned {r.status_code}: {r.text[:500]}")
        return r.json()

    def _post_stream(self, path: str, payload: Dict[str, Any]):
        self.rate.throttle()
        url = f"{self.host}{path}"
        with requests.post(url, json=payload, stream=True, timeout=self.request_timeout) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama returned {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
