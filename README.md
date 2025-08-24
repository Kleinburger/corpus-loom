# CorpusLoom

**Python SDK + CLI for local LLM workflows on Ollama.**  
Ingest and embed documents into a corpus, retrieve top-K context (RAG), template prompts, persist chat history, and enforce **strict JSON/typed outputs** — all locally, privacy-first. The CLI is `cloom`.

---

## Table of contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Core concepts](#core-concepts)
4. [Modes](#modes)
   - [Generation](#generation)
   - [Chat](#chat)
   - [JSON mode (schema-enforced)](#json-mode-schemaenforced)
   - [Ingest & Retrieval (RAG)](#ingest--retrieval-rag)
5. [Re-ingest strategies & incremental updates](#re-ingest-strategies--incremental-updates)
6. [Model options (behavior tuning)](#model-options-behavior-tuning)
7. [CLI usage (`cloom`)](#cli-usage-cloom)
8. [Recipes](#recipes)
9. [Templating](#templating)
10. [Streaming](#streaming)
11. [Performance & scaling tips](#performance--scaling-tips)
12. [Tests, coverage, and CI](#tests-coverage-and-ci)
13. [Troubleshooting](#troubleshooting)
14. [FAQ](#faq)

---

## Install

Requirements:
- Python 3.9+
- [Ollama](https://ollama.com) running locally (default: `http://localhost:11434`)
- The model(s) you want pulled, e.g.:
  ```bash
  ollama pull gpt-oss:20b
  ollama pull nomic-embed-text
  ```

Install the package (and optional extras):

```bash
pip install -e .            # local dev install (installs the 'cloom' CLI)
pip install -e ".[json]"    # add Pydantic for JSON-mode validation (recommended)
pip install -e ".[dev]"     # dev tools: pytest, ruff, mypy, coverage
```

---

## Quick start

**Python**

```python
from typing import List
from pydantic import BaseModel, Field
from corpusloom import OllamaClient

# Initialize the client
client = OllamaClient(
    model="gpt-oss:20b",
    host="http://localhost:11434",
    keep_alive="10m",                 # keep model warm
    default_options={"temperature": 0.2, "num_ctx": 16384},  # generation defaults
)

# 1) Ingest documents into your corpus
client.add_files(["docs/**/*.md", "specs/*.txt"])  # chunk + embed + store

# 2) Build a context block via similarity search (RAG)
ctx = client.build_context("overshoot < 5% attitude controller", top_k=5)

# 3) Generate using that context
res = client.generate(f"Using this context, outline tests:\n{ctx}")
print(res.response_text)

# 4) Enforce structured JSON with Pydantic
class MiniPlan(BaseModel):
    requirement_id: str
    overview: str
    cases: List[str] = Field(default_factory=list)

plan = client.generate_json(
    prompt="Create a MiniPlan for REQ-123 from this context:\n" + ctx,
    schema=MiniPlan,   # validated; auto-repair loop on validation error
)
print(plan)
```

**CLI**

```bash
# Ingest
cloom ingest docs/**/*.md --strategy auto

# Retrieve & stitch a context block
cloom context "overshoot < 5%" --top-k 6 > ctx.md

# Single-shot generation
cloom generate --prompt-file prompt.txt --opt temperature=0.2

# Chat with state
cloom chat new --system "Be concise."
cloom chat send --convo-id <UUID> --message "Draft test cases" --stream

# JSON-only output
cloom json --prompt 'Return ONLY a JSON object with fields a, b'
```

---

## Core concepts

- **Corpus → Documents → Chunks → Embeddings → Index → Retrieval**
  - **Corpus:** your total body of reference content.
  - **Document:** a single logical item (file, page), stored with metadata.
  - **Chunk:** a slice of a document sized for the model (default ≈ 800 tokens, ~120 overlap).
  - **Embedding:** a vector for each chunk (default embed model: `nomic-embed-text`).
  - **Index/store:** persisted in SQLite; used to retrieve relevant chunks.
  - **Retrieval:** cosine similarity search; stitch top-K chunks into a context block.

---

## Modes

### Generation

Single-shot text completion via Ollama’s `/api/generate`.

```python
res = client.generate("Write a haiku about orbital control.")
print(res.response_text)
```

- **Streaming**: `client.generate(..., stream=True)` yields tokens as they arrive.
- **Options**: pass `options={...}` per call or set `default_options` at init.

---

### Chat

Multi-turn conversations via `/api/chat`. History is **persisted**.

```python
cid = client.new_conversation(system="Be concise and cite standards when relevant.")
r1 = client.chat(cid, "Draft test cases for REQ-045.")
r2 = client.chat(cid, "Add fuzzing strategies for sensor noise.")
print(r2.reply.content)

# Inspect conversation:
for msg in client.history(cid):
    print(msg.role, ">>", msg.content[:120])
```

Use chat when you need iterative refinement with retained context.

---

### JSON mode (schema-enforced)

Produce **strict JSON** that must validate against a Pydantic model (v1 or v2).  
If validation fails, a **repair loop** is auto-prompted (bounded retries).

```python
from typing import List
from pydantic import BaseModel, Field

class TestCase(BaseModel):
    id: str
    title: str
    steps: List[str]
    expected: List[str]

class TestPlan(BaseModel):
    requirement_id: str
    overview: str
    cases: List[TestCase] = Field(default_factory=list)

obj = client.generate_json(
    "Create a plan for REQ-045 from the context below:\n" + ctx,
    schema=TestPlan,
    # options={"temperature": 0.0},  # default forced to 0.0 for JSON mode
)
```

> Under the hood, we apply a strict “JSON-only” system message, extract a valid JSON block, and validate it with Pydantic. You can also use the server-side `format="json"` option (see [Model options](#model-options-behavior-tuning)).

---

### Ingest & Retrieval (RAG)

**Ingest**: chunk + embed + store documents.

```python
client.add_files(
    ["docs/**/*.md", "src/**/*.py"],
    strategy="auto",                 # auto | replace | skip
)
# or raw text:
doc_id, chunk_ids = client.add_text(big_text_blob, source="inline")
```

**Retrieve**: rank by cosine similarity and **stitch** a context block.

```python
ctx = client.build_context("PID overshoot tests", top_k=6)
res = client.generate(f"Using this context, propose 5 robust tests:\n{ctx}")
```

---

## Re-ingest strategies & incremental updates

`add_files(..., strategy="auto" | "replace" | "skip")`

- **auto** (default): if file unchanged (content hash) → skip. If changed → delete prior chunks under the same doc id and rebuild (embedding cache prevents re-embedding identical text).
- **replace**: always rebuild the doc’s chunks (even if unchanged).
- **skip**: if any doc exists for that source path → skip (even if changed).

Advanced: `add_text(..., doc_id=<existing>, reuse_incremental=True)` will **insert only new/changed chunks** keyed by a `chunk_hash`, keeping identical ones.

---

## Model options (behavior tuning)

You can pass options at init via `default_options={...}` or per call via `options={...}`.  
These map to Ollama’s generation parameters. Availability can vary by model.

| Option | What it controls | Typical range / notes |
|---|---|---|
| `num_ctx` | **Context window size** (tokens) | 2048–32768 (model-dependent); larger lets you stuff more context |
| `num_predict` | **Max new tokens** to generate | 64–4096; `-1` unlimited; `-2` fill to context |
| `temperature` | **Creativity vs determinism** | 0.0–1.2; lower = focused, higher = diverse |
| `top_k` | Sample from top-K candidates | 20–100; lower = safer, higher = varied |
| `top_p` | **Nucleus sampling** threshold | 0.7–0.97; keep minimal set reaching probability `p` |
| `min_p` | Minimum prob cutoff (alt to `top_p`) | 0.01–0.1 |
| `repeat_penalty` | Repetition penalty | 1.05–1.5; higher reduces repeats |
| `repeat_last_n` | Window for repeat penalty | 64–2048; `0` disables, `-1` full `num_ctx` |
| `tfs_z` | Tail-free sampling | 1 (off) – 2 |
| `mirostat` | Mirostat perplexity control | 0 off, 1 v1, 2 v2 |
| `mirostat_eta` | Mirostat learning rate | ~0.1 |
| `mirostat_tau` | Mirostat target surprise | ~5.0 |
| `seed` | Random seed | Fix for reproducible outputs |
| `stop` | **Stop sequences** | List of strings to halt generation |
| `format="json"` | Server-side JSON enforcement | Useful for structured outputs (pairs well with Pydantic) |
| `system` | System prompt (chat) | Global instructions/role |
| `template` | Model prompt template override | Advanced; model-specific |
| `suffix` | Text appended after model response | Code completion scenarios |
| `raw` | Disable template formatting | Send your prompt as-is |
| `context` | Continue from prior `/generate` | Token array returned by Ollama |
| `keep_alive` | Keep model loaded post-call | e.g. `"10m"`; `0` to unload |

**Practical presets**

- **Factual/terse**: `temperature=0.1`, `top_p=0.9`, `repeat_penalty=1.1`
- **Creative**: `temperature=0.8`, `top_p=0.95`, `top_k=100`
- **Strict JSON**: `temperature=0.0` (also enable `format="json"` if the model supports it)
- **Fast debug**: `num_predict=200`, `temperature=0.2`

---

## CLI usage (`cloom`)

The CLI mirrors the Python API. Generation options are passed via repeated `--opt key=value` or a JSON blob via `--opts-json`.

Common global flags:
- `--model` (default: `gpt-oss:20b`)
- `--host` (default: `http://localhost:11434`)
- `--db` (default: `./.ollama_client/cache.sqlite`)
- `--keep-alive` (default: `10m`)
- `--calls-per-minute` (rate limit; `0` = off)
- `--opt key=value` (repeatable), `--opts-json <JSON_OR_PATH>`

**Commands**

```bash
# Ingest files (chunk + embed + store)
cloom ingest docs/**/*.md --strategy auto --opt num_ctx=16384

# Search similar chunks
cloom search "overshoot < 5%" --top-k 6

# Build a stitched context block
cloom context "attitude controller step response" --top-k 6 --out ctx.md

# Single-shot generation (stream or non-stream)
cloom generate --prompt-file prompt.txt --opt temperature=0.2 --stream

# Chat (create/send/history)
cloom chat new --system "Be terse." --message "Hello"
cloom chat send --convo-id <UUID> --message "Add fuzz strategies" --stream
cloom chat history --convo-id <UUID>

# JSON mode (with or without Pydantic schema)
cloom json --prompt 'Return ONLY a JSON object with fields a, b'
cloom json --prompt-file spec.txt --schema mypkg.schemas:TestPlan
```

---

## Recipes

**1) Factual summary / plans (low hallucination)**  
```bash
cloom generate --prompt-file plan.txt \
  --opt temperature=0.1 --opt top_p=0.9 --opt repeat_penalty=1.1
```

**2) Creative brainstorming**  
```bash
cloom generate --prompt "Invent 5 patch ideas." \
  --opt temperature=0.8 --opt top_p=0.95 --opt top_k=100
```

**3) Structured JSON output**  
```bash
cloom json --prompt-file reqs.txt --schema mypkg.schemas:TestPlan \
  --opt temperature=0.0
```

**4) Long-context retrieval + generation**  
```bash
cloom context "overshoot < 5%" --top-k 8 | \
  cloom generate --prompt "Using the following context, propose validation steps:"
```

**5) Fast answers / debugging**  
```bash
cloom generate --prompt "Summarize 3 risks for uplink comms." \
  --opt num_predict=200 --opt temperature=0.2
```

---

## Templating

Register, list, and render prompt templates:

```bash
cloom template add --name testplan --file templates/testplan.md
cloom template list
cloom template render --name testplan --var requirement=REQ-045 --var phase=init
```

Python:

```python
client.register_template("risk", "List risks for {system} in {phase}.")
prompt = client.render_template("risk", system="uplink comms", phase="init")
res = client.generate(prompt)
```

---

## Streaming

Both `generate` and `chat` support streaming:

```python
for tok in client.generate("Write a limerick:", stream=True):
    print(tok, end="", flush=True)

cid = client.new_conversation(system="Be concise.")
for delta in client.chat(cid, "Hello!", stream=True):
    print(delta, end="", flush=True)
```

---

## Performance & scaling tips

- **Keep the model warm**: `keep_alive="10m"` avoids repeated loads for large models.
- **Chunk sizing**: default ~800 tokens with ~120 overlap balances coherence and precision.
- **Context budget**: prompt + context + response share `num_ctx`; cap `num_predict` if you pass long contexts.
- **Determinism**: set `temperature=0.0` + `seed` for reproducible, auditable runs (especially JSON mode).
- **Embedding cache**: identical text reuses vectors; even after `replace`, cache avoids re-embedding.

---

## Tests, coverage, and CI

- Tests are written with the Goal to achieve **100% branch coverage** (a practical proxy for MC/DC).  **Currently 93.43%**
- GitHub Actions workflow runs:
  - Lint (`ruff`), type check (`mypy`)
  - `pytest --cov=corpusloom --cov-branch --cov-fail-under=100`
  - Uploads HTML coverage as an artifact

Run locally:

```bash
pytest -q --cov=corpusloom --cov-branch --cov-fail-under=100
```

---

## Troubleshooting

- **Connection refused** → Ensure `ollama serve` is running and the model is pulled.
- **Model option ignored** → Some options are model-dependent; verify your model supports it.
- **Long prompts get cut** → Increase `num_ctx` and/or reduce `num_predict`.
- **JSON mode fails validation** → The client retries with a repair prompt. If it still fails, reduce `temperature` and consider adding `format="json"`.

---

## FAQ

**Q: Where is data stored?**  
A: SQLite DB at `--db` (default `./.ollama_client/cache.sqlite`) holds templates, messages, documents, chunks, and vectors (as JSON).

**Q: Which embedding model is used?**  
A: Default is `nomic-embed-text`. Override via `embed_model=` in API calls or CLI flags.

**Q: Can I swap the retriever?**  
A: Yes. The built-in retriever is cosine over stored vectors; you can replace it with FAISS/ANN later without changing high-level APIs.

**Q: Do you support server-side JSON output?**  
A: You can set `options={"format": "json"}`. The Pydantic path is still recommended for robust validation.

---

**CorpusLoom** — Local RAG + typed JSON on Ollama, wrapped in a clean Python API and `cloom` CLI.
