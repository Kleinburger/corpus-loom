# Contributing to CorpusLoom

Thanks for your interest in improving **CorpusLoom**! This guide explains how to set up your environment, run the test suite (including MC/DC-style coverage), follow our style rules, and submit high-quality pull requests.

---

## Quick start

```bash
# 1) Fork & clone your fork
git clone https://github.com/<you>/corpus-loom.git
cd corpus-loom

# 2) Create and activate a virtualenv (any tool is fine)
python -m venv .venv
. .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# 3) Install CorpusLoom in editable mode with dev extras
pip install -U pip
pip install -e ".[dev,json]"    # dev deps + optional pydantic for JSON-mode

# 4) Run the test suite + coverage
pytest -q
pytest --cov=corpusloom --cov-branch --cov-report=term-missing
```

Project layout:

```
src/corpusloom/         # library code
tests/                  # unit tests
pyproject.toml          # build config & entry points
```

---

## Testing (incl. MC/DC coverage)

We aim for **high branch coverage** and targeted **MC/DC-style** tests in critical modules:

- `utils.py` – deterministic tests for `RateLimiter`, `extract_json_str`, `cosine`, etc.
- `chunking.py` – drive both the “fast path”, split/overlap, and hard-wrap branches.
- `client.py` – stub network I/O to cover generate/chat (stream & non-stream), file ingest strategies, and error paths.
- `json_mode.py` – exercise repair loop (invalid → valid) with schema success/failure, both with and without Pydantic available.
- `store.py` – cover connection/context-manager behavior, template APIs, and vector updates.

Helpful commands:

```bash
# Full run with branch coverage
pytest --cov=corpusloom --cov-branch --cov-report=term-missing

# Single test file (faster inner loop)
pytest tests/test_chunking.py -q

# Stop on first failure
pytest -x
```

**Tips for MC/DC-style coverage**

- Prefer *small*, *explicit* inputs that flip one condition at a time.
- Use `monkeypatch` to isolate time or network effects (e.g., patch `time.time`, `time.sleep`, `OllamaClient._post`, etc.).
- Confirm both **true** and **false** paths of compound conditions (e.g., “if A or not B”).
- When testing error branches, assert on the **exception message** to lock in behavior.

---

## Stubbing I/O in tests

**Network**  
Never hit real network services in tests. Patch the client internals:

```python
monkeypatch.setattr(OllamaClient, "_post", stub_post, raising=False)
monkeypatch.setattr(OllamaClient, "_post_stream", stub_post_stream, raising=False)
```

**Time / Rate limiter**

```python
# Freeze time deterministically
monkeypatch.setattr("corpusloom.utils.time.time", lambda: 1000.0)
monkeypatch.setattr("corpusloom.utils.time.sleep", lambda s: None)
```

**SQLite / Store**

- Use `tmp_path` to point the DB to a fresh file per test.
- Interact with DB only via `Store` methods; if you must patch sqlite, patch where it’s **looked up** (e.g., `corpusloom.store.sqlite3.connect`).

---

## Linting & typing

We keep CI fast and strict:

```bash
# Lint
ruff check .

# Type-check
mypy --config-file mypy.ini src
```

Style rules (enforced by **ruff**):

- Imports grouped as stdlib / third-party / local.
- No unused imports, no unused variables.
- Keep functions small and focused.

Typing rules (checked by **mypy**):

- Add type hints for all public functions.
- Prefer `typing` / `collections.abc` over runtime duck-typing in API surfaces.
- Use `Optional[...]` only if `None` is truly valid and documented.

---

## Commit messages & branches

- **Conventional Commits** (enforced informally):
  - `feat: …`, `fix: …`, `docs: …`, `test: …`, `refactor: …`, `chore: …`
- Create topic branches off `main`:
  - `git checkout -b feat/better-chunking`
- Rebase before opening a PR to keep history tidy.

---

## Pull Request checklist

- [ ] Tests: added/updated and **pass locally**.
- [ ] Coverage: meaningful lines & branches covered (no dead code left untested).
- [ ] Lint & types: `ruff` and `mypy` pass.
- [ ] Docs: update README/CLI help strings if behavior changed.
- [ ] Changelog: add an entry in **“Unreleased”** if user-visible.

---

## Adding/adjusting tests by module

**`chunking.py`**
- Cover:
  - Fast path (multiple small blocks that fit in one chunk).
  - Splitting long non-code paragraphs vs. preserving fenced code.
  - Overlap logic (pre/after flush).
  - Hard-wrap path (`> 1.25 * max_tokens`) and its pre-emptive flush.

**`utils.py`**
- `extract_json_str`: with/without fences, mismatched braces, stray closers, invalid candidate.
- `RateLimiter`: refill path and sleep path via `time` monkeypatch.

**`client.py`**
- `generate` stream & non-stream; `chat` stream & non-stream.
- `add_files` strategies: `auto`, `replace`, `skip`, unchanged vs changed file hashing.
- Error paths: wrong content type from embeddings (raise), non-200 HTTP status (by stubbing).

**`json_mode.py`**
- First invalid JSON, then a valid repair; schema validate both failure and success.
- With “no Pydantic present” (monkeypatch module flags) vs with model_validate_json.

**`store.py`**
- Template CRUD (`upsert_template`, `get_template`, `list_templates`).
- Vector update (`update_chunk_vector`) path.
- Conversation history append and retrieval.
- Ensure no **ResourceWarnings**: prefer context managers or explicit closes.

---

## CLI development

The CLI entry point is `cloom`.

Local smoke-tests:

```bash
# help
python -m corpusloom.cli --help

# generate
cloom generate --prompt "Hello!"

# json (strict JSON, no schema)
echo '{"a": 1}' | cloom json

# templates
cloom template add --name greet --file tmpl.txt
cloom template list
cloom template render --name greet --var name=World
```

Test strategy:

- Simulate `sys.argv` inside tests and stub `OllamaClient` methods.
- Capture stdout with `capsys`.
- Avoid reading real files unless using `tmp_path`.

---

## Releases

We use **semantic versioning**:

- **PATCH**: bug fixes only
- **MINOR**: new features, backwards compatible
- **MAJOR**: breaking changes

**How we release**

1. Update `version` in `pyproject.toml`.
2. Update CHANGELOG.
3. Create a Git tag: `vX.Y.Z`.
4. Create a GitHub Release for that tag.
5. CI builds sdist/wheel and:
   - runs tests, lint, type-check
   - (optionally) publishes to **TestPyPI** on pre-release
   - publishes to **PyPI** on GitHub Release via **Trusted Publishers**
     - Ensure the PyPI project **name** matches `project.name` (`corpusloom`)
     - In PyPI “Manage project → Publishing → Trusted Publishers”:
       - **Repository:** `Kleinburger/corpus-loom`
       - **Environment:** GitHub Actions
       - **Workflow filename:** `.github/workflows/release.yml`
       - **Environment/Tag guard:** `refs/tags/v*` (or specific)
     - If you see *invalid-publisher* errors, the PyPI publisher claims (repo/ref/workflow) must exactly match the release workflow.

---

## Security & conduct

- Please **do not** open security issues publicly. Email the maintainer listed in `pyproject.toml`.
- This project follows the standard “be excellent to each other” code of conduct. Be kind and constructive.

---

## Questions?

Open a discussion or an issue. We’re happy to help get your PR across the finish line.
