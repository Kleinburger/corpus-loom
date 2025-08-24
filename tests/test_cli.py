import sys
import types
import builtins
import importlib
import pytest

# NOTE: these tests assume your CLI module exposes a `main()` that reads sys.argv (argparse-style).
# If the CLI is different, adjust the argv arrays accordingly.

class DummyClient:
    def __init__(self, *a, **kw): pass
    def add_files(self, *a, **k): return [("doc1", ["c1"])]
    def build_context(self, *a, **k): return "[CTX 1]\nHello"
    def generate(self, *a, **k):
        class R: response_text = "ok"
        return R()
    def new_conversation(self, *a, **k): return "convo-1"
    def chat(self, *a, **k):
        class R: reply = types.SimpleNamespace(content="hi")
        return R()
    def json_mode(self): pass  # not reached in this smoke

def run(argv):
    sys_argv = list(sys.argv)
    try:
        sys.argv = argv
        import corpusloom.cli as cli
        cli.OllamaClient = DummyClient
        try:
            cli.main()
        except SystemExit as e:
            # Allow --help to exit cleanly
            if e.code not in (0, None):
                raise
    finally:
        sys.argv = sys_argv
        importlib.invalidate_caches()

def test_cli_help_runs(capsys):
    run(["cloom", "--help"])
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "usage" in out.lower()

def test_cli_ingest_runs(capsys, tmp_path):
    f = tmp_path / "a.md"; f.write_text("# A")
    run(["cloom", "ingest", str(f)])
    out = capsys.readouterr().out
    assert "ingest" in out.lower() or out != ""  # just smoke

def test_cli_context_and_generate_runs(capsys):
    run(["cloom", "context", "Alpha", "--top-k", "1"])
    run(["cloom", "generate", "--prompt", "Hello"])