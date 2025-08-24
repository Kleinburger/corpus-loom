import importlib
import io
import json
import runpy
import sys
import types

import pytest

TEMPLATES_STORE = {}


class StubClient:
    """Stub for cli.OllamaClient that supports all subcommands."""

    last_opts = None  # capture default_options from _build_client

    def __init__(self, **kw):
        type(self).last_opts = kw.get("default_options", {})
        self.opts = kw
        self._history = [
            types.SimpleNamespace(role="user", content="hello"),
            types.SimpleNamespace(role="assistant", content='{"ok":true}'),
        ]
        self.templates = {}

    # ingest / corpus
    def add_files(
        self, paths, *, encoding="utf-8", embed_model="nomic-embed-text", strategy="auto"
    ):
        # pretend we chunked each file into 2 chunks
        return [(f"doc-{i + 1}", ["c1", "c2"]) for i, _ in enumerate(paths)]

    # search/context
    def search_similar(self, q, *, embed_model="nomic-embed-text", top_k=5):
        return [{"text": "hit", "score": 0.99, "source": "src"} for _ in range(top_k)]

    def build_context(self, q, *, top_k=5, embed_model="nomic-embed-text"):
        lines = [f"[CTX {i + 1} | score=0.99 | src=src]\nchunk-{i + 1}" for i in range(top_k)]
        return "\n".join(lines)

    # generate
    class _GenRes:
        response_text = "GEN_OK"

    def generate(self, prompt, stream=False):
        if stream:
            return iter(["A", "B", "C"])
        return self._GenRes()

    # chat
    def new_conversation(self, name=None, system=None):
        return "convo-1"

    class _ChatRes:
        def __init__(self, text):
            self.reply = types.SimpleNamespace(content=text)

    def chat(self, convo_id, message, stream=False):
        if stream:
            return iter(["x", "y"])
        return self._ChatRes(f"echo:{message}")

    def history(self, convo_id):
        return list(self._history)

    # json mode
    def generate_json(self, *, prompt, schema):
        class Obj:
            def model_dump_json(self, indent=2):
                return json.dumps({"from_schema": True, "prompt": prompt}, indent=indent)

        return Obj()

    # templates
    def register_template(self, template_name, text):
        TEMPLATES_STORE[template_name] = text

    def list_templates(self):
        return dict(TEMPLATES_STORE)

    def render_template(self, template_name, **vars):
        return TEMPLATES_STORE[template_name].format(**vars)


def run_with_cli(argv, stdin_text=None):
    import importlib
    import sys

    import corpusloom.cli as cli

    old_argv = sys.argv
    try:
        sys.argv = argv
        cli.OllamaClient = StubClient
        if stdin_text is not None:
            import io

            old_stdin = sys.stdin
            try:
                sys.stdin = io.StringIO(stdin_text)
                cli.main()
            finally:
                sys.stdin = old_stdin
        else:
            cli.main()
    except SystemExit as e:
        # allow normal argparse exits
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv
        importlib.invalidate_caches()


def test_cli_top_help(capsys):
    run_with_cli(["cloom", "--help"])
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "usage" in out.lower()


def test_cli_run_as_module_main(monkeypatch):
    # Covers: if __name__ == "__main__": SystemExit(main())
    monkeypatch.setattr(sys, "argv", ["ocp", "--help"])
    with pytest.raises(SystemExit) as ei:
        runpy.run_module("corpusloom.cli", run_name="__main__")
    assert ei.value.code in (0, None)


def test_cli_parse_options_variants(tmp_path, capsys):
    opts_file = tmp_path / "opts.json"
    opts_file.write_text(json.dumps({"file_opt": 7, "flag": True}), encoding="utf-8")

    # includes: int, float, bool, list, bare (warning), opts-json (path), opts-json (string)
    run_with_cli(
        [
            "cloom",
            "generate",
            "--prompt",
            "X",
            "--opt",
            "top_k=5",
            "--opt",
            "temperature=0.25",
            "--opt",
            "debug=true",
            "--opt",
            "modes=a,b,c",
            "--opt",
            "bare",  # triggers warning
            "--opts-json",
            str(opts_file),
        ]
    )
    # also ensure JSON string path is parsed
    run_with_cli(
        ["cloom", "generate", "--prompt", "Y", "--opts-json", '{"alpha": 1, "beta": false}']
    )

    # non-dict opts-json -> SystemExit
    with pytest.raises(SystemExit):
        run_with_cli(["cloom", "generate", "--prompt", "Z", "--opts-json", '["not", "a", "dict"]'])

    err = capsys.readouterr().err
    assert "ignoring --opt value" in err


def test_cli_ingest_search_context_with_out(tmp_path, capsys):
    f = tmp_path / "a.md"
    f.write_text("# A", encoding="utf-8")

    # ingest
    run_with_cli(["cloom", "ingest", str(f)])
    out1 = capsys.readouterr().out
    assert '"ingested"' in out1

    # search
    run_with_cli(["cloom", "search", "query", "--top-k", "3"])
    out2 = capsys.readouterr().out
    assert '"score":' in out2

    # context with --out
    out_path = tmp_path / "ctx.txt"
    run_with_cli(["cloom", "context", "Alpha", "--top-k", "2", "--out", str(out_path)])
    out3 = capsys.readouterr().out
    assert "wrote context" in out3
    assert out_path.read_text(encoding="utf-8").startswith("[CTX 1")


def test_cli_generate_from_stdin_streaming(capsys):
    # no --prompt/--prompt-file => stdin used; stream branch prints tokens then newline
    run_with_cli(["cloom", "generate", "--stream"], stdin_text="Hello stdin")
    out = capsys.readouterr().out
    assert out == "ABC\n" or out.strip() == "ABC"


def test_cli_generate_no_prompt_raises():
    # empty stdin and no prompt -> SystemExit
    with pytest.raises(SystemExit):
        run_with_cli(["cloom", "generate"], stdin_text="")


def test_cli_chat_new_send_history(capsys):
    # new + auto message
    run_with_cli(["cloom", "chat", "new", "--message", "hey"])
    out1 = capsys.readouterr().out
    assert '"convo_id":' in out1 and "echo:hey" in out1

    # send non-stream
    run_with_cli(["cloom", "chat", "send", "--convo-id", "convo-1", "--message", "ping"])
    out2 = capsys.readouterr().out
    assert "echo:ping" in out2

    # send stream
    run_with_cli(
        ["cloom", "chat", "send", "--convo-id", "convo-1", "--message", "pong", "--stream"]
    )
    out3 = capsys.readouterr().out
    assert out3.strip().endswith("y")  # streamed "x","y"

    # history
    run_with_cli(["cloom", "chat", "history", "--convo-id", "convo-1"])
    out4 = capsys.readouterr().out
    assert "user: hello" in out4 and 'assistant: {"ok":true}' in out4


def test_cli_json_without_schema_uses_extract(capsys):
    run_with_cli(["cloom", "json", "--prompt", '{"a":1}'])
    out = capsys.readouterr().out
    assert out.strip().startswith("{")  # printed raw JSON (already valid)


def test_cli_json_with_schema_ok(monkeypatch, capsys):
    # fabricate a module with a class name, then ensure cli imports it successfully
    mod = types.ModuleType("tmp_schema_mod")

    class DummyModel: ...

    mod.DummyModel = DummyModel
    sys.modules["tmp_schema_mod"] = mod
    run_with_cli(["cloom", "json", "--prompt", "X", "--schema", "tmp_schema_mod:DummyModel"])
    out = capsys.readouterr().out
    assert '"from_schema": true' in out


def test_cli_json_with_schema_import_error():
    with pytest.raises(SystemExit):
        run_with_cli(["cloom", "json", "--prompt", "X", "--schema", "nope_mod:Nope"])


@pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
def test_cli_template_add_list_render(tmp_path, capsys):
    # Reset global store
    TEMPLATES_STORE.clear()

    tf = tmp_path / "t.txt"
    tf.write_text("Hello {name}", encoding="utf-8")

    run_with_cli(["cloom", "template", "add", "--name", "t", "--file", str(tf)])
    out1 = capsys.readouterr().out
    assert "template 't' registered" in out1

    run_with_cli(["cloom", "template", "list"])
    out2 = capsys.readouterr().out
    data = json.loads(out2)
    assert "t" in data

    run_with_cli(["cloom", "template", "render", "--name", "t", "--var", "name=World"])
    out3 = capsys.readouterr().out.strip()
    assert out3 == "Hello World"


def test_cmd_template_fallback_return_1(monkeypatch):
    # Patch ArgumentParser.parse_args inside the function to bypass required subparser
    import corpusloom.cli as cli

    cli.OllamaClient = StubClient

    def fake_parse_args(self, argv):
        # subcmd unrecognized; include all common args to pass _build_client
        return types.SimpleNamespace(
            subcmd="unknown",
            model="m",
            host="h",
            db=":memory:",
            keep_alive="0",
            calls_per_minute=0,
            opt=[],
            opts_json=None,
            name=None,
            file=None,
            var=[],  # template branch args
        )

    monkeypatch.setattr(
        importlib.import_module("argparse").ArgumentParser, "parse_args", fake_parse_args
    )
    assert cli.cmd_template([]) == 1


def test_cmd_chat_fallback_return_1(monkeypatch):
    import corpusloom.cli as cli

    cli.OllamaClient = StubClient

    def fake_parse_args(self, argv):
        return types.SimpleNamespace(
            subcmd="weird",
            model="m",
            host="h",
            db=":memory:",
            keep_alive="0",
            calls_per_minute=0,
            opt=[],
            opts_json=None,
            convo_id="c",
            message="hi",
            stream=False,
            name=None,
            system=None,
        )

    monkeypatch.setattr(
        importlib.import_module("argparse").ArgumentParser, "parse_args", fake_parse_args
    )
    assert cli.cmd_chat([]) == 1


def test_main_unknown_cmd_returns_1(monkeypatch):
    # Force parse_known_args to produce an unknown cmd
    import corpusloom.cli as cli

    def fake_parse_known_args(self, argv):
        ns = types.SimpleNamespace(cmd="unknown")
        return ns, []

    monkeypatch.setattr(
        importlib.import_module("argparse").ArgumentParser,
        "parse_known_args",
        fake_parse_known_args,
    )

    assert cli.main(["bogus"]) == 1


def test_read_prompt_from_file(tmp_path, capsys):
    p = tmp_path / "p.txt"
    p.write_text("FROM_FILE", encoding="utf-8")
    run_with_cli(["cloom", "generate", "--prompt-file", str(p)])
    out = capsys.readouterr().out
    assert "GEN_OK" in out


def test_read_prompt_no_input_raises(monkeypatch):
    import corpusloom.cli as cli

    class S:
        def __init__(self, **kw):
            pass

        class R:
            response_text = "OK"

        def generate(self, prompt, stream=False):
            return self.R()

    cli.OllamaClient = S
    # no prompt args and empty stdin -> SystemExit with message
    monkeypatch.setattr(sys, "argv", ["cloom", "generate"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(SystemExit):
        cli.main()
