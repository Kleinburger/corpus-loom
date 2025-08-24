import builtins
import importlib
import json
from types import SimpleNamespace

import pytest

import corpusloom.json_mode as jm_mod
from corpusloom.exceptions import ValidationFailedError
from corpusloom.json_mode import JsonMode

# ---------- Helpers


def _stub_jm(post, keep_alive=None, default_options=None, get_system=lambda _: "", history=()):
    """Build a JsonMode with simple stubs for all callables."""

    def get_hist(_):
        return list(history)

    appends = []

    def append(cid, role, content, meta=None):
        appends.append((cid, role, content, meta or {}))

    jm = JsonMode(
        post_func=post,
        get_history_func=get_hist,
        append_message_func=append,
        get_system_func=lambda cid: get_system(cid),
        model="test-model",
        keep_alive=keep_alive,
        default_options=default_options or {},
    )
    jm._appends = appends  # expose for assertions
    return jm


def test_pydantic_fallback_branch_then_restore(monkeypatch):
    # Force ImportError when importing 'pydantic', then reload module.
    orig_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "pydantic":
            raise ImportError("nope")
        return orig_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import, raising=True)

    mod = importlib.reload(jm_mod)
    try:
        assert mod._PYDANTIC_AVAILABLE is False

        # BaseModel is the lightweight fallback; ValidationError is Exception
        class M(mod.BaseModel):  # should be subclassable
            pass

        assert issubclass(mod.ValidationError, Exception)
        # _ensure_pydantic() must raise when fallback is active
        with pytest.raises(RuntimeError):
            mod.JsonMode(
                lambda *a, **k: None,
                lambda *_: [],
                lambda *a, **k: None,
                lambda *_: "",
                "m",
                None,
                {},
            ).generate_json("x", schema=M)
    finally:
        # Restore real module with real import
        monkeypatch.setattr(builtins, "__import__", orig_import, raising=True)
        importlib.reload(jm_mod)  # back to normal for subsequent tests


def test_generate_json_parse_raw_and_options_merged(tmp_path):
    captured = {}

    def post(path, payload):
        captured["payload"] = payload
        # Candidate is valid JSON straight away (also exercises extract_json_str pass-through)
        return {"message": {"content": '{"ok": true}'}, "model": "test-model"}

    jm = _stub_jm(post, keep_alive="1m", default_options={"top_p": 0.9, "temperature": 0.7})

    # Bypass real pydantic requirement so test runs regardless of environment
    jm._ensure_pydantic = lambda: None  # type: ignore[attr-defined]

    class SchemaParseRaw:
        @staticmethod
        def parse_raw(s: str):
            obj = json.loads(s)
            obj["parsed_with"] = "parse_raw"
            return obj

    out = jm.generate_json("do it", schema=SchemaParseRaw)
    assert out == {"ok": True, "parsed_with": "parse_raw"}

    # Options must be merged and temperature forced to 0.0; keep_alive propagated
    pl = captured["payload"]
    assert pl["keep_alive"] == "1m"
    assert pl["options"]["top_p"] == 0.9
    assert pl["options"]["temperature"] == 0.0


def test_generate_json_not_json_then_validation_error_then_ok(monkeypatch):
    calls = {"n": 0}

    def post(path, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"message": {"content": "not json at all"}, "model": "test-model"}
        if calls["n"] == 2:
            return {"message": {"content": '{"a": 1}'}, "model": "test-model"}  # missing b
        return {"message": {"content": '{"a": 1, "b": 2}'}, "model": "test-model"}

    jm = _stub_jm(post)
    jm._ensure_pydantic = lambda: None  # bypass dependency

    class SchemaMVJ:
        @staticmethod
        def model_validate_json(s: str):
            obj = json.loads(s)
            if "a" not in obj or "b" not in obj:
                # Raise arbitrary exception type to exercise broad except
                raise RuntimeError("missing keys")
            return obj

    out = jm.generate_json("return a,b", schema=SchemaMVJ, retries=3)
    assert out == {"a": 1, "b": 2}
    assert calls["n"] >= 3  # we hit both "not json" and "validation error" branches


def test_generate_json_exhaustion_raises():
    def post(path, payload):
        return {"message": {"content": "still not json"}, "model": "test-model"}

    jm = _stub_jm(post)
    jm._ensure_pydantic = lambda: None  # bypass dependency

    class SchemaMVJ:
        @staticmethod
        def model_validate_json(s: str):
            return json.loads(s)

    with pytest.raises(ValidationFailedError):
        jm.generate_json("won't fix", schema=SchemaMVJ, retries=1)


def test_chat_json_repair_and_appends_recorded():
    calls = {"n": 0}

    def post(path, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            # non-JSON to trigger repair
            return {"message": {"content": "```txt\nnope\n```"}, "model": "test-model"}
        return {"message": {"content": '{"k": 3}'}, "model": "test-model"}

    hist = [SimpleNamespace(role="user", content="previous")]
    jm = _stub_jm(post, get_system=lambda cid: "SYS", history=hist)
    jm._ensure_pydantic = lambda: None  # bypass dependency

    class SchemaMVJ:
        @staticmethod
        def model_validate_json(s: str):
            return json.loads(s)

    out = jm.chat_json("cid-1", "now", schema=SchemaMVJ, retries=2)
    assert out == {"k": 3}

    # Two appends: input (user) and final assistant output
    kinds = [meta.get("kind") for (_, _, _, meta) in jm._appends]
    assert kinds == ["chat_json_input", "chat_json_output"]


def test_chat_json_exhaustion_raises():
    def post(path, payload):
        # always returns non-JSON so extract_json_str yields ""
        return {"message": {"content": "not {json}"}, "model": "test-model"}

    jm = _stub_jm(post)
    jm._ensure_pydantic = lambda: None

    class SchemaMVJ:
        @staticmethod
        def model_validate_json(s: str):
            return json.loads(s)

    with pytest.raises(ValidationFailedError):
        jm.chat_json("cid-2", "msg", schema=SchemaMVJ, retries=1)
