import json
import types
import pytest

from corpusloom.json_mode import JsonMode, ValidationFailedError
from corpusloom import OllamaClient

# We'll stub _post to first return invalid JSON then valid JSON to exercise repair loop.
class StubClient(OllamaClient):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._calls = 0
    def _post(self, path, payload):
        self._calls += 1
        if path.endswith("/api/chat"):
            if self._calls == 1:
                # invalid (missing key 'b')
                return {"message": {"content": "{\"a\": 1}"}, "model": self.model}
            else:
                return {"message": {"content": "{\"a\": 1, \"b\": 2}"}, "model": self.model}
        raise AssertionError("unexpected path")

def test_generate_json_with_retry(monkeypatch, tmp_path):
    c = StubClient(cache_db_path=str(tmp_path / "db.sqlite"))
    jm = c.json_mode

    # Bypass real pydantic requirement
    monkeypatch.setattr(jm, "_ensure_pydantic", lambda: None, raising=True)

    # Create a dummy "schema" with model_validate_json that enforces keys a and b.
    class DummySchema:
        @staticmethod
        def model_validate_json(s: str):
            obj = json.loads(s)
            if "a" not in obj or "b" not in obj:
                # Use the ValidationError symbol that JsonMode imported (present even without pydantic)
                from corpusloom.json_mode import ValidationError
                raise ValidationError("missing keys")
            return obj

    out = jm.generate_json("return object with a,b", schema=DummySchema)
    assert out == {"a": 1, "b": 2}
