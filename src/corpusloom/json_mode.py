from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .exceptions import ValidationFailedError
from .utils import extract_json_str

try:
    from pydantic import BaseModel, ValidationError

    _PYDANTIC_AVAILABLE = True
except Exception:

    class BaseModel: ...  # type: ignore

    class ValidationError(Exception): ...

    _PYDANTIC_AVAILABLE = False

JSON_SYSTEM = (
    "You are a strict JSON generator. Return ONLY valid JSON that conforms exactly to the schema.\n"
    "Do not include prose, code fences, or explanations. No trailing commas. Use explicit nulls "
    "where needed."
)


class JsonMode:
    def __init__(
        self,
        post_func,
        get_history_func,
        append_message_func,
        get_system_func,
        model: str,
        keep_alive: Optional[str],
        default_options: Dict[str, Any],
    ):
        self._post = post_func
        self._get_history = get_history_func
        self._append = append_message_func
        self._get_system = get_system_func
        self.model = model
        self.keep_alive = keep_alive
        self.default_options = default_options

    def _ensure_pydantic(self):
        if not _PYDANTIC_AVAILABLE:
            raise RuntimeError(
                "Install pydantic to use JSON validation: pip install 'pydantic>=1.10'"
            )

    def _pydantic_from_json(self, schema: Type[BaseModel], json_str: str):
        if hasattr(schema, "model_validate_json"):
            return schema.model_validate_json(json_str)
        return schema.parse_raw(json_str)

    def generate_json(
        self,
        prompt: str,
        schema: Type[BaseModel],
        options: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ):
        self._ensure_pydantic()
        messages = [
            {"role": "system", "content": JSON_SYSTEM},
            {"role": "user", "content": "Produce JSON for this task:\n" + prompt},
        ]
        opts = {**self.default_options, **(options or {}), "temperature": 0.0}
        last_raw = ""
        for _ in range(retries + 1):
            res = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": messages,
                    "options": opts,
                    "keep_alive": self.keep_alive,
                    "stream": False,
                },
            )
            candidate = res.get("message", {}).get("content", "")
            last_raw = candidate
            jtxt = extract_json_str(candidate)
            if not jtxt:
                messages += [
                    {"role": "assistant", "content": candidate},
                    {"role": "user", "content": "Return ONLY valid JSON."},
                ]
                continue
            try:
                return self._pydantic_from_json(schema, jtxt)
            except Exception as ve:
                messages += [
                    {"role": "assistant", "content": candidate},
                    {
                        "role": "user",
                        "content": f"Validation error:\n{ve}\nReturn ONLY corrected JSON.",
                    },
                ]
        raise ValidationFailedError("Failed to produce valid JSON:\n" + last_raw)

    def chat_json(
        self,
        convo_id: str,
        user_message: str,
        schema: Type[BaseModel],
        options: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ):
        self._ensure_pydantic()
        system = self._get_system(convo_id) or ""
        sys_combo = (JSON_SYSTEM + "\n\n" + system).strip()
        self._append(convo_id, "user", user_message, {"kind": "chat_json_input"})
        msgs = [{"role": "system", "content": sys_combo}] + [
            {"role": m.role, "content": m.content} for m in self._get_history(convo_id)
        ]
        opts = {**self.default_options, **(options or {}), "temperature": 0.0}
        last_raw = ""
        for _ in range(retries + 1):
            res = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": msgs,
                    "options": opts,
                    "keep_alive": self.keep_alive,
                    "stream": False,
                },
            )
            candidate = res.get("message", {}).get("content", "")
            last_raw = candidate
            jtxt = extract_json_str(candidate)
            if not jtxt:
                msgs += [
                    {"role": "assistant", "content": candidate},
                    {"role": "user", "content": "Return ONLY valid JSON."},
                ]
                continue
            try:
                obj = self._pydantic_from_json(schema, jtxt)
                self._append(convo_id, "assistant", candidate, {"kind": "chat_json_output"})
                return obj
            except Exception as ve:
                msgs += [
                    {"role": "assistant", "content": candidate},
                    {
                        "role": "user",
                        "content": f"Validation error:\n{ve}\nReturn ONLY corrected JSON.",
                    },
                ]
        raise ValidationFailedError("Failed to produce valid JSON:\n" + last_raw)
