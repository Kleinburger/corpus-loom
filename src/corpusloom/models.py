from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class GenerateResult:
    model: str
    prompt: str
    response_text: str
    total_duration_ms: int
    eval_count: Optional[int] = None
    eval_duration_ms: Optional[int] = None
    context: Optional[List[int]] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class ChatResult:
    model: str
    messages: List[Message]
    reply: Message
    total_duration_ms: int
    eval_count: Optional[int] = None
    eval_duration_ms: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None
