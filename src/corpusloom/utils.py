from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import List, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


def hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return -1.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return -1.0
    return num / (da * db)


def extract_json_str(text: str) -> Optional[str]:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    stack = []
    for j in range(start, len(text)):
        ch = text[j]
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            top = stack.pop()
            if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                return None
            if not stack:
                candidate = text[start : j + 1].strip()
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    pass
    return None


class RateLimiter:
    def __init__(self, calls_per_minute: int = 0):
        self.cap = calls_per_minute or 0
        self.tokens = float(self.cap)
        self.last_refill = time.time()

    def throttle(self):
        if self.cap <= 0:
            return
        now = time.time()
        refill = (now - self.last_refill) * (self.cap / 60.0)
        if refill > 0:
            self.tokens = min(self.cap, self.tokens + refill)
            self.last_refill = now
        if self.tokens < 1.0:
            needed = (1.0 - self.tokens) / (self.cap / 60.0)
            time.sleep(max(0.0, needed))
            self.tokens = 0.0
            self.last_refill = time.time()
        else:
            self.tokens -= 1.0
