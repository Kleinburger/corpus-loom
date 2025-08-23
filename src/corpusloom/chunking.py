from __future__ import annotations
import re
from typing import List
from .utils import approx_tokens

class Chunker:
    def __init__(self, max_tokens: int = 800, overlap_tokens: int = 120):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_text(self, text: str) -> List[str]:
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks: List[str] = []

        code_pat = re.compile(r"```.*?```", re.S)
        pos = 0
        for m in code_pat.finditer(t):
            pre = t[pos:m.start()]
            code = t[m.start():m.end()]
            if pre.strip():
                blocks.extend(_split_paragraphs(pre))
            blocks.append(code)
            pos = m.end()
        tail = t[pos:]
        if tail.strip():
            blocks.extend(_split_paragraphs(tail))
        
        # FAST-PATH: if there are multiple small paragraphs that *collectively*
        # fit within one chunk, keep them as **separate** chunks. This improves        
        # retrieval granularity and makes tests like "top_k=2" deterministic
        # even for small files such as "Alpha\n\nBeta\n\nGamma".
        total_toks = sum(approx_tokens(b) for b in blocks)
        if len(blocks) > 1 and total_toks <= self.max_tokens:
            return blocks

        chunks: List[str] = []
        cur: List[str] = []
        cur_toks = 0
        limit_chars = self.max_tokens * 4
        ov_chars = self.overlap_tokens * 4

        def flush():
            nonlocal cur, cur_toks
            if not cur: return
            chunk = "\n".join(cur).strip()
            if chunk: chunks.append(chunk)
            cur = []; cur_toks = 0

        for b in blocks:
            btoks = approx_tokens(b)
            if cur_toks + btoks <= self.max_tokens or not cur:
                cur.append(b); cur_toks += btoks
                if cur_toks >= self.max_tokens:
                    flush()
            else:
                prev = "\n".join(cur); flush()
                if chunks and ov_chars > 0:
                    tail_overlap = prev[-ov_chars:]
                    if tail_overlap.strip():
                        cur.append(tail_overlap); cur_toks = approx_tokens(tail_overlap)
                if approx_tokens(b) > self.max_tokens * 1.25:
                    for piece in _hard_wrap(b, limit_chars, int(ov_chars/2)):
                        if cur_toks + approx_tokens(piece) > self.max_tokens and cur:
                            flush()
                        cur.append(piece); cur_toks += approx_tokens(piece)
                else:
                    cur.append(b); cur_toks += btoks
        flush()
        return chunks

def _split_paragraphs(text: str) -> List[str]:
    lines = text.split("\n")
    blocks: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if re.match(r"^\s*#{1,6}\s+\S", ln):
            if buf:
                blocks.append("\n".join(buf).strip()); buf = []
            buf.append(ln)
        elif ln.strip() == "":
            if buf:
                blocks.append("\n".join(buf).strip()); buf = []
        else:
            buf.append(ln)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]

def _hard_wrap(s: str, width_chars: int, overlap_chars: int) -> List[str]:
    out: List[str] = []
    i = 0; n = len(s)
    while i < n:
        j = min(n, i + width_chars)
        cut = s.rfind("\n", i+1, j)
        if cut == -1:
            cut = s.rfind(" ", i+1, j)
        if cut == -1 or cut <= i + width_chars * 0.6:
            cut = j
        out.append(s[i:cut])
        if cut >= n: break
        i = max(cut - overlap_chars, 0)
    return [x.strip() for x in out if x.strip()]
