from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .store import Store
from .utils import cosine


class Retriever:
    def __init__(self, store: Store):
        self.store = store

    def rank_chunks(self, qvec: List[float]) -> List[Tuple[float, Dict[str, Any]]]:
        results: List[Tuple[float, Dict[str, Any]]] = []
        for row in self.store.iter_chunks():
            vec_json = row[4]
            vec = json.loads(vec_json)
            sim = cosine(qvec, vec)
            results.append(
                (
                    sim,
                    {
                        "chunk_id": row[0],
                        "doc_id": row[1],
                        "idx": row[2],
                        "text": row[3],
                        "meta": json.loads(row[5] or "{}"),
                        "score": sim,
                    },
                )
            )
        results.sort(key=lambda x: x[0], reverse=True)
        return results
