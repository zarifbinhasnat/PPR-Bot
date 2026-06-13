"""Step 7b: cross-encoder reranking (BAAI/bge-reranker-v2-m3).

Hybrid search is fast but approximate: the embedder encodes the query and each
chunk *separately* (a "bi-encoder"), so it never directly compares them — it
just hopes their independent vectors line up. Good for scanning thousands of
chunks cheaply, but imprecise at the top.

A *cross-encoder* reranker fixes the ordering of the top candidates. It feeds
the query AND a chunk together into the model, which attends across both and
outputs a single relevance score. This is far more accurate but far slower —
so we only run it on the ~20-30 candidates that hybrid search already
shortlisted, never the whole corpus. This "retrieve cheap, rerank precise"
two-stage pattern is standard in modern RAG.

bge-reranker-v2-m3 is multilingual (Bangla + English), matching our embedder.
"""

import os
from functools import cached_property

import torch
from sentence_transformers import CrossEncoder

from ppr_bot.config import settings


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        # reranker_model_ref resolves to a local dir if the weights were
        # pre-downloaded, else the HF repo id.
        self.model_name = model_name or settings.reranker_model_ref

    @cached_property
    def _model(self) -> CrossEncoder:
        # Use all physical CPU cores — torch otherwise defaults to half, which
        # roughly doubles latency for this 568M cross-encoder on CPU.
        torch.set_num_threads(os.cpu_count() or 4)
        # max_length caps the (query + chunk) token window the cross-encoder
        # scores. The default (512) is ~2x slower on CPU; 256 keeps the rule's
        # key content (title + opening clauses sit first) while making
        # reranking interactive on this hardware.
        return CrossEncoder(self.model_name, max_length=256, device="cpu")

    def rerank(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        """Reorder candidate chunks by cross-encoder relevance to the query.

        Args:
            query: the (standalone) user query.
            candidates: chunk dicts, each with a "text" field.
            top_k: how many to keep after reranking.

        Returns the top_k candidates, each annotated with a "rerank_score".
        """
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
