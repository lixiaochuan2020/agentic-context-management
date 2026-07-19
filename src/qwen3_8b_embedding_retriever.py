"""Qwen3-8B-Embedding retriever for BrowseComp-Plus.

Corpus embeddings are pre-built by BCP and shipped under
`indexes/qwen3-embedding-8b/corpus.shard*_of_*.pkl` — each shard is a
`(np.ndarray[float32, (n, 4096)], list[docid])` tuple. We concatenate the
shards into one FAISS IndexFlatIP for cosine similarity (embeddings are
already L2-normalized by the embedding model).

For document text lookup we duck-type the existing BM25Searcher's
`.get_document(docid)`. No need to re-store text.

Query-time embedding goes through the user-supplied callable so the same
retriever works with either:
  - an HTTP `/v1/embeddings` endpoint backed by vLLM (preferred for online
    multi-worker rollouts), or
  - a local sentence_transformers instance (single-process indexing/eval).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


class Qwen3_8B_EmbeddingRetriever:
    """FAISS-backed Qwen3-8B-Embedding retriever duck-typed to BM25Searcher."""

    def __init__(
        self,
        shards_dir: str | Path,
        text_searcher,
        embed_fn: Callable[[list[str]], np.ndarray],
        query_prompt: str = _DEFAULT_QUERY_PROMPT,
    ):
        """
        shards_dir: contains `corpus.shard*_of_*.pkl` files.
        text_searcher: anything with `.get_document(docid) -> {"docid","text"} | None`;
            typically an existing BM25Searcher, since corpus text is already
            stored in the BM25 Lucene index.
        embed_fn: takes list[str] of QUERIES (already prepended with
            `query_prompt`) and returns (n, dim) float32 numpy array,
            L2-normalized.
        """
        self.shards_dir = Path(shards_dir)
        self.text_searcher = text_searcher
        self._embed_fn = embed_fn
        self._query_prompt = query_prompt
        self._index: faiss.Index | None = None
        self._docids: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        shards = sorted(self.shards_dir.glob("corpus.shard*_of_*.pkl"))
        if not shards:
            raise FileNotFoundError(f"no shards in {self.shards_dir}")
        embs_chunks: list[np.ndarray] = []
        docids: list[str] = []
        for fp in shards:
            with open(fp, "rb") as f:
                arr, ids = pickle.load(f)
            arr = np.asarray(arr, dtype=np.float32)
            # Ensure L2-normalized so inner product == cosine.
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            embs_chunks.append(arr)
            docids.extend(str(d) for d in ids)
            logger.info("loaded shard %s: %d × %d", fp.name, arr.shape[0], arr.shape[1])
        embs = np.concatenate(embs_chunks, axis=0)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        self._index = index
        self._docids = docids
        logger.info("dense retriever ready: %d docs × %d dim", embs.shape[0], dim)

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k {docid, text, score} hits.

        Mirrors BM25Searcher.search return shape.
        """
        q_text = self._query_prompt + query
        q_emb = self._embed_fn([q_text])
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.ndim == 1:
            q_emb = q_emb[None, :]
        # Normalize query embedding (defensive).
        norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_emb = q_emb / norms

        scores, idxs = self._index.search(q_emb, k)
        scores = scores[0]
        idxs = idxs[0]

        hits: list[dict] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            docid = self._docids[i]
            doc = self.text_searcher.get_document(docid)
            text = doc["text"] if doc else ""
            hits.append({"docid": docid, "text": text, "score": float(score)})
        return hits

    def get_document(self, docid: str) -> dict | None:
        """Pass through to the text-bearing searcher (e.g. BM25 Lucene)."""
        return self.text_searcher.get_document(str(docid))


# ── Process-level cache ────────────────────────────────────
# Building the FAISS index loads 1.6 GB of shards + L2-normalizes; this is
# slow (~30-50 s) and pointless to repeat per question. Cache per shards_dir
# so a single rollout-shard subprocess pays the cost once.

_RETRIEVER_CACHE: dict[str, "Qwen3_8B_EmbeddingRetriever"] = {}


def get_or_build(
    shards_dir: str | Path,
    text_searcher,
    embed_fn,
    query_prompt: str = _DEFAULT_QUERY_PROMPT,
) -> "Qwen3_8B_EmbeddingRetriever":
    """Return a cached retriever, or build + cache one for this shards_dir.

    The cached instance's `text_searcher` and `embed_fn` are refreshed on every
    call so different questions can swap them out (rare, but supported).
    """
    key = str(shards_dir)
    cached = _RETRIEVER_CACHE.get(key)
    if cached is not None:
        cached.text_searcher = text_searcher
        cached._embed_fn = embed_fn
        return cached
    r = Qwen3_8B_EmbeddingRetriever(shards_dir, text_searcher, embed_fn, query_prompt)
    _RETRIEVER_CACHE[key] = r
    return r
