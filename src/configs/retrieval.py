from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    """Search backend + snippet / document truncation policy."""

    # "bm25" = Lucene; "qwen3_8b_embedding" = FAISS over pre-built Qwen3-Embedding-8B shards.
    backend: str = "bm25"

    # BCP / DR9K: path to a BM25 Lucene index (also used as the text source
    # for get_document() lookups under the dense backend).
    index_path: Optional[str] = None

    # Dense backend: directory containing Qwen3-Embedding-8B corpus shards.
    qwen3_8b_embedding_shards_dir: Optional[str] = None

    # DR9K: FastAPI URL of the external retrieval server (e.g. http://localhost:8001).
    retrieval_server_url: Optional[str] = None

    # Top-k search hits returned per query. Single field for both BCP and DR9K
    # — they share the same policy. (Was bcp_k=10 / dr9k_k=5 historically.)
    k: int = 10

    # 3-tier snippet sizes.
    snippet_new_tokens: int = 512   # docs not yet seen in this rollout
    snippet_seen_tokens: int = 128  # docs already seen (either in_context or compressed)

    # get_document() result cap.
    get_document_max_tokens: int = 8192
