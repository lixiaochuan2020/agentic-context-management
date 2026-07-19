"""Embedding helpers for DenseRetriever.

Two implementations, same interface `(texts: list[str]) -> np.ndarray`:

- `VLLMEmbedClient`: POST to a vLLM `/v1/embeddings` endpoint. Preferred for
  online rollouts where many workers share one GPU-backed encoder.
- `LocalSentenceTransformersEmbedder`: in-process `sentence_transformers`
  model. Convenient for offline indexing / small smoke tests.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import requests

logger = logging.getLogger(__name__)


class VLLMEmbedClient:
    """Thin HTTP client for a vLLM embedding endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0,
                 api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.session = requests.Session()

    def __call__(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        r = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        embs = np.asarray([item["embedding"] for item in body["data"]], dtype=np.float32)
        return embs


class LocalSentenceTransformersEmbedder:
    """In-process embedder. Loads the model into the calling process."""

    def __init__(self, model_path: str, device: str | None = None,
                 batch_size: int = 32, normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

    def __call__(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embs, dtype=np.float32)


def make_embedder_from_env() -> "VLLMEmbedClient | LocalSentenceTransformersEmbedder":
    """Auto-pick embedder based on env vars.

    `EMBED_API_BASE` set → VLLMEmbedClient.
    `EMBED_MODEL_PATH` set (and no API base) → LocalSentenceTransformersEmbedder.
    """
    api_base = os.environ.get("EMBED_API_BASE")
    if api_base:
        return VLLMEmbedClient(
            base_url=api_base,
            model=os.environ.get("EMBED_MODEL_NAME", "qwen3-embedding-8b"),
        )
    model_path = os.environ.get("EMBED_MODEL_PATH", "Qwen/Qwen3-Embedding-8B")
    return LocalSentenceTransformersEmbedder(model_path)
