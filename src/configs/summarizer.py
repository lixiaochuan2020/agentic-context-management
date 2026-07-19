from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SummarizerConfig:
    """LLM used for manage_context compression and query_memory recall."""

    model: str = "gpt-5.4-mini"
    api_base: str = "https://api.openai.com/v1"
