"""BrowseComp dataset loader.

Expects a JSON file: [{id, problem/question, answer, ...}]
Normalizes to standard format with 'question' field.
"""
from __future__ import annotations

import json

from src.data_loading import register


@register("browsecomp")
def load_browsecomp(data_path: str, **kwargs) -> list[dict]:
    """Load BrowseComp dataset from a JSON file.

    Supports both 'problem' and 'question' field names.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = []
    for i, item in enumerate(raw):
        dataset.append({
            "id": item.get("id", f"q{i}"),
            "question": item.get("problem") or item.get("question", ""),
            "answer": item.get("answer", ""),
        })
    return dataset
