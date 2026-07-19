"""BrowseComp-Plus dataset loader.

Supports two input formats:
  1. Standalone JSON file: [{id, question, answer}]  (preferred — just pass --data path)
  2. TSV queries file: loads queries.tsv + optional decrypted JSONL for answers
     (legacy — auto-detected when data_path ends in .tsv)
"""
from __future__ import annotations

import csv
import json
import os

from src.data_loading import register


@register("browsecomp-plus")
def load_browsecomp_plus(data_path: str, **kwargs) -> list[dict]:
    """Load BrowseComp-Plus dataset.

    Args:
        data_path: Path to either:
            - A JSON file [{id, question, answer}]
            - A TSV file (query_id \\t query_text), with optional jsonl_path kwarg
        **kwargs:
            jsonl_path: Path to decrypted JSONL with answers (only for TSV mode)
    """
    if data_path.endswith(".tsv"):
        return _load_from_tsv(data_path, kwargs.get("jsonl_path"))
    else:
        return _load_from_json(data_path)


def _load_from_json(data_path: str) -> list[dict]:
    """Load from a standalone JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = []
    for i, item in enumerate(raw):
        dataset.append({
            "id": str(item.get("id", item.get("query_id", f"q{i}"))),
            "question": item.get("question") or item.get("query") or item.get("problem", ""),
            "answer": item.get("answer", ""),
        })
    return dataset


def _load_from_tsv(tsv_path: str, jsonl_path: str | None = None) -> list[dict]:
    """Load from TSV queries + optional JSONL answers (legacy BCP format)."""
    answers = {}
    if jsonl_path and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                answers[str(item["query_id"])] = item["answer"]

    dataset = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            qid, query = row[0].strip(), row[1].strip()
            dataset.append({
                "id": qid,
                "question": query,
                "answer": answers.get(qid, ""),
            })
    return dataset
