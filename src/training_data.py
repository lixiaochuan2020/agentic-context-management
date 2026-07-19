"""Training data recording for context management calls.

Every time the model invokes summary / offload / manage_context,
we snapshot the full context before & after execution and persist
it as an independent JSON file for downstream SFT / analysis.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def save_training_sample(
    out_dir: str,
    question_id: str,
    call_index: int,
    question: str,
    correct_answer: str,
    mode: str,
    turn: int,
    context_before: list[dict],
    tokens_before: int,
    tool_name: str,
    tool_args: dict,
    success: bool,
    feedback: str,
    tool_result_content: str | None,
    context_after: list[dict],
    tokens_after: int,
    metadata: dict | None = None,
) -> str:
    """Save one training sample and return the file path.

    Parameters
    ----------
    out_dir : root training-data directory for this config
        e.g. results/deepsearchqa/.../training_data/memtool_full_free_L_0_Se_sep_Su_int
    question_id : e.g. "q0"
    call_index : 0-based sequential index of this CM call within the question
    question : original question text
    correct_answer : ground-truth answer
    mode : "memtool"
    turn : turn number in the runner loop
    context_before : messages list *before* CM execution (includes assistant tool_call msg)
    tokens_before : token count of context_before
    tool_name : "summary" / "offload" / "manage_context"
    tool_args : full arguments dict from the model
    success : whether the CM operation succeeded
    feedback : the feedback message appended after execution
    tool_result_content : the actual content produced by the CM tool
        (e.g. the summary text for summary / manage_context, None for offload)
    context_after : messages list *after* CM execution + feedback
    tokens_after : token count of context_after
    metadata : extra info (search_strategy, model, benchmark, …)

    Returns
    -------
    str : path of the saved JSON file
    """
    sample_dir = os.path.join(out_dir, question_id)
    os.makedirs(sample_dir, exist_ok=True)

    sample = {
        "question_id": question_id,
        "question": question,
        "correct_answer": correct_answer,
        "mode": mode,
        "turn": turn,
        "call_index": call_index,
        "context_before": context_before,
        "tokens_before": tokens_before,
        "tool_call": {
            "name": tool_name,
            "arguments": tool_args,
        },
        "tool_result": {
            "success": success,
            "content": tool_result_content,
            "feedback_message": feedback,
        },
        "context_after": context_after,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_before - tokens_after,
        "metadata": {
            **(metadata or {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    path = os.path.join(sample_dir, f"cm_{call_index:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    logger.info("  [training_data] saved %s", path)
    return path
