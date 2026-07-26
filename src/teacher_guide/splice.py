"""Splice a teacher annotation into a ReAct initial trajectory.

Produces an `initial_history` (and aligned `initial_raw_history`) suitable for
`src.runner.run(..., config.initial_history=...)`. The student then resumes
from the synthetic teacher-authored mc/qm turn.

Design constraints (see experiments/20260517.md § Exp2):
  - Input initial is ReAct (2-tool); we swap `history[0]` to the 4-tool system
    prompt so the resumed student has the mc/qm strategy guidance.
  - `initial_raw_history = initial_history` (decision d1: initial was uncompressed,
    so the raw archive and live view are identical at splice time).
  - `initial_last_boundary_pos = 2` (system + user before any mc).
  - `initial_summary_id = 0` (no summaries yet — runner will produce id=1 when
    it executes the spliced mc on turn 0).
"""
from __future__ import annotations

import json
from typing import Optional

from src.prompts import build_system_prompt


def build_synthetic_mc_turn(qid: str, teacher_think: str) -> dict:
    """Construct the synthetic asst turn that carries teacher's reasoning and the mc call.

    Format matches what `src/runner.py` and the saved-trajectory convention
    expect: `content` carries the explicit `<think>...</think>` block; `tool_calls`
    is OpenAI/litellm-style with `function.arguments` as a JSON-encoded string.
    """
    return {
        "role": "assistant",
        "content": f"<think>\n{teacher_think}\n</think>",
        "tool_calls": [
            {
                "id": f"call_teacher_mc_{qid}",
                "type": "function",
                "function": {
                    "name": "manage_context",
                    "arguments": "{}",
                },
            }
        ],
    }


def splice_annotation_to_initial_history(
    traj: dict,
    annotation: dict,
    *,
    benchmark: str,
    context_window: int,
) -> Optional[dict]:
    """Return a dict of fields to seed `runner.run` via `RunConfig` overrides.

    Returns None if the annotation can't be applied (decision != "mc",
    after_id out of range, etc.). For this iteration we only handle "mc" —
    qm is illegal in iter1 and no_action_needed has nothing to do.
    """
    if annotation.get("decision") != "mc":
        return None

    after_id = annotation.get("after_id")
    if not isinstance(after_id, int):
        return None

    history = traj.get("history") or []
    if not (0 <= after_id < len(history)):
        return None
    if history[after_id].get("role") not in ("system", "user", "tool"):
        return None

    qid = str(traj.get("query_id") or traj.get("_qid") or "")
    think = annotation.get("think") or ""

    # Rebuild history[0] as 4-tool system prompt so the resumed student has the
    # mc/qm strategy section. initial ran as 2-tool ReAct, so its system prompt
    # lacked _MEMORY_MANAGEMENT_STRATEGY.
    new_system = {
        "role": "system",
        "content": build_system_prompt(
            benchmark=benchmark,
            context_window=context_window,
            use_memory_tools=True,
        ),
    }

    spliced_prefix = [new_system] + list(history[1 : after_id + 1])
    synthetic_turn = build_synthetic_mc_turn(qid, think)
    initial_history = spliced_prefix + [synthetic_turn]

    return {
        "qid": qid,
        "question": traj.get("question", "") or "",
        "correct_answer": traj.get("correct_answer", "") or "",
        "initial_history": initial_history,
        "initial_raw_history": list(initial_history),  # d1: same as live view
        "initial_last_boundary_pos": 2,                # system + user, no prior mc
        "initial_summary_id": 0,                       # runner produces id=1 on turn 0 mc
        "teacher_annotation": {
            "after_id": after_id,
            "think": think,
            "decision": "mc",
        },
    }
