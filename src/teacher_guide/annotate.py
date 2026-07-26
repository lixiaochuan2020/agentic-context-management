"""Teacher annotation step for the teacher-guided pipeline.

Given a ReAct initial trajectory that the student failed on, call GPT-5 to emit
one structured annotation: where in the trajectory should `manage_context` or
`query_memory` have been invoked, and the first-person rationale.

See `experiments/20260517.md` § Exp2 for design.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.client import LiteLLMClient
from src.teacher_guide.prompts import (
    _TEACHER_ANNOTATE_INSTRUCTION,
    render_gold_docs_block,
    render_numbered_history,
)

logger = logging.getLogger(__name__)


TEACHER_FORBIDDEN_RE = re.compile(
    r"\b(coach|coaches|coaching|feedback|review(?:ed|er|ers|s)?|"
    r"advis(?:e|ed|er|or|ors)|guidance|guided|told me|"
    r"someone said|external|instructed)\b",
    re.IGNORECASE,
)

VALID_DECISIONS = {"mc", "qm", "no_action_needed"}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_teacher_json(raw: str) -> dict:
    """Extract the single JSON object from teacher output.

    Tries (in order): direct `json.loads`, fenced ```json…``` block, first-`{`
    to last-`}` slice. Raises `ValueError` if none yield a valid object.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty teacher response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    i, j = raw.find("{"), raw.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(raw[i : j + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("could not parse JSON object from teacher response")


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


def validate_annotation(
    parsed: dict,
    history: list[dict],
    summary_ids: list[int],
    correct_answer: str,
) -> ValidationResult:
    """Check schema + leakage constraints. Returns ok flag + list of error strings."""
    errors: list[str] = []

    decision = parsed.get("decision")
    if decision not in VALID_DECISIONS:
        errors.append(f"decision={decision!r} not in {sorted(VALID_DECISIONS)}")

    after_id = parsed.get("after_id")
    if not isinstance(after_id, int):
        errors.append(f"after_id must be int, got {type(after_id).__name__}")
    elif decision != "no_action_needed":
        n = len(history)
        if not (0 <= after_id < n):
            errors.append(f"after_id={after_id} not in [0, {n - 1}]")
        else:
            role = history[after_id].get("role")
            if role not in ("system", "user", "tool"):
                errors.append(
                    f"after_id={after_id} points to role={role!r}; must be system/user/tool"
                )

    think = parsed.get("think") or ""
    if decision != "no_action_needed":
        if not (200 <= len(think) <= 1200):
            errors.append(f"len(think)={len(think)} outside [200, 1200]")
        m = TEACHER_FORBIDDEN_RE.search(think)
        if m:
            errors.append(f"think contains forbidden word: {m.group(0)!r}")
        ans = (correct_answer or "").strip()
        if len(ans) >= 4 and ans.lower() in think.lower():
            errors.append("think contains golden-answer substring")

    if decision == "qm":
        if not summary_ids:
            errors.append("decision='qm' but workspace summary_ids is empty")
        if not (parsed.get("qm_query") or "").strip():
            errors.append("decision='qm' requires non-empty qm_query")

    return ValidationResult(ok=not errors, errors=errors)


def call_teacher_annotate(
    *,
    client: LiteLLMClient,
    question: str,
    correct_answer: str,
    gold_docs: list[dict] | None,
    history: list[dict],
    summary_ids: list[int],
    max_new_tokens: int = 4096,
) -> tuple[str, dict]:
    """One teacher call. Returns (raw_response, parsed_json).

    Caller is responsible for retry / validation. Raises on parse failure
    so the retry loop can detect it.
    """
    prompt = _TEACHER_ANNOTATE_INSTRUCTION.format(
        question=question,
        correct_answer=correct_answer,
        gold_docs_block=render_gold_docs_block(gold_docs),
        numbered_history=render_numbered_history(history),
        summary_ids=summary_ids or "[]",
    )
    raw = client.generate(
        [{"role": "user", "content": prompt}],
        tools=None,
        max_new_tokens=max_new_tokens,
    )
    parsed = parse_teacher_json(raw)
    return raw, parsed


def annotate_one(
    *,
    client: LiteLLMClient,
    qid: str,
    traj: dict,
    gold_docs: list[dict] | None,
    summary_ids: list[int] | None = None,
    max_retries: int = 2,
) -> dict:
    """Annotate one trajectory; retry on parse / validation failure.

    Always returns a serializable dict. On total failure `decision = None`
    and `validation.ok = False`; the caller decides whether to save the
    record (typically yes, under `invalid_<qid>.json`).
    """
    summary_ids = list(summary_ids or [])
    history = traj.get("history") or []
    correct_answer = (traj.get("correct_answer") or "")
    question = (traj.get("question") or "")

    log: list[dict] = []
    parsed: dict | None = None
    final_raw: str | None = None
    final_validation: ValidationResult | None = None

    for attempt in range(max_retries + 1):
        try:
            raw, p = call_teacher_annotate(
                client=client,
                question=question,
                correct_answer=correct_answer,
                gold_docs=gold_docs,
                history=history,
                summary_ids=summary_ids,
            )
        except Exception as e:
            log.append({"attempt": attempt, "stage": "call", "error": f"{type(e).__name__}: {e}"})
            continue

        try:
            v = validate_annotation(p, history, summary_ids, correct_answer)
        except Exception as e:
            log.append({"attempt": attempt, "stage": "validate", "error": f"{type(e).__name__}: {e}",
                        "raw": raw or ""})
            continue

        log.append({"attempt": attempt, "stage": "validate", "errors": v.errors,
                    "raw": raw or ""})
        final_raw, parsed, final_validation = raw, p, v
        if v.ok:
            break

    ok = bool(final_validation and final_validation.ok)
    out: dict = {
        "query_id": qid,
        "validation": {
            "ok": ok,
            "attempts": len(log),
            "errors": (final_validation.errors if final_validation else ["no successful call"]),
            "log": log,
        },
        "teacher_raw_last": final_raw,
    }
    if ok and parsed is not None:
        out["decision"] = parsed.get("decision")
        out["after_id"] = parsed.get("after_id")
        out["think"] = parsed.get("think") or ""
        if parsed.get("decision") == "qm":
            out["qm_query"] = parsed.get("qm_query") or ""
    else:
        out["decision"] = None

    return out


def save_annotation(out: dict, out_dir: Path) -> Path:
    """Write annotation to `run_<qid>.json` if valid, else `invalid_<qid>.json`."""
    qid = out["query_id"]
    ok = bool((out.get("validation") or {}).get("ok"))
    name = f"run_{qid}.json" if ok else f"invalid_{qid}.json"
    p = Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
