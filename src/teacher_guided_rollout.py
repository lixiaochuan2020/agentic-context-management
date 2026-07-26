"""Teacher-guided rollout pipeline for iterative improvement on wrong answers.

For each question where student_model answered incorrectly:
  1. Build a prompt: student trajectory + golden answer + student answer + instruction
  2. Call teacher_model (gpt-5.5) to generate a guidance message
  3. Append guidance as an assistant message to the student's history
  4. Continue student_model rollout with compress_initial_history=True so the
     full prior context remains compressible
  5. Save results to out_run_dir for eval and further iteration

Usage:
    python -m src.teacher_guided_rollout \\
        --run_dir  results/browsecomp-plus/Qwen3.5-9B/result_base_qwen3.5-9b/run_all \\
        --eval_dir results/browsecomp-plus/Qwen3.5-9B/eval_bcp/result_base_qwen3.5-9b/run_all \\
        --out_run_dir results/browsecomp-plus/Qwen3.5-9B/result_base_qwen3.5-9b_iter1/run_all \\
        --teacher_model gpt-5.5 \\
        --student_model openai/Qwen3.5-9B \\
        --index_path data/BrowseComp-Plus/indexes/bm25
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
import traceback
from pathlib import Path

from dataclasses import replace
from pathlib import Path as _Path

from src.client import LiteLLMClient
from src.configs import RunConfig, load_config
from src.runner import run
from src.tools import get_tools

logger = logging.getLogger(__name__)


_TEACHER_INSTRUCTION = """\
You are a research coach reviewing the agent trajectory below. The agent failed to find the correct answer.

# STUDENT TRAJECTORY
{trajectory}

# QUESTION
{question}

# GOLDEN ANSWER  (FOR YOUR REFERENCE ONLY — never reveal to the agent)
{correct_answer}

# GOLDEN EVIDENCE  (FOR YOUR REFERENCE ONLY — never cite docids, quote, or paraphrase from these in your output)
{gold_docs_block}

# STUDENT'S FINAL ANSWER
{student_answer}

# YOUR TASK
Guide the agent toward the right search direction WITHOUT giving it the answer or pointing directly at the answer-bearing evidence. The agent must DISCOVER the answer through your directional hints, not verify what you have already told it.

Cover the following, in this order:

1. Diagnose the agent's search pattern (no answer leakage; reference only the agent's own behavior):
   - Which query vocabulary recurred across multiple searches without yielding new evidence
   - Which categorical directions were missed — e.g., it stuck to one source type / language / time period / entity type when a different category would have been more productive
   - If the agent is stuck in a dead-end loop, say so explicitly and recommend it compress its working memory before pivoting
   - If the agent previously retrieved evidence relevant to the question but it has been compressed out of context, recommend it retrieve that content back from long-term memory before re-searching

2. Suggest next directions by ALTERNATE VOCABULARY or ANGLE, not by direct reference to evidence the agent has not yet seen:
   - "Try [vocabulary X] instead of [vocabulary Y]"
   - "Try a different source type (e.g., local news, official biographical page, press release, archive index, …)"
   - "Try a different time-period / geographic / role-based filter"
   - You MAY reference docids the agent has ALREADY retrieved (mentioning them is not new information).

# STRICT CONSTRAINTS — answer-leakage guards
- DO NOT cite any docid the agent has not retrieved in this trajectory. Naming a new docid is direct leakage (the agent can fetch it verbatim).
- DO NOT quote, paraphrase, or hint at the content of the golden evidence.
- DO NOT name the entity / fact / answer the question asks about, even obliquely. If the answer is a person's name, do not write any part of that name; if it is a place, do not name it; if it is a number / color / category, do not state it. Use indirect descriptions ("the role held by that individual", "the specific clothing color in the 2018 police description", etc.).
- The student should arrive at the answer by following your directional hints, not by transcribing them.

Address the agent in second person ("you"). The feedback will be re-internalized by the student model into first-person reflection. Do not name specific tools by their schema name — refer to "memory compression", "long-term memory lookup", "search", etc.

Write the feedback directly (no preamble, no meta-commentary).\
"""


def _render_gold_docs_block(gold_docs: list[dict] | None, max_chars_per_doc: int = 2000) -> str:
    """Render gold_docs as a flat block: docid + url + first ~512 tokens of text.

    Caps per-doc text at `max_chars_per_doc` (~512 tokens for English prose).
    """
    if not gold_docs:
        return "(no gold docs available for this question)"
    parts = []
    for d in gold_docs:
        docid = d.get("docid", "?")
        url = (d.get("url") or "").strip()
        text = (d.get("text") or "").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "\n... [truncated]"
        head = f"docid={docid}"
        if url:
            head += f"\nurl: {url}"
        parts.append(f"{head}\ntext:\n{text}")
    return "\n\n---\n\n".join(parts)


# ── Student-revise instruction (1st-person rewrite + next action only) ────────

_STUDENT_REVISE_INSTRUCTION = """\
[external interjection]
Pause your investigation. The conversation above is your most recent context — earlier turns may have been compressed into summary_id entries or trimmed for brevity. The following message is an out-of-band instruction; once you have followed it, resume normal operation as if no interruption occurred.

Your most recent next-step (now retracted, to be replaced by your next assistant turn):
{init_last_attempt}

External feedback on that retracted next-step:
{teacher_feedback}

Now produce the single assistant turn that replaces the retracted next-step. The output must follow this exact format:

1. Use the already-open `<think>` block for private planning, then close it with `</think>`. The content inside `<think>...</think>` is your private working thought — it does NOT appear in the persisted reflection and does not need to mimic any specific voice.

2. After `</think>`, emit a `<reflection>...</reflection>` block. This is the visible, persisted artifact — write it in first person, as if you arrived at these insights through your own reasoning:
   - Acknowledge what direction you tried and why it failed to pan out.
   - Propose a concrete new direction (specific query vocabulary, document type, or angle).
   - You may reference specific docids / queries / evidence already in your context, but always in first-person ("I").
   - STRICTLY FORBIDDEN inside `<reflection>`: the words "coach", "feedback", "review", "reviewed", "reviewer", "advise", "advised", "advisor", "guidance", "told me", "someone said", "external", "instructed". The reflection must read entirely as your own honest first-person reflection — any of these words leaks the off-trajectory origin.

3. After `</reflection>`: issue one or more tool calls (search / get_document / manage_context / query_memory) to take the next investigative action(s) in the standard tool-call format. The number of tool calls is your choice. Do NOT produce a final answer in this turn — your job is to act on the new direction, not conclude. Even if the feedback strongly hints at a likely answer, you must verify it with at least one tool call.

The output should look exactly like a natural assistant turn you would produce on your own.\
"""

_STUDENT_REVISE_INSTRUCTION_LAST_TURN_ONLY = """\
You will be given your previous action (which was incorrect) and feedback on why it was wrong. Your task is to internalize the feedback as your own self-reflection, then produce a new, corrected action.

Critical constraints for the reflection:
1. Write entirely in the first person ("I", "my", "myself"). The reflection must read as your own internal thought process.
2. Do NOT reference the feedback, the source of the feedback, or any external party. Forbidden phrases include but are not limited to: "the feedback says", "I was told", "I should follow the advice", "the teacher / reviewer / coach pointed out", "as suggested", "based on the hint", "according to the correction", or any equivalent in any language.
3. Do NOT acknowledge that someone else evaluated your action. The reflection must look as if you arrived at these insights on your own by re-examining your previous action.
4. Do NOT include meta-commentary about the rewriting process itself (e.g., "let me rephrase this", "in my own words").
5. The reflection should diagnose what went wrong in your previous action, explain the correct reasoning, and motivate the next action — all as a natural continuation of your own thinking.
6. After the reflection, output a new action that supersedes the previous incorrect one.

Inputs:
<previous_action>
{previous_action}
</previous_action>

<feedback>
{feedback}
</feedback>

Output format (follow exactly):
<reflection>
[Your first-person self-reflection here. No references to any external evaluator or feedback source.]
</reflection>
Then issue the tool calls you want to execute. You must produce at least one tool call.\
"""

# _STUDENT_REVISE_INSTRUCTION_SIMPLE = """\
# {teacher_feedback}

# Rewrite this feedback as if it were your own thinking. Replace all instances of "you" with "I". Replace all second-person references with first-person references.

# Follow this output format:
# ## Rewrite: Put the result here.\
# """



# Words that betray the off-trajectory teacher origin if they leak into guided.
_FORBIDDEN_RE = re.compile(
    r"\b(coach|coaches|coaching|feedback|review(ed|er|ers|s)?|"
    r"advis(e|ed|er|or|ors)|guidance|guided|guided me|told me|"
    r"someone said|external|instructed)\b",
    re.IGNORECASE,
)
# Sentence-splitter for the "delete offending sentence" fallback. Conservative:
# splits on `.`, `!`, `?` only when followed by whitespace and capital/newline.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z<\n])")


def _has_forbidden(text: str) -> bool:
    return bool(_FORBIDDEN_RE.search(text or ""))


def _strip_forbidden_sentences(text: str) -> str:
    """Drop any sentence that contains a forbidden word."""
    if not text:
        return text
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept = [s for s in sentences if not _FORBIDDEN_RE.search(s)]
    return " ".join(kept).strip()


_USER_REF_RE = re.compile(r"\buser\b", re.IGNORECASE)


def _strip_first_sentence_if_user_ref(text: str) -> str:
    """If the first sentence references 'user' (e.g. "The user is pointing
    out that ..."), drop that sentence. Leaks the off-trajectory origin
    — the rollout itself has no 'user' addressing the model in 2nd person.
    """
    if not text:
        return text
    sentences = _SENTENCE_SPLIT_RE.split(text, maxsplit=1)
    if len(sentences) < 2:
        return text  # only one sentence; leave alone
    first, rest = sentences[0], sentences[1]
    if _USER_REF_RE.search(first):
        return rest.strip()
    return text


# ── Student revise: turn teacher feedback into guided (1st-person + next action) ──

def _extract_init_think_part(content: str) -> str:
    """Return the pre-`</think>` portion of initial's last-turn content.

    Chat template auto-opens `<think>` for assistant turns, so the model's
    output is `[think content]</think>\\n\\n[visible text]`. For
    student_revise we want only the thinking process (the model's reasoning
    that ended the investigation prematurely) — NOT the Exact Answer /
    Confidence text that comes after `</think>`, which is just the wrong
    commitment.
    """
    text = (content or "").strip()
    idx = text.find("</think>")
    if idx >= 0:
        return text[:idx].rstrip()
    return text


def student_revise(
    student_client: LiteLLMClient,
    base_config: RunConfig,
    question: str,
    init_last_asst: dict,
    teacher_feedback: str,
    max_retries: int = 1,
) -> tuple[dict, bool] | None:
    """Generate guided: a single assistant message {role, content, tool_calls} that
    replaces initial's failure-point turn, plus a `reflection_valid` flag.

    Uses _STUDENT_REVISE_INSTRUCTION_LAST_TURN_ONLY — a self-contained prompt
    (question + initial thinking + teacher feedback packaged into a single user
    message; no history replay). The caller splices only the returned `guided`
    assistant message into the saved history.

    Returns (guided, reflection_valid) where `reflection_valid` is True iff the
    model emitted a properly bounded `<reflection>...</reflection>` block
    that the post-processing successfully extracted. Returns None if no
    usable guided could be produced.
    """
    init_think = _extract_init_think_part(init_last_asst.get("content") or "")
    revise_prompt = _STUDENT_REVISE_INSTRUCTION_LAST_TURN_ONLY.format(
        previous_action=init_think[:4000],   # cap to avoid blowing context
        feedback=teacher_feedback,
    )
    tools = get_tools(
        base_config.runtime.benchmark,
        use_memory_tools=base_config.runtime.use_memory_tools,
    )

    last_content, last_tool_calls = "", []
    for attempt in range(max_retries + 1):
        msgs = [{"role": "user", "content": revise_prompt}]
        try:
            content = student_client.generate(msgs, tools=tools, max_new_tokens=4096)
        except Exception as e:
            logger.error("  [revise] generation failed: %s", e)
            return None
        tool_calls = student_client.parse_all_tool_calls() or []
        last_content, last_tool_calls = content, tool_calls

        if not tool_calls:
            logger.warning("  [revise] attempt %d: no tool_call in guided (retry); content tail (last 2000 chars): %s",
                           attempt, repr(content[-2000:]))
            continue
        # Post-processing forbidden-word retry disabled while testing <reflection>
        # delimiter. Re-enable if reflection content still leaks off-trajectory wording.
        # if _has_forbidden(content):
        #     logger.warning("  [revise] attempt %d: forbidden word detected (retry)", attempt)
        #     continue
        logger.info("  [revise] attempt %d: guided OK (%d tool_calls, %d chars)",
                    attempt, len(tool_calls), len(content))
        break

    if not last_tool_calls:
        logger.error("  [revise] no tool_call after %d attempts — giving up", max_retries + 1)
        return None

    # Post-processing strips disabled while testing <reflection> delimiter. The
    # tag-based prompt should keep the model's "real" thinking inside <think> and
    # the persisted output inside <reflection>, so substring leakage filters
    # become unnecessary. Re-enable if real runs still show leakage.
    # if _has_forbidden(last_content):
    #     stripped = _strip_forbidden_sentences(last_content)
    #     logger.warning("  [revise] forbidden words survived retries — stripping %d chars",
    #                    len(last_content) - len(stripped))
    #     last_content = stripped
    #
    # # Drop a leading "The user is/was/wants/etc..." sentence if present:
    # # the model still leaks off-trajectory framing in the opening line.
    # cleaned = _strip_first_sentence_if_user_ref(last_content)
    # if cleaned != last_content:
    #     logger.info("  [revise] dropped leading 'user'-referencing sentence (%d chars)",
    #                 len(last_content) - len(cleaned))
    #     last_content = cleaned

    # Extract the inner content of <reflection>...</reflection>. The model is
    # prompted to wrap its persisted self-reflection in these tags; we save
    # only the inner prose so the SFT-time guided reads as natural first-person
    # text (without the model learning a fixed <reflection>...</reflection>
    # output template). The chat-template auto-`<think>` block before the
    # opening tag — which always contains meta-prep ("the user is asking me
    # to...", "the feedback says...") — is discarded with the same slice.
    refl_match = re.search(r"<reflection>(.*?)</reflection>", last_content, re.DOTALL)
    if refl_match:
        dropped_n = len(last_content) - len(refl_match.group(1))
        last_content = refl_match.group(1).strip()
        logger.info("  [revise] extracted <reflection> inner content (dropped %d chars of wrappers + auto-think)",
                    dropped_n)
    else:
        logger.warning("  [revise] no <reflection>...</reflection> in guided — keeping full content")

    # parse_all_tool_calls() returns flat shape {id, name, arguments(dict)}.
    # OpenAI/litellm requires the nested shape on assistant.tool_calls:
    # {id, type:"function", function:{name, arguments: json_str}}. Convert
    # so this guided message can be replayed via initial_history to the API.
    api_tool_calls = [
        {
            "id": t["id"],
            "type": "function",
            "function": {
                "name": t["name"],
                "arguments": json.dumps(t["arguments"]),
            },
        }
        for t in last_tool_calls
    ]

    return {
        "role": "assistant",
        "content": last_content,
        "tool_calls": api_tool_calls,
    }, bool(refl_match)


# ── guided boundary helper ──────────────────────────────────────

_SUMMARY_MARKER_RE = re.compile(r"\[summary_id:\s*\d+\]")
_GUIDED_BOUNDARY_DEFAULT = 2  # right after system + user in a fresh initial rollout

# Character budget for the history slice replayed to student_revise. Replaying
# the full initial history (often 80K+ tokens) put the vLLM agent under enough
# concurrent KV pressure to crash it; cropping to a fixed char budget keeps the
# replay cheap while still priming the qwen3_xml tool-call format via the
# remaining tail of assistant turns.
_HISTORY_PREFIX_CHAR_BUDGET = 64 * 1024


def _msg_char_size(m: dict) -> int:
    """Rough char size of a message: content + serialized tool_calls."""
    n = len(m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        n += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    return n


def _is_summary_tool(m: dict) -> bool:
    return (
        m.get("role") == "tool"
        and bool(_SUMMARY_MARKER_RE.match((m.get("content") or "").lstrip()))
    )


def _trim_history_prefix(
    history: list[dict],
    init_last_idx: int,
    char_budget: int = _HISTORY_PREFIX_CHAR_BUDGET,
) -> list[dict]:
    """Return a cropped prefix of `history[:init_last_idx]` for student_revise.

    Always preserves: index 0 (system), index 1 (user question), and every
    role="tool" message in [2, init_last_idx) whose content starts with a
    `[summary_id: N]` marker (compressed-memory tool responses).

    The remaining budget is filled greedily from init_last_idx-1 walking
    backward, adding whole messages until the next one would exceed budget.
    The tail's leftmost index is snapped forward to the first assistant
    message (so any leading orphan tool message — whose paired assistant has
    been cropped out — is dropped; vLLM rejects tool messages without a
    matching preceding assistant.tool_calls).

    Returned messages keep their original order. Caller is responsible for
    appending the user-side revise instruction after this prefix.
    """
    if init_last_idx < 2:
        return list(history[:init_last_idx])

    fixed: set[int] = {0, 1}
    for i in range(2, init_last_idx):
        if _is_summary_tool(history[i]):
            fixed.add(i)

    used = sum(_msg_char_size(history[i]) for i in fixed)
    remaining = max(0, char_budget - used)

    tail: set[int] = set()
    i = init_last_idx - 1
    while i >= 2:
        if i in fixed:
            i -= 1
            continue
        size = _msg_char_size(history[i])
        if size > remaining:
            break
        tail.add(i)
        remaining -= size
        i -= 1

    # Snap tail-start forward to the first assistant message: a leading tool
    # whose assistant was cropped is orphaned and would 400 on vLLM.
    if tail:
        # Walk tail's leftmost index forward (it's contiguous by construction).
        start = min(tail)
        while start < init_last_idx and history[start].get("role") != "assistant":
            tail.discard(start)
            start += 1

    keep_idx = sorted(fixed | tail)
    return [history[i] for i in keep_idx]


def _find_last_boundary_pos(history: list[dict]) -> int:
    """Position in `history` just after the most recent summary tool_response.

    A summary tool_response is identified by the `[summary_id: N]` marker that
    the runner prepends. Returns _GUIDED_BOUNDARY_DEFAULT if initial never compressed.

    Used to seed `RunConfig.initial_last_boundary_pos` when handing off the
    guided-replaced history to the runner — the runner will pick up guided's
    tool_calls (including any manage_context) on turn 0 and use this boundary
    to compute the correct compress range.
    """
    last = None
    for i, m in enumerate(history):
        if m.get("role") == "tool" and _SUMMARY_MARKER_RE.search(m.get("content") or ""):
            last = i
    return _GUIDED_BOUNDARY_DEFAULT if last is None else (last + 1)


# ── History serialization ──────────────────────────────────

def _serialize_history(history: list[dict]) -> str:
    """Render history as readable text for the teacher prompt."""
    lines = []
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Flatten structured content blocks
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text") or str(block))
                else:
                    parts.append(str(block))
            content = "\n".join(parts)
        lines.append(f"[{role}]\n{content}")
    return "\n\n---\n\n".join(lines)


# ── Teacher call ───────────────────────────────────────────

def call_teacher(
    teacher_client: LiteLLMClient,
    history: list[dict],
    question: str,
    correct_answer: str,
    student_answer: str,
    gold_docs: list[dict] | None = None,
) -> str:
    """Call teacher_model with the full prompt rendered from _TEACHER_INSTRUCTION.

    Retries once on empty (whitespace-only) content — reasoning-token-heavy
    teachers (e.g. gpt-5.5) occasionally consume the full budget on hidden
    reasoning and emit no visible text. Raises if still empty after retry.
    """
    prompt = _TEACHER_INSTRUCTION.format(
        trajectory=_serialize_history(history),
        question=question,
        correct_answer=correct_answer,
        student_answer=student_answer,
        gold_docs_block=_render_gold_docs_block(gold_docs),
    )
    for attempt in range(2):
        guidance = teacher_client.generate(
            [{"role": "user", "content": prompt}],
            tools=None, max_new_tokens=131072,
        )
        guidance = (guidance or "").strip()
        if guidance:
            return guidance
        logger.warning(
            "call_teacher attempt %d returned empty content (finish_reason=%s)",
            attempt, getattr(teacher_client, "last_finish_reason", None),
        )
    raise RuntimeError("teacher returned empty content after 2 attempts")


# ── Load wrong questions ───────────────────────────────────

def load_wrong_questions(run_dir: str, eval_dir: str) -> list[dict]:
    """Return trajectory dicts for questions that are completed but wrong."""
    wrong = []
    eval_path = Path(eval_dir)
    for eval_file in sorted(eval_path.glob("run_*_eval.json")):
        with open(eval_file) as f:
            eval_data = json.load(f)

        if not eval_data.get("is_completed"):
            continue
        jr = eval_data.get("judge_result") or {}
        if jr.get("correct"):
            continue

        qid = eval_data.get("query_id")
        if not qid:
            # Extract qid from filename: run_{qid}_eval.json
            stem = eval_file.stem  # "run_769_eval"
            qid = stem.removeprefix("run_").removesuffix("_eval")

        traj_file = Path(run_dir) / f"run_{qid}.json"
        if not traj_file.exists():
            logger.warning("Trajectory file not found for qid=%s, skipping", qid)
            continue

        with open(traj_file) as f:
            traj = json.load(f)

        if not traj.get("history"):
            logger.warning("Empty history for qid=%s, skipping", qid)
            continue

        traj["_qid"] = str(qid)
        traj["_eval"] = eval_data
        wrong.append(traj)

    logger.info("Found %d completed-but-wrong questions", len(wrong))
    return wrong


# ── Workspace preparation ──────────────────────────────────

def _prepare_workspace(src_workspace: str | None, dst_workspace: str) -> None:
    """Copy iter_N workspace (summary files) into iter_N+1 workspace so that
    query_memory can still reach previous summaries."""
    os.makedirs(dst_workspace, exist_ok=True)
    if src_workspace and os.path.isdir(src_workspace):
        for fname in os.listdir(src_workspace):
            if fname.startswith("summary_") and fname.endswith(".json"):
                src = os.path.join(src_workspace, fname)
                dst = os.path.join(dst_workspace, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)


# ── Per-question rollout ───────────────────────────────────

def run_one(
    traj: dict,
    guidance: str,
    out_run_dir: str,
    student_client: LiteLLMClient,
    base_config: RunConfig,
    src_workspace_root: str | None = None,
    skip_existing: bool = True,
) -> None:
    qid = traj["_qid"]
    out_path = os.path.join(out_run_dir, f"run_{qid}.json")
    cache_path = os.path.join(out_run_dir, f"run_{qid}.partial.json")

    if skip_existing and os.path.exists(out_path):
        logger.info("  [%s] skipping (exists)", qid)
        return

    # ── 1. Identify initial's failure point (last assistant turn). Keep its content
    #       verbatim — `Exact Answer:` and what follows show the student the
    #       wrong commitment it made, which is part of what the teacher's
    #       feedback is correcting.
    history = list(traj["history"])
    init_last_idx = None
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "assistant":
            init_last_idx = idx
            break
    if init_last_idx is None:
        logger.error("  [%s] no assistant turn in initial history — skipping", qid)
        return

    init_last_asst = dict(history[init_last_idx])

    # ── 2. Off-trajectory: have the student rewrite teacher_feedback as guided
    #       (1st-person <think> + exactly one tool_call; no coach wording).
    revise_result = student_revise(
        student_client=student_client,
        base_config=base_config,
        question=traj.get("question", ""),
        init_last_asst=init_last_asst,
        teacher_feedback=guidance,
    )
    if revise_result is None:
        logger.error("  [%s] student_revise failed to produce a valid guided — skipping", qid)
        return
    guided, reflection_valid = revise_result

    # ── 3. Workspace + carry-over summary files from initial (so query_memory
    #       can still reach previous summaries).
    dst_workspace = os.path.join(out_run_dir, "workspace", qid)
    os.makedirs(dst_workspace, exist_ok=True)
    src_ws = os.path.join(src_workspace_root, qid) if src_workspace_root else None
    _prepare_workspace(src_ws, dst_workspace)

    summary_id_counter = 0
    import glob as _glob
    for _p in _glob.glob(os.path.join(dst_workspace, "summary_*.json")):
        try:
            n = int(os.path.basename(_p).removeprefix("summary_").removesuffix(".json"))
            summary_id_counter = max(summary_id_counter, n)
        except ValueError:
            pass

    # ── 4. Build initial_history (live view) and initial_raw_history (full
    #       archive). new_history just replaces initial's failure point with guided;
    #       guided's tool_calls are NOT executed here — runner's main loop will
    #       resume by executing them on turn 0 (so mc/mem_operations/
    #       tool_call_counts bookkeeping all stays in one place).
    new_history = history[:init_last_idx] + [guided]
    raw_history_init = list(traj.get("raw_history") or history)
    raw_init_last_idx = None
    for idx in range(len(raw_history_init) - 1, -1, -1):
        if raw_history_init[idx].get("role") == "assistant":
            raw_init_last_idx = idx
            break
    raw_init_prefix = raw_history_init[:raw_init_last_idx] if raw_init_last_idx is not None else raw_history_init
    new_raw_history = raw_init_prefix + [guided]

    # Boundary for the FIRST mc the runner sees: if guided itself is mc, this
    # is the right range to compress. Otherwise it just records the
    # already-existing boundary from initial.
    initial_boundary = _find_last_boundary_pos(history)

    runtime = replace(
        base_config.runtime,
        use_memory_tools=True,
        workspace_root=dst_workspace,
        compress_initial_history=False,
    )
    config = replace(
        base_config,
        runtime=runtime,
        question_id=qid,
        correct_answer=traj.get("correct_answer", ""),
        cache_path=cache_path,
        initial_history=new_history,
        initial_raw_history=new_raw_history,
        initial_last_boundary_pos=initial_boundary,
        initial_summary_id=summary_id_counter,
    )

    t0 = time.time()
    try:
        result = run(student_client, traj.get("question", ""), config)
    except Exception as e:
        logger.error("  [%s] CRASHED: %s\n%s", qid, e, traceback.format_exc())
        result = {
            "query_id": qid, "question": traj.get("question", ""),
            "correct_answer": traj.get("correct_answer", ""),
            "elapsed_sec": round(time.time() - t0, 1),
            "status": "crashed", "error": f"{type(e).__name__}: {e}",
            "final_answer": "", "confidence": 0, "num_turns": 0,
            "token_trajectory": [], "tool_calls": [], "mem_operations": [],
            "result": [], "retrieved_docids": [], "history": [], "raw_history": [],
            "summary_parse_ok": [],
        }

    result["elapsed_sec"] = round(time.time() - t0, 1)
    # Persist the off-trajectory intervention for observability + analysis.
    # The teacher feedback never appears in `history` / `raw_history` (so iter2
    # can resume cleanly from this turn), but is recorded here verbatim. Set on
    # both success and crash paths.
    result["teacher_intervention"] = {
        "teacher_feedback": guidance,
        "init_failure_turn": init_last_asst,
        "guided_rewrite_turn": guided,
    }
    # Flags for downstream SFT filtering: did guided emit a parseable <reflection>
    # block, and did each manage_context summary parse cleanly? `summary_parse_ok`
    # is plumbed up from runner.run() (per-mc parse-ok flag from `_extract_memory`).
    result["reflection_valid"] = reflection_valid
    # result["summary_parse_ok"] is already populated by runner.run() — no-op here.
    os.makedirs(out_run_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if os.path.exists(cache_path):
        os.remove(cache_path)

    cost = student_client.get_and_reset_cost() if hasattr(student_client, "get_and_reset_cost") else None
    logger.info(
        "  [%s] done in %.1fs, answer='%s', turns=%d%s",
        qid, result["elapsed_sec"],
        result.get("final_answer", "")[:60],
        result.get("num_turns", 0),
        f", cost=${cost:.4f}" if cost else "",
    )


# ── Main iteration loop ────────────────────────────────────

def _load_gold_docs_map(jsonl_path: str | None) -> dict[str, list[dict]]:
    """Load query_id → list of gold_docs from BrowseComp-Plus decrypted jsonl.

    Each gold_doc is `{docid, text, url}`. Empty dict on missing file.
    """
    if not jsonl_path or not os.path.isfile(jsonl_path):
        if jsonl_path:
            logger.warning("gold_docs jsonl not found: %s", jsonl_path)
        return {}
    out: dict[str, list[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(r.get("query_id", ""))
            if qid:
                out[qid] = r.get("gold_docs", []) or []
    logger.info("loaded gold_docs for %d queries from %s", len(out), jsonl_path)
    return out


def run_iteration(
    run_dir: str,
    eval_dir: str,
    out_run_dir: str,
    teacher_client: LiteLLMClient,
    student_client: LiteLLMClient,
    base_config: RunConfig,
    src_workspace_root: str | None = None,
    gold_docs_path: str | None = None,
    skip_existing: bool = True,
    limit: int = 0,
    qids: list[str] | None = None,
) -> None:
    os.makedirs(out_run_dir, exist_ok=True)

    wrong_trajs = load_wrong_questions(run_dir, eval_dir)
    if not wrong_trajs:
        logger.info("No wrong questions found — nothing to do.")
        return

    if qids:
        qids_set = {str(q) for q in qids}
        before = len(wrong_trajs)
        wrong_trajs = [t for t in wrong_trajs if t["_qid"] in qids_set]
        logger.info("Filtering to %d/%d questions matching --qids=%s",
                    len(wrong_trajs), before, sorted(qids_set))
        if not wrong_trajs:
            logger.info("No --qids matched a wrong question — nothing to do.")
            return

    if limit and limit > 0:
        wrong_trajs = wrong_trajs[:limit]
        logger.info("Limiting to %d questions (--limit)", limit)

    gold_docs_map = _load_gold_docs_map(gold_docs_path)

    for i, traj in enumerate(wrong_trajs):
        qid = traj["_qid"]
        logger.info("[%d/%d] %s — calling teacher ...", i + 1, len(wrong_trajs), qid)

        try:
            guidance = call_teacher(
                teacher_client,
                history=traj["history"],
                question=traj.get("question", ""),
                correct_answer=traj.get("correct_answer", ""),
                student_answer=traj.get("final_answer", ""),
                gold_docs=gold_docs_map.get(str(qid)),
            )
        except Exception as e:
            logger.error("  [%s] teacher call failed: %s", qid, e)
            continue

        logger.info("  [%s] guidance generated (%d chars)", qid, len(guidance))

        run_one(
            traj=traj,
            guidance=guidance,
            out_run_dir=out_run_dir,
            student_client=student_client,
            base_config=base_config,
            src_workspace_root=src_workspace_root,
            skip_existing=skip_existing,
        )


# ── CLI ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-guided rollout: inject teacher guidance and continue student rollout on wrong questions"
    )
    parser.add_argument("--run_dir", required=True,
        help="Directory with iter_N student trajectories (run_{qid}.json)")
    parser.add_argument("--eval_dir", required=True,
        help="Directory with iter_N eval results (run_{qid}_eval.json)")
    parser.add_argument("--out_run_dir", required=True,
        help="Output directory for iter_N+1 trajectories")
    parser.add_argument("--teacher_model", default="gpt-5.5",
        help="Teacher model (default: gpt-5.5). Called via api.openai.com.")
    parser.add_argument("--src_workspace_root", default=None,
        help="Workspace root from the iter_N run (to copy summary files for query_memory)")
    parser.add_argument("--gold_docs_path",
        default="data/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
        help="BCP decrypted JSONL with gold_docs per query_id. Pass empty string to disable.")
    parser.add_argument("--limit", type=int, default=0,
        help="Only process first N wrong questions (0 = all)")
    parser.add_argument("--qids", default="",
        help="Comma-separated query ids to target (overrides --limit ordering; useful for spot-tests)")
    parser.add_argument("--no_skip", action="store_true",
        help="Re-run even if output file already exists")
    parser.add_argument("--log_level", default="INFO")

    # Config plumbing (same pattern as src/run.py): YAML + dotlist + named flags.
    parser.add_argument("--config", default="configs/default.yaml",
        help="Project YAML; pass empty string to disable.")
    parser.add_argument("--override", nargs="*", default=[],
        metavar="KEY=VAL",
        help="OmegaConf dotlist overrides, e.g. retrieval.k=20")
    # Named CLI flags routed to RunConfig (all default None — only override when present).
    parser.add_argument("--student_model", default=None,
        help="agent.model (the student) — default from configs/default.yaml")
    parser.add_argument("--agent_api_base", default=os.environ.get("AGENT_API_BASE"),
        help="agent.api_base — default $AGENT_API_BASE")
    parser.add_argument("--index_path", default=None,
        help="retrieval.index_path")
    parser.add_argument("--summarizer_model", default=None,
        help="summarizer.model")
    parser.add_argument("--summarizer_api_base", default=os.environ.get("SUMMARIZER_API_BASE"),
        help="summarizer.api_base — default $SUMMARIZER_API_BASE")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Build base RunConfig ─────────────────────────────
    yaml_path = args.config or None
    if yaml_path and not _Path(yaml_path).is_file():
        logger.info("Config YAML %s not found — using dataclass defaults", yaml_path)
        yaml_path = None

    overrides: list[str] = []
    if args.student_model:        overrides.append(f"agent.model={args.student_model}")
    if args.agent_api_base:       overrides.append(f"agent.api_base={args.agent_api_base}")
    if args.index_path:           overrides.append(f"retrieval.index_path={args.index_path}")
    if args.summarizer_model:     overrides.append(f"summarizer.model={args.summarizer_model}")
    if args.summarizer_api_base:  overrides.append(f"summarizer.api_base={args.summarizer_api_base}")
    overrides.extend(args.override or [])

    base_config = load_config(yaml_path=yaml_path, cli_overrides=overrides)

    logger.info("Resolved RunConfig (teacher rollout):")
    logger.info("  student agent.model = %s", base_config.agent.model)
    logger.info("  agent.api_base      = %s", base_config.agent.api_base)
    logger.info("  summarizer.model    = %s", base_config.summarizer.model)
    logger.info("  retrieval.index_path= %s", base_config.retrieval.index_path)
    logger.info("  retrieval.backend   = %s  k=%d", base_config.retrieval.backend, base_config.retrieval.k)

    # Teacher client: always hits api.openai.com (independent of base_config).
    teacher_client = LiteLLMClient(model=args.teacher_model, api_base="https://api.openai.com/v1")

    # Student client uses agent.* from base_config.
    student_client = LiteLLMClient(model=base_config.agent.model,
                                   api_base=base_config.agent.api_base)

    qids_list = [q.strip() for q in args.qids.split(",") if q.strip()] if args.qids else None

    run_iteration(
        run_dir=args.run_dir,
        eval_dir=args.eval_dir,
        out_run_dir=args.out_run_dir,
        teacher_client=teacher_client,
        student_client=student_client,
        base_config=base_config,
        src_workspace_root=args.src_workspace_root,
        gold_docs_path=(args.gold_docs_path or None),
        skip_existing=not args.no_skip,
        limit=args.limit,
        qids=qids_list,
    )


if __name__ == "__main__":
    main()
