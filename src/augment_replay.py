"""src/augment_replay.py — Training data augmentation via API replay.

For each q*.json in normal_free_L_0_Se_{se}_Su_{su}/:
  1. Detect C1/C2/C3 trigger points from raw_history
  2. Filter by token threshold (C1/C3: ≥15K, C2: ≥20K)
  3. Replace raw_history[0] with a memory-tool-enabled prompt (adds mem tools)
  4. Force manage_context call via tool_choice
  5. Execute CM via HistoryManager to populate all output fields
  6. Save training sample in identical format to cm_*.json

C1: Repetitive Search Cluster — 3 similar queries in 10-turn window (sim > 0.65)
C2: Dead-End Run — 4+ consecutive tool results each under 400 chars
C3: URL Revisit — same URL opened a second time
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from src.history import HistoryManager
from src.client import LiteLLMClient
from src.runner import call_summarizer
from src.prompts import build_initial_prompt
from src.tools import get_all_tools

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 200_000
DEFAULT_MODEL = "bedrock/us.anthropic.claude-sonnet-4-6"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_BENCHMARK = "deepsearchqa"

# Regex to parse [Round N] prefix from message content
ROUND_IDX_RE = re.compile(r"^\[Round (\d+)\]")

# C1: similarity threshold and window size
C1_SIM_THRESHOLD = 0.65
C1_WINDOW_SIZE = 10
C1_MIN_CLUSTER = 3
C1_MIN_TOKENS = 15_000

# C2: consecutive short tool results
C2_WINDOW_SIZE = 4
C2_MAX_CONTENT_LEN = 400
C2_MIN_TOKENS = 20_000

# C3: URL revisit
C3_MIN_TOKENS = 15_000


# ── Similarity helper ──────────────────────────────────────────────────────────

def jaccard_sim(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


# ── HistoryManager reconstruction ─────────────────────────────────────────────

def build_history_from_raw(messages: list[dict]) -> HistoryManager:
    """Reconstruct HistoryManager from raw_history messages.

    Each message with text content has a '[Round N]' prefix (written by
    HistoryManager.add_message / add_message_dict).  Messages with
    content=None (assistant tool_calls-only) have no such prefix.

    Strategy: parse _idx from '[Round N]' where available; for
    content=None messages assign seq_idx (last_known + 1, ...).
    """
    hm = HistoryManager()
    hm.messages = []
    hm.raw_messages = []

    seq_idx = 0
    for msg in messages:
        entry = dict(msg)
        content = entry.get("content") or ""
        m = ROUND_IDX_RE.match(content) if isinstance(content, str) else None
        if m:
            idx = int(m.group(1))
            seq_idx = idx + 1
        else:
            idx = seq_idx
            seq_idx += 1
        entry["_idx"] = idx
        hm.messages.append(entry)
        hm.raw_messages.append(copy.deepcopy(entry))

    max_idx = max((m.get("_idx", -1) for m in hm.messages), default=-1)
    hm._next_index = max_idx + 1
    return hm


# ── Token lookup ───────────────────────────────────────────────────────────────

def get_tokens_at_turn(token_trajectory: list, assistant_turn: int) -> int:
    """Return token count at (or near) the given assistant turn index."""
    if not token_trajectory:
        return 0
    idx = min(assistant_turn, len(token_trajectory) - 1)
    return token_trajectory[idx]


# ── Trigger detection ──────────────────────────────────────────────────────────

def detect_triggers(
    raw_history: list[dict],
    token_trajectory: list,
) -> list[tuple[int, str]]:
    """Detect C1/C2/C3 trigger points in raw_history.

    Returns a deduplicated, sorted list of (msg_idx, criterion) tuples.
    msg_idx is the slice index: raw_history[:msg_idx] is context_before.

    token_trajectory[k] = total token count after the k-th assistant turn.
    """
    triggers: list[tuple[int, str]] = []

    # C1 state: list of (msg_idx_after, query) per search call
    search_events: list[tuple[int, str]] = []

    # C2 state: sliding window of (msg_idx, content_len) for tool results
    recent_tool_results: list[tuple[int, int]] = []

    # C3 state: url -> first-seen msg_idx
    seen_urls: dict[str, int] = {}

    assistant_turn = 0

    for i, msg in enumerate(raw_history):
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}

                if name == "search":
                    query = args.get("query", "")
                    if not query:
                        continue
                    # Record search event (trigger slice is AFTER this assistant msg)
                    trigger_idx = i + 1
                    search_events.append((trigger_idx, query))

                    # C1 check: last C1_WINDOW_SIZE search queries
                    window = search_events[-C1_WINDOW_SIZE:]
                    if len(window) >= C1_MIN_CLUSTER:
                        queries = [w[1] for w in window]
                        found = False
                        for a in range(len(queries)):
                            similar = [a]
                            for b in range(a + 1, len(queries)):
                                if jaccard_sim(queries[a], queries[b]) > C1_SIM_THRESHOLD:
                                    similar.append(b)
                            if len(similar) >= C1_MIN_CLUSTER:
                                found = True
                                break
                        if found:
                            toks = get_tokens_at_turn(token_trajectory, assistant_turn)
                            if toks >= C1_MIN_TOKENS:
                                triggers.append((trigger_idx, "C1"))

                elif name == "open":
                    url = args.get("url", "")
                    if not url:
                        continue
                    if url in seen_urls:
                        # C3: revisit
                        toks = get_tokens_at_turn(token_trajectory, assistant_turn)
                        if toks >= C3_MIN_TOKENS:
                            triggers.append((i + 1, "C3"))
                    else:
                        seen_urls[url] = i

            assistant_turn += 1

        elif role == "tool":
            # C2: sliding window of recent tool results
            recent_tool_results.append((i, len(content)))
            if len(recent_tool_results) > C2_WINDOW_SIZE:
                recent_tool_results.pop(0)
            if (
                len(recent_tool_results) == C2_WINDOW_SIZE
                and all(length < C2_MAX_CONTENT_LEN for _, length in recent_tool_results)
            ):
                toks = get_tokens_at_turn(token_trajectory, assistant_turn)
                if toks >= C2_MIN_TOKENS:
                    triggers.append((i + 1, "C2"))

    # Sort and deduplicate exact (idx, criterion) pairs
    seen: set[tuple[int, str]] = set()
    result: list[tuple[int, str]] = []
    for t in sorted(triggers, key=lambda x: x[0]):
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ── Context-before construction ────────────────────────────────────────────────

def build_context_before(
    raw_history: list[dict],
    trigger_msg_idx: int,
    question: str,
) -> list[dict]:
    """Slice raw_history[:trigger_msg_idx] and replace Round 0 with a memory-tool-enabled prompt.

    Extends the slice past any trailing assistant messages so the context
    always ends with a user/tool message (required by Bedrock Converse API).
    C1/C3 triggers fire right after an assistant message; extending by one
    includes the following tool result.
    """
    end = trigger_msg_idx
    while 0 < end < len(raw_history) and raw_history[end - 1].get("role") == "assistant":
        end += 1
    messages = copy.deepcopy(raw_history[:end])
    if not messages:
        return messages

    new_content = "[Round 0] " + build_initial_prompt(
        "browsecomp",
        question,
        context_window=DEFAULT_CONTEXT_WINDOW,
        use_memory_tools=True,
    )
    messages[0] = {"role": "user", "content": new_content}
    return messages


# ── Forced manage_context API call ─────────────────────────────────────────────

def call_forced_manage_context(
    client: LiteLLMClient,
    context_before: list[dict],
    tools: list[dict],
) -> dict | None:
    """Force the model to call manage_context via tool_choice.

    Returns {"name": "manage_context", "arguments": {...}} or None on failure.
    """
    tool_choice = {"type": "function", "function": {"name": "manage_context"}}
    try:
        client.generate(
            context_before,
            tools=tools,
            max_new_tokens=2048,
            tool_choice=tool_choice,
        )
        tool_call = client.parse_tool_call("")  # reads from cache
        if tool_call and tool_call["name"] == "manage_context":
            return tool_call
        logger.warning("  [augment] model did not produce manage_context call")
        return None
    except Exception as e:
        logger.error("  [augment] generate() failed: %s", e)
        return None


# ── Sample saving ──────────────────────────────────────────────────────────────

def save_augmented_sample(
    out_dir: str,
    question_id: str,
    criterion: str,
    trigger_turn: int,
    question: str,
    correct_answer: str,
    context_before: list[dict],
    tokens_before: int,
    tool_call: dict,
    success: bool,
    summary_text: str | None,
    feedback: str,
    context_after: list[dict],
    tokens_after: int,
    metadata: dict,
) -> str:
    """Save one augmented training sample. Filename: {criterion}_t{turn:04d}.json"""
    sample_dir = os.path.join(out_dir, question_id)
    os.makedirs(sample_dir, exist_ok=True)

    sample = {
        "question_id": question_id,
        "question": question,
        "correct_answer": correct_answer,
        "mode": "memtool",
        "turn": trigger_turn,
        "call_index": 0,
        "context_before": context_before,
        "tokens_before": tokens_before,
        "tool_call": {
            "name": tool_call["name"],
            "arguments": tool_call["arguments"],
        },
        "tool_result": {
            "success": success,
            "content": summary_text if success else None,
            "feedback_message": feedback,
        },
        "context_after": context_after,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_before - tokens_after,
        "metadata": {
            **metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    filename = f"{criterion}_t{trigger_turn:04d}.json"
    path = os.path.join(sample_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    logger.info("  [augment] saved %s", path)
    return path


# ── Per-question processing ────────────────────────────────────────────────────

def process_question(
    client: LiteLLMClient,
    q_data: dict,
    out_dir: str,
    benchmark: str,
    dry_run: bool = False,
) -> int:
    """Process one question. Returns number of new samples saved (or would-save in dry_run)."""
    question_id = q_data.get("question_id", "q?")
    question = q_data.get("question", "")
    correct_answer = q_data.get("correct_answer", "")
    raw_history = q_data.get("raw_history", [])
    token_trajectory = q_data.get("token_trajectory", [])

    if not raw_history:
        logger.warning("[augment] %s: empty raw_history, skip", question_id)
        return 0

    triggers = detect_triggers(raw_history, token_trajectory)
    if not triggers:
        logger.debug("[augment] %s: no triggers found", question_id)
        return 0

    logger.info("[augment] %s: %d triggers detected", question_id, len(triggers))

    tools = get_all_tools()
    saved = 0

    for trigger_msg_idx, criterion in triggers:
        filename = f"{criterion}_t{trigger_msg_idx:04d}.json"
        out_path = os.path.join(out_dir, question_id, filename)

        if os.path.exists(out_path):
            logger.debug("[augment] %s/%s: exists, skip", question_id, filename)
            continue

        if dry_run:
            logger.info("[augment] [DRY RUN] would process %s/%s", question_id, filename)
            saved += 1
            continue

        # Build context_before: slice + replace Round 0
        context_before = build_context_before(
            raw_history, trigger_msg_idx, question
        )
        if not context_before:
            logger.warning("[augment] %s/%s: empty context, skip", question_id, filename)
            continue

        tokens_before = client.count_tokens(context_before, tools=tools)

        # Force manage_context call
        tool_call = call_forced_manage_context(client, context_before, tools)
        if tool_call is None:
            logger.warning("[augment] %s/%s: no tool call, skip", question_id, filename)
            continue

        args = tool_call["arguments"]
        rng = args.get("range")
        offload_rngs = args.get("offload_ranges") or []

        if not rng or not isinstance(rng, list) or len(rng) != 2:
            logger.warning(
                "[augment] %s/%s: invalid manage_context args (range=%s), skip",
                question_id, filename, rng,
            )
            continue

        s, e = rng[0], rng[1]

        # Normalize offload ranges
        def _normalize_range(r):
            if isinstance(r, int):
                return (r, r)
            if hasattr(r, '__len__'):
                if len(r) == 1:
                    return (r[0], r[0])
                if len(r) >= 2:
                    return (r[0], r[1])
            return None
        valid_offload = [n for r in offload_rngs if (n := _normalize_range(r)) is not None]

        # Build HistoryManager, call summarizer, offload from real context
        hm = build_history_from_raw(context_before)

        def _parse_round_idx(msg):
            c = msg.get("content") or ""
            m_re = ROUND_IDX_RE.match(c)
            return int(m_re.group(1)) if m_re else -1

        def _in_offload(msg):
            idx = _parse_round_idx(msg)
            return any(ds <= idx <= de for ds, de in valid_offload)

        full_range = hm.get_rounds_in_range(s, e)
        summary_text, _parse_ok = call_summarizer(client, question, full_range)
        for ds, de in valid_offload:
            hm.apply_offload(ds, de)

        # Append summary to history
        hm.add_message("assistant", summary_text)
        success = True

        feedback = (
            f"[manage_context applied: Rounds {s}-{e} processed "
            f"(offloaded {len(offload_rngs)} sub-ranges, then summarized)]"
        )

        context_after = hm.get_messages_for_model()
        tokens_after = client.count_tokens(context_after, tools=tools)

        metadata = {
            "benchmark": benchmark,
            "model": client.model,
            "augmentation": "api_replay",
            "trigger_criterion": criterion,
            "source_mode": "normal",
        }

        save_augmented_sample(
            out_dir=out_dir,
            question_id=question_id,
            criterion=criterion,
            trigger_turn=trigger_msg_idx,
            question=question,
            correct_answer=correct_answer,
            context_before=context_before,
            tokens_before=tokens_before,
            tool_call=tool_call,
            success=success,
            summary_text=summary_text,
            feedback=feedback,
            context_after=context_after,
            tokens_after=tokens_after,
            metadata=metadata,
        )
        saved += 1
        logger.info(
            "[augment] %s/%s: tokens %d→%d (saved %d), cost=$%.4f (total=$%.4f)",
            question_id, filename,
            tokens_before, tokens_after, tokens_before - tokens_after,
            client._session_cost, client.total_cost,
        )

    return saved


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment training data via API replay (C1/C2/C3 content-aware triggers)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM model string for augmentation (summarizer)")
    parser.add_argument("--source_model", default=None,
        help="Model that produced the source normal-run data (derives source model dir). "
             "Defaults to --model if not set.")
    parser.add_argument("--out_dir", default=None,
        help="Output directory for augmented samples. "
             "Defaults to <source_model_dir>/training_data/augmented_api_replay")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--source_mode_dir", default=None,
        help="Source result directory name (default: normal_free_L_0)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N question files (0 = all)",
    )
    parser.add_argument(
        "--question_ids", nargs="+", default=None,
        help="Only process specific question IDs, e.g. --question_ids q2 q5",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Count triggers without making API calls",
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Resolve source directory name
    source_dir_name = args.source_mode_dir or "normal_free_L_0"

    # Find model directory under results/<benchmark>/
    results_root = Path(args.results_dir)
    benchmark_dir = results_root / args.benchmark
    if not benchmark_dir.exists():
        logger.error("Benchmark directory not found: %s", benchmark_dir)
        return

    # Derive source model dir from --source_model (falls back to --model)
    source_model_str = args.source_model or args.model
    model_dir_name = source_model_str.replace("bedrock/", "").replace("/", ".")
    model_dir = benchmark_dir / model_dir_name
    if not model_dir.exists():
        # Fall back: search for any matching directory
        candidates = sorted(benchmark_dir.iterdir())
        candidates = [d for d in candidates if d.is_dir()]
        if not candidates:
            logger.error("No model directories found under %s", benchmark_dir)
            return
        model_dir = candidates[0]
        logger.info("Model dir not found by name, using: %s", model_dir)

    source_dir = model_dir / source_dir_name
    if not source_dir.exists():
        logger.error("Source directory not found: %s", source_dir)
        return

    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "training_data" / "augmented_api_replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    q_files = sorted(source_dir.glob("q*.json"))
    if args.question_ids:
        ids = set(args.question_ids)
        q_files = [f for f in q_files if f.stem in ids]
    elif args.limit > 0:
        q_files = q_files[: args.limit]

    logger.info(
        "Source: %s  (%d files)", source_dir, len(q_files)
    )
    logger.info("Output: %s", out_dir)
    if args.dry_run:
        logger.info("DRY RUN mode — no API calls will be made")

    client = LiteLLMClient(model=args.model)

    total_saved = 0
    total_errors = 0

    for q_path in q_files:
        with open(q_path, encoding="utf-8") as f:
            q_data = json.load(f)

        q_id = q_data.get("question_id", q_path.stem)
        try:
            n = process_question(
                client=client,
                q_data=q_data,
                out_dir=str(out_dir),
                benchmark=args.benchmark,
                dry_run=args.dry_run,
            )
            total_saved += n
        except Exception as exc:
            logger.error("Error on %s: %s", q_id, exc, exc_info=True)
            total_errors += 1

        # Log cost per question (only non-dry_run)
        if not args.dry_run:
            cost = client.get_and_reset_cost()
            if cost > 0:
                logger.info(
                    "[augment] %s done — total=$%.4f", q_id, client.total_cost
                )

    logger.info(
        "Finished. saved=%d, errors=%d, total_cost=$%.4f",
        total_saved, total_errors, client.total_cost,
    )


if __name__ == "__main__":
    main()
