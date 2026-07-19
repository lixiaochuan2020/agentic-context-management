"""Convert teacher-guided rollout trajectories into SFT JSONL.

Two output formats are supported via --format:

  --format messages
    One JSON line per trajectory: {"messages": [...]}.
    Use with TRL SFTConfig(assistant_only_loss=True) — every assistant turn
    contributes loss. Suitable when you want full-trajectory training of A_0
    deep-research examples.

  --format prompt-completion  (policy1)
    One JSON line per ASSISTANT TURN TO TRAIN: {"prompt": str, "completion": str}.
    Use with TRL SFTConfig in prompt-completion mode (completion-only loss).
    Only the completion tokens contribute loss; the prompt is pure context.

policy1 specifics (--format prompt-completion):
  - iter1/iter2 only (--a0_dir omitted): A_0 supplement disabled
  - For each correct iter trajectory: emit ONE sample per assistant turn from
    A_1 through the final answer turn
  - For each wrong iter trajectory: emit exactly ONE sample (A_1 only); the
    wrong continuation is dropped
  - prompt = chat_template(traj[:target_idx], add_generation_prompt=True)
  - completion = chat_template(traj[:target_idx+1], add_generation_prompt=False)
                 minus prompt prefix (i.e. just the rendered target asst)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_COACH_PATTERN = re.compile(r"[^.!?]*coach[^.!?]*[.!?]\s*", re.IGNORECASE)


def _strip_coach_mentions(text: str) -> str:
    if not text or "coach" not in text.lower():
        return text
    return _COACH_PATTERN.sub("", text)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Parse tool_call args (str→dict) and strip 'coach' sentences."""
    out: list[dict] = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = _strip_coach_mentions(c)
        tcs = m.get("tool_calls")
        if tcs:
            new_tcs = []
            for tc in tcs:
                tc = dict(tc)
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args) if args else {}
                        except json.JSONDecodeError:
                            fn["arguments"] = {}
                    tc["function"] = fn
                new_tcs.append(tc)
            m["tool_calls"] = new_tcs
        out.append(m)
    return out


def _trim_trailing_non_assistant(messages: list[dict]) -> list[dict]:
    end = len(messages)
    while end > 0 and messages[end - 1].get("role") != "assistant":
        end -= 1
    return messages[:end]


def _load_eval_correct_map(eval_path: Path) -> dict[str, bool]:
    if not eval_path.is_file():
        logger.warning("eval file not found: %s", eval_path)
        return {}
    d = json.loads(eval_path.read_text(encoding="utf-8"))
    return {str(q["id"]): q["correct"] for q in d["per_question"]}


def process_a0_prompt_completion_reconstructed(
    dir_path: Path, eval_path: Path, tokenizer
) -> list[dict]:
    """p3 A_0 path: correct A_0 only, all asst turns, per-turn prompt reconstructed.

    A_0 is a fresh rollout (no initial_history pinned), so boundary_0 = 2
    (right after system + user). All asst turns are emitted as separate
    samples (no A_1 cutoff).
    """
    correct_map = _load_eval_correct_map(eval_path)
    samples = []
    skipped = {"no_grade": 0, "not_correct": 0, "broken": 0, "render_fail": 0}
    n_trajs = 0
    for fp in sorted(dir_path.glob("run_*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped["broken"] += 1
            continue
        qid = str(d.get("query_id"))
        is_correct = correct_map.get(qid)
        if is_correct is None:
            skipped["no_grade"] += 1
            continue
        if not is_correct:
            skipped["not_correct"] += 1
            continue
        raw = d.get("raw_history") or []
        if len(raw) < 3:
            skipped["broken"] += 1
            continue
        # A_0 fresh rollout: system+user are positions 0,1; boundary_0 = 2.
        boundary_0 = 2
        raw_norm = _normalize_messages(raw)
        asst_pos = [i for i, m in enumerate(raw_norm) if m.get("role") == "assistant"]
        if not asst_pos:
            skipped["broken"] += 1
            continue
        n_trajs += 1

        for ti in asst_pos:
            prefix = _reconstruct_prefix_at(raw_norm, ti, boundary_0)
            target_msg = raw_norm[ti]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prefix, tokenize=False, add_generation_prompt=True
                )
                full_text = tokenizer.apply_chat_template(
                    prefix + [target_msg], tokenize=False, add_generation_prompt=False
                )
            except Exception as e:
                logger.warning("chat_template failed at %s#asst@%d: %s", fp.name, ti, e)
                skipped["render_fail"] += 1
                continue
            if not full_text.startswith(prompt_text):
                skipped["render_fail"] += 1
                continue
            completion = full_text[len(prompt_text):]
            samples.append({
                "prompt": prompt_text,
                "completion": completion,
                "source": f"a0/{fp.name}#asst@{ti}",
                "category": "a0_correct_reconstructed",
                "qid": qid,
            })
    logger.info("A_0 reconstructed %s: emitted %d samples from %d correct trajectories; skipped %s",
                dir_path, len(samples), n_trajs, skipped)
    return samples


def _subsample_to_token_share(samples: list[dict], tokenizer, target_category: str,
                              target_share: float) -> list[dict]:
    """Drop whole trajectories from `target_category` (highest-token first) until
    that category's share of total tokens is <= `target_share`.

    Trajectory grouping uses the 'qid' field. Other-category samples untouched.
    """
    # Tokenize each sample to know its size.
    def n_toks(s):
        return len(tokenizer(s["prompt"] + s["completion"], add_special_tokens=False)["input_ids"])

    sizes = {id(s): n_toks(s) for s in samples}

    # Build per-trajectory totals for target category.
    target_qid_tokens = {}
    target_qid_samples = {}
    other_total = 0
    target_total = 0
    for s in samples:
        n = sizes[id(s)]
        if s.get("category") == target_category:
            qid = s.get("qid", s.get("source"))
            target_qid_tokens[qid] = target_qid_tokens.get(qid, 0) + n
            target_qid_samples.setdefault(qid, []).append(s)
            target_total += n
        else:
            other_total += n

    total = target_total + other_total
    if total == 0:
        return samples
    cur_share = target_total / total
    logger.info("%s share before drop: %.1f%% (%d / %d tokens, %d trajectories)",
                target_category, 100 * cur_share, target_total, total,
                len(target_qid_tokens))

    # Drop largest trajectories until share <= target.
    qids_by_size = sorted(target_qid_tokens.items(), key=lambda kv: -kv[1])
    dropped_qids = set()
    for qid, n in qids_by_size:
        if cur_share <= target_share:
            break
        target_total -= n
        total -= n
        dropped_qids.add(qid)
        cur_share = target_total / total if total else 0

    if dropped_qids:
        logger.info("dropped %d %s trajectories to hit %.0f%% target",
                    len(dropped_qids), target_category, 100 * target_share)

    kept = [s for s in samples
            if s.get("category") != target_category or s.get("qid") not in dropped_qids]
    new_target = sum(sizes[id(s)] for s in kept if s.get("category") == target_category)
    new_total = sum(sizes[id(s)] for s in kept)
    logger.info("%s share after drop: %.1f%% (%d / %d tokens, %d trajectories kept)",
                target_category, 100 * new_target / new_total if new_total else 0,
                new_target, new_total,
                len(target_qid_tokens) - len(dropped_qids))
    return kept


def _asst_indices(history: list[dict]) -> list[int]:
    return [i for i, m in enumerate(history) if m.get("role") == "assistant"]


def _find_a1_index(history: list[dict], num_turns: int) -> int | None:
    """A_1 = last assistant message before the runner-generated `num_turns` turns."""
    ai = _asst_indices(history)
    if num_turns >= len(ai):
        return None
    return ai[-(num_turns + 1)]


# ─────────────────────────────────────────────────────────────────────────
# Format: messages (full trajectory, assistant_only_loss=True)
# ─────────────────────────────────────────────────────────────────────────

def process_a0_messages(dir_path: Path, eval_path: Path, turn_threshold: int) -> list[dict]:
    correct_map = _load_eval_correct_map(eval_path)
    samples = []
    skipped = {"not_correct": 0, "short": 0, "broken": 0}
    for fp in sorted(dir_path.glob("run_*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped["broken"] += 1
            continue
        qid = str(d.get("query_id"))
        if not correct_map.get(qid):
            skipped["not_correct"] += 1
            continue
        if d.get("num_turns", 0) < turn_threshold:
            skipped["short"] += 1
            continue
        history = d.get("history") or []
        msgs = _trim_trailing_non_assistant(_normalize_messages(history))
        if not msgs:
            skipped["broken"] += 1
            continue
        samples.append({
            "messages": msgs,
            "source": f"A_0/{fp.name}",
            "category": "a0_correct_deep",
        })
    logger.info("A_0 messages %s: kept %d  (skipped: %s)", dir_path, len(samples), skipped)
    return samples


def process_iter_messages(dir_path: Path, eval_path: Path, stage_label: str) -> list[dict]:
    correct_map = _load_eval_correct_map(eval_path)
    samples = []
    skipped = {"no_grade": 0, "a1_not_found": 0, "broken": 0}
    n_correct = n_wrong = 0
    for fp in sorted(dir_path.glob("run_*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped["broken"] += 1
            continue
        qid = str(d.get("query_id"))
        is_correct = correct_map.get(qid)
        if is_correct is None:
            skipped["no_grade"] += 1
            continue
        history = d.get("history") or []
        num_turns = d.get("num_turns", 0)
        if is_correct:
            msgs = _trim_trailing_non_assistant(_normalize_messages(history))
            category = f"{stage_label}_correct"
            n_correct += 1
        else:
            a1_idx = _find_a1_index(history, num_turns)
            if a1_idx is None:
                skipped["a1_not_found"] += 1
                continue
            truncated = history[: a1_idx + 1]
            msgs = _trim_trailing_non_assistant(_normalize_messages(truncated))
            category = f"{stage_label}_wrong_truncated_at_a1"
            n_wrong += 1
        if not msgs:
            skipped["broken"] += 1
            continue
        samples.append({"messages": msgs, "source": f"{stage_label}/{fp.name}", "category": category})
    logger.info("%s messages %s: kept %d (correct=%d, wrong=%d; skipped: %s)",
                stage_label, dir_path, len(samples), n_correct, n_wrong, skipped)
    return samples


# ─────────────────────────────────────────────────────────────────────────
# Reconstruction helpers (p2_pc): build per-turn prompt from raw_history
# by replaying each manage_context compress as it fired during rollout.
# ─────────────────────────────────────────────────────────────────────────

def _is_manage_context_call(msg: dict) -> bool:
    if msg.get("role") != "assistant":
        return False
    for tc in msg.get("tool_calls") or []:
        fn = (tc.get("function") or {}).get("name") or tc.get("name")
        if fn == "manage_context":
            return True
    return False


def _reconstruct_prefix_at(raw: list[dict], target_raw_idx: int, boundary_0: int) -> list[dict]:
    """Reconstruct hm.messages as it stood just before raw[target_raw_idx] was generated.

    For target inside initial_history (target_raw_idx <= boundary_0), no
    compression could have fired yet — return list(raw[:target_raw_idx]).
    Otherwise simulate runner's append-then-compress loop from boundary_0
    onward, applying each manage_context call we encounter.
    """
    if target_raw_idx <= boundary_0:
        return list(raw[:target_raw_idx])
    working = list(raw[:boundary_0])
    boundary = boundary_0
    raw_pos = boundary_0
    while raw_pos < target_raw_idx:
        msg = raw[raw_pos]
        if _is_manage_context_call(msg):
            call_pos = len(working)
            working.append(msg)
            if boundary < call_pos:
                working = working[:boundary] + working[call_pos:]
            if raw_pos + 1 < len(raw):
                working.append(raw[raw_pos + 1])  # summary tool_response
                boundary = len(working)
            raw_pos += 2
        else:
            working.append(msg)
            raw_pos += 1
    return working


def _find_a1_in_raw(raw: list[dict], num_turns: int) -> tuple[int, int] | None:
    """Return (i_a1, boundary_0) using raw_history + runner-generated num_turns.

    Returns None if num_turns is inconsistent with raw_history asst counts.
    """
    asst_pos = [i for i, m in enumerate(raw) if m.get("role") == "assistant"]
    if num_turns >= len(asst_pos):
        return None
    initial_asst_count = len(asst_pos) - num_turns
    i_a1 = asst_pos[initial_asst_count - 1]
    if initial_asst_count < len(asst_pos):
        boundary_0 = asst_pos[initial_asst_count]  # position of A_2 (or first runner-gen asst)
    else:
        boundary_0 = len(raw)
    return i_a1, boundary_0


def process_iter_prompt_completion_reconstructed(
    dir_path: Path, eval_path: Path, stage_label: str, tokenizer
) -> list[dict]:
    """p2_pc: correct iter only, per-turn prompt reconstructed from raw_history."""
    correct_map = _load_eval_correct_map(eval_path)
    samples = []
    skipped = {"no_grade": 0, "not_correct": 0, "a1_not_found": 0, "broken": 0, "render_fail": 0}
    n_correct_trajs = 0
    for fp in sorted(dir_path.glob("run_*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped["broken"] += 1
            continue
        qid = str(d.get("query_id"))
        is_correct = correct_map.get(qid)
        if is_correct is None:
            skipped["no_grade"] += 1
            continue
        if not is_correct:
            skipped["not_correct"] += 1
            continue
        raw = d.get("raw_history") or []
        num_turns = d.get("num_turns", 0)
        found = _find_a1_in_raw(raw, num_turns)
        if found is None:
            skipped["a1_not_found"] += 1
            continue
        i_a1, boundary_0 = found

        raw_norm = _normalize_messages(raw)

        # Targets: every asst turn from A_1 through end of raw.
        asst_pos = [i for i, m in enumerate(raw_norm) if m.get("role") == "assistant"]
        target_indices = [i for i in asst_pos if i >= i_a1]
        n_correct_trajs += 1

        for ti in target_indices:
            prefix = _reconstruct_prefix_at(raw_norm, ti, boundary_0)
            target_msg = raw_norm[ti]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prefix, tokenize=False, add_generation_prompt=True
                )
                full_text = tokenizer.apply_chat_template(
                    prefix + [target_msg], tokenize=False, add_generation_prompt=False
                )
            except Exception as e:
                logger.warning("chat_template failed at %s#asst@%d: %s", fp.name, ti, e)
                skipped["render_fail"] += 1
                continue
            if not full_text.startswith(prompt_text):
                skipped["render_fail"] += 1
                continue
            completion = full_text[len(prompt_text):]
            samples.append({
                "prompt": prompt_text,
                "completion": completion,
                "source": f"{stage_label}/{fp.name}#asst@{ti}",
                "category": f"{stage_label}_correct_reconstructed",
            })
    logger.info("%s reconstructed %s: emitted %d samples from %d correct trajectories; skipped %s",
                stage_label, dir_path, len(samples), n_correct_trajs, skipped)
    return samples


# ─────────────────────────────────────────────────────────────────────────
# Format: prompt-completion (policy1)
# ─────────────────────────────────────────────────────────────────────────

def _emit_prompt_completion(tokenizer, full_msgs: list[dict], target_idx: int) -> tuple[str, str] | None:
    """Render (prompt, completion) for training the asst at `target_idx`.

    prompt    = chat_template(full_msgs[:target_idx],     add_generation_prompt=True)
    completion= chat_template(full_msgs[:target_idx+1],   add_generation_prompt=False)
                minus the prompt as a prefix.

    Returns None on inconsistency (prompt not a prefix of full render).
    """
    prompt_text = tokenizer.apply_chat_template(
        full_msgs[:target_idx], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs[: target_idx + 1], tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prompt_text):
        logger.warning("chat_template prompt is not a prefix of full render — skipping target_idx=%d", target_idx)
        return None
    completion = full_text[len(prompt_text):]
    return prompt_text, completion


def process_iter_prompt_completion(dir_path: Path, eval_path: Path, stage_label: str, tokenizer) -> list[dict]:
    correct_map = _load_eval_correct_map(eval_path)
    samples = []
    skipped = {"no_grade": 0, "a1_not_found": 0, "broken": 0, "render_fail": 0}
    n_correct_trajs = n_wrong_trajs = 0
    for fp in sorted(dir_path.glob("run_*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped["broken"] += 1
            continue
        qid = str(d.get("query_id"))
        is_correct = correct_map.get(qid)
        if is_correct is None:
            skipped["no_grade"] += 1
            continue
        history = d.get("history") or []
        num_turns = d.get("num_turns", 0)
        a1_idx = _find_a1_index(history, num_turns)
        if a1_idx is None:
            skipped["a1_not_found"] += 1
            continue

        norm_hist = _normalize_messages(history)

        # Build list of TARGET asst indices to emit.
        asst_idx_all = _asst_indices(norm_hist)
        if is_correct:
            # All asst turns from A_1 through end.
            target_indices = [i for i in asst_idx_all if i >= a1_idx]
            n_correct_trajs += 1
            cat = f"{stage_label}_correct"
        else:
            # Only A_1.
            target_indices = [a1_idx]
            n_wrong_trajs += 1
            cat = f"{stage_label}_wrong_a1_only"

        for ti in target_indices:
            pc = _emit_prompt_completion(tokenizer, norm_hist, ti)
            if pc is None:
                skipped["render_fail"] += 1
                continue
            prompt_text, completion = pc
            samples.append({
                "prompt": prompt_text,
                "completion": completion,
                "source": f"{stage_label}/{fp.name}#asst@{ti}",
                "category": cat,
            })
    logger.info("%s prompt-completion %s: emitted %d samples from %d correct-trajs + %d wrong-trajs; skipped %s",
                stage_label, dir_path, len(samples), n_correct_trajs, n_wrong_trajs, skipped)
    return samples


# ─────────────────────────────────────────────────────────────────────────
# Token-count filter
# ─────────────────────────────────────────────────────────────────────────

def _filter_by_token_length(samples: list[dict], tokenizer, max_tokens: int) -> list[dict]:
    kept, dropped = [], []
    for s in samples:
        try:
            if "messages" in s:
                txt = tokenizer.apply_chat_template(s["messages"], tokenize=False)
                n = len(tokenizer(txt, add_special_tokens=False)["input_ids"])
            else:
                n = len(tokenizer(s["prompt"] + s["completion"], add_special_tokens=False)["input_ids"])
        except Exception as e:
            logger.warning("tokenize failed for %s: %s — dropping", s.get("source"), e)
            dropped.append((s.get("source"), -1))
            continue
        if n > max_tokens:
            dropped.append((s.get("source"), n))
            continue
        s["n_tokens"] = n
        kept.append(s)
    logger.info("token filter @ %d: kept %d / dropped %d", max_tokens, len(kept), len(dropped))
    if dropped:
        sizes = [n for _, n in dropped if n > 0]
        if sizes:
            logger.info("  dropped sample sizes: min=%d median=%d max=%d",
                        min(sizes), sorted(sizes)[len(sizes)//2], max(sizes))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=["messages", "prompt-completion"], required=True,
                    help="Output format. policy1 uses 'prompt-completion'.")
    ap.add_argument("--a0_dir", type=Path, default=None,
                    help="A_0 run_all dir (optional, supplementary). Disabled for policy1.")
    ap.add_argument("--a0_eval", type=Path, default=None,
                    help="A_0's gpt5_eval.json. Defaults to <a0_dir>/gpt5_eval.json.")
    ap.add_argument("--turn_threshold", type=int, default=16,
                    help="A_0 correct turns>=this. Default 16. Only used with --a0_dir.")
    ap.add_argument("--iter_dirs", nargs="+", type=Path, default=[],
                    help="One or more iter run_all directories.")
    ap.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="Drop samples whose token count exceeds this. 0 = no filter.")
    ap.add_argument("--tokenizer", type=str, required=True,
                    help="HF tokenizer path (needed for chat-template rendering / filter).")
    ap.add_argument("--reconstruct_context", action="store_true",
                    help="p2_pc: correct iter only; per-turn prompt rebuilt from raw_history "
                         "(replays compress ops). Requires --format prompt-completion.")
    ap.add_argument("--a0_target_share", type=float, default=0.0,
                    help="p3: target A_0 token share in final mix (e.g. 0.2 = 20%). "
                         "Drop whole A_0 trajectories (largest-first) until share <= target. "
                         "Requires --a0_dir + --reconstruct_context.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    samples: list[dict] = []

    if args.a0_dir is not None and args.format == "messages":
        a0_dir = args.a0_dir.expanduser().resolve()
        a0_eval = (args.a0_eval or a0_dir / "gpt5_eval.json").expanduser().resolve()
        samples.extend(process_a0_messages(a0_dir, a0_eval, args.turn_threshold))
    elif args.a0_dir is not None and args.format == "prompt-completion":
        if not args.reconstruct_context:
            logger.warning("--a0_dir + prompt-completion ignored (needs --reconstruct_context).")
        else:
            a0_dir = args.a0_dir.expanduser().resolve()
            a0_eval = (args.a0_eval or a0_dir / "gpt5_eval.json").expanduser().resolve()
            samples.extend(process_a0_prompt_completion_reconstructed(a0_dir, a0_eval, tokenizer))

    if args.reconstruct_context and args.format != "prompt-completion":
        ap.error("--reconstruct_context requires --format prompt-completion")

    for d in args.iter_dirs:
        d = d.expanduser().resolve()
        eval_path = d / "gpt5_eval.json"
        # Stage label = trailing '-iter1' / '-iter2' segment of parent dir
        # (e.g. 'rollout-v3_indirect_iter12_qwen3emb-iter1' -> 'iter1').
        parent = d.parent.name
        stage_label = parent.split("-")[-1] if "-iter" in parent else parent
        if args.format == "messages":
            samples.extend(process_iter_messages(d, eval_path, stage_label))
        elif args.reconstruct_context:
            samples.extend(process_iter_prompt_completion_reconstructed(d, eval_path, stage_label, tokenizer))
        else:
            samples.extend(process_iter_prompt_completion(d, eval_path, stage_label, tokenizer))

    logger.info("total collected (pre-filter): %d samples", len(samples))

    if args.max_tokens > 0:
        samples = _filter_by_token_length(samples, tokenizer, args.max_tokens)

    if args.a0_target_share > 0:
        samples = _subsample_to_token_share(
            samples, tokenizer, "a0_correct_reconstructed", args.a0_target_share)

    logger.info("final samples: %d", len(samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("wrote → %s", args.output)

    from collections import Counter
    cats = Counter(s.get("category", "?") for s in samples)
    logger.info("category breakdown:")
    for cat, cnt in sorted(cats.items()):
        logger.info("  %s: %d", cat, cnt)


if __name__ == "__main__":
    main()
