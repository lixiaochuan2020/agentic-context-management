"""Build SFT JSONL for the teacher-guided data.

Two data sources are combined:

- main body (~80%):  per-turn samples from EACH correct post-teacher-annotation
  rollout in `data/teacher_guide_pass@4_correct.json`. For each unique qid,
  pick the first correct attempt (iter1 > retry1 > retry2 > retry3). Emit
  (prompt, completion) for every asst turn at raw_history position >= after_id+1
  — the teacher's mc step onwards. The original failed initial prefix
  (positions 2..after_id) is NOT trained on.

- 20% mix:           per-turn samples from CORRECT initial trajectories (those that
  answered correctly with 2-tool ReAct and did NOT need mc). These supply
  search / get_document / answer supervision and stabilize training. Subsampled
  to ~20% token share via the existing token-share subsample convention.

BOTH paths share:
  - history[0] rebuilt as 4-tool system prompt (matches inference).
  - tools=get_tools(use_memory_tools=True) passed to chat_template (matches
    inference: LiteLLM injects JSON schemas).
  - per-turn prefix reconstructed by replaying `manage_context` compressions in
    raw_history (re-uses `_reconstruct_prefix_at`).

Output: data/sft/teacher_guide.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "train"))

from src.prompts import build_system_prompt
from src.tools import get_tools

# Reuse compression-replay logic from the existing preprocess.
# `train/` is not a package — import as a top-level module.
from preprocess_teacher_iter import (
    _reconstruct_prefix_at,
    _normalize_messages,
    _subsample_to_token_share,
)

logger = logging.getLogger("preprocess_teacher_guide")


ATTEMPT_PRIORITY = ["iter1", "retry1", "retry2", "retry3"]


def pick_one_per_qid(trajectories: list[dict]) -> dict[str, dict]:
    """Group pass@4 entries by qid, picking the first attempt per qid by priority."""
    by_qid: dict[str, dict] = {}
    for attempt in ATTEMPT_PRIORITY:
        for entry in trajectories:
            if entry.get("attempt") != attempt:
                continue
            qid = str(entry["qid"])
            if qid not in by_qid:
                by_qid[qid] = entry
    return by_qid


def emit_per_turn_samples(
    *,
    raw: list[dict],
    asst_positions: list[int],
    boundary_0: int,
    sys_prompt_4tool: str,
    tools_4tool: list[dict],
    tokenizer,
    source_label: str,
    qid: str,
    category: str,
) -> tuple[list[dict], Counter]:
    """For each target asst turn at position ti, render (prompt, completion).

    Returns (samples, skip_counter).
    """
    samples: list[dict] = []
    skipped: Counter = Counter()
    raw_norm = _normalize_messages(raw)

    for ti in asst_positions:
        prefix = _reconstruct_prefix_at(raw_norm, ti, boundary_0)
        if not prefix:
            skipped["empty_prefix"] += 1
            continue
        # Rebuild history[0] as 4-tool system prompt (overrides whatever was there).
        prefix = [{"role": "system", "content": sys_prompt_4tool}] + list(prefix[1:])
        target_msg = raw_norm[ti]
        try:
            prompt_text = tokenizer.apply_chat_template(
                prefix, tools=tools_4tool, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                prefix + [target_msg], tools=tools_4tool,
                tokenize=False, add_generation_prompt=False,
            )
        except Exception as e:
            logger.warning("chat_template failed for %s qid=%s @%d: %s",
                           source_label, qid, ti, e)
            skipped["render_fail"] += 1
            continue
        if not full_text.startswith(prompt_text):
            skipped["prefix_mismatch"] += 1
            continue
        completion = full_text[len(prompt_text):]
        samples.append({
            "prompt": prompt_text,
            "completion": completion,
            "qid": qid,
            "source": f"{source_label}#asst@{ti}",
            "category": category,
        })
    return samples, skipped


def process_teacher_guide_trajectories(
    *,
    pass4_index: dict,
    sys_prompt_4tool: str,
    tools_4tool: list[dict],
    tokenizer,
) -> list[dict]:
    """Per-turn samples from correct post-teacher-annotation rollouts. One traj
    per qid (priority iter1 > retry1/2/3). Emit asst turns at raw_history
    position >= after_id+1.
    """
    picked = pick_one_per_qid(pass4_index["trajectories"])
    logger.info("teacher-guided: %d unique correct qids picked (priority %s)",
                len(picked), ATTEMPT_PRIORITY)

    samples: list[dict] = []
    aggregate_skipped: Counter = Counter()
    by_attempt = Counter()

    for qid, entry in picked.items():
        traj_fp = REPO / entry["trajectory_path"]
        ann_fp = REPO / entry["annotation_path"]
        if not traj_fp.is_file() or not ann_fp.is_file():
            aggregate_skipped["missing_files"] += 1
            continue
        try:
            traj = json.loads(traj_fp.read_text(encoding="utf-8"))
            ann = json.loads(ann_fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            aggregate_skipped["broken_json"] += 1
            continue

        after_id = ann.get("after_id")
        if not isinstance(after_id, int):
            aggregate_skipped["bad_after_id"] += 1
            continue

        raw = traj.get("raw_history") or traj.get("history") or []
        if len(raw) <= after_id:
            aggregate_skipped["short_raw"] += 1
            continue

        # asst positions starting at the teacher's mc turn (after_id+1) onwards.
        asst_positions = [
            i for i, m in enumerate(raw)
            if m.get("role") == "assistant" and i >= after_id + 1
        ]
        if not asst_positions:
            aggregate_skipped["no_target_turns"] += 1
            continue

        # boundary_0 = 2 (spliced trajs start with [system, user] from initial).
        new_samples, skips = emit_per_turn_samples(
            raw=raw,
            asst_positions=asst_positions,
            boundary_0=2,
            sys_prompt_4tool=sys_prompt_4tool,
            tools_4tool=tools_4tool,
            tokenizer=tokenizer,
            source_label=f"teacher_guide_{entry['attempt']}",
            qid=qid,
            category="teacher_guide_correct",
        )
        samples.extend(new_samples)
        aggregate_skipped.update(skips)
        by_attempt[entry["attempt"]] += 1

    logger.info("teacher-guided: emitted %d samples from %d trajs (by attempt: %s)  skipped=%s",
                len(samples), sum(by_attempt.values()), dict(by_attempt), dict(aggregate_skipped))
    return samples


def process_init_correct(
    *,
    init_dir: Path,
    init_eval_dir: Path,
    sys_prompt_4tool: str,
    tools_4tool: list[dict],
    tokenizer,
) -> list[dict]:
    """Per-turn samples from CORRECT initial trajectories. Rebuild history[0] as
    4-tool system prompt; emit every asst turn from position 2 onwards.
    """
    samples: list[dict] = []
    aggregate_skipped: Counter = Counter()
    n_correct_qids = 0

    for eval_fp in sorted(init_eval_dir.glob("run_*_eval.json")):
        try:
            ed = json.loads(eval_fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not (ed.get("judge_result") or {}).get("correct"):
            continue
        qid = str(ed.get("query_id") or eval_fp.stem.removeprefix("run_").removesuffix("_eval"))
        traj_fp = init_dir / f"run_{qid}.json"
        if not traj_fp.is_file():
            aggregate_skipped["traj_missing"] += 1
            continue
        try:
            traj = json.loads(traj_fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            aggregate_skipped["broken_json"] += 1
            continue

        raw = traj.get("raw_history") or traj.get("history") or []
        if len(raw) < 3:
            aggregate_skipped["short_raw"] += 1
            continue
        asst_positions = [i for i, m in enumerate(raw) if m.get("role") == "assistant"]
        if not asst_positions:
            aggregate_skipped["no_asst"] += 1
            continue
        n_correct_qids += 1

        new_samples, skips = emit_per_turn_samples(
            raw=raw,
            asst_positions=asst_positions,
            boundary_0=2,
            sys_prompt_4tool=sys_prompt_4tool,
            tools_4tool=tools_4tool,
            tokenizer=tokenizer,
            source_label="init_correct",
            qid=qid,
            category="init_correct_reconstructed",
        )
        samples.extend(new_samples)
        aggregate_skipped.update(skips)

    logger.info("initial correct: emitted %d samples from %d correct trajs  skipped=%s",
                len(samples), n_correct_qids, dict(aggregate_skipped))
    return samples


def filter_by_token_length(samples: list[dict], tokenizer, max_tokens: int) -> list[dict]:
    kept, dropped = [], []
    for s in samples:
        n = len(tokenizer(s["prompt"] + s["completion"], add_special_tokens=False)["input_ids"])
        if n > max_tokens:
            dropped.append(n)
            continue
        s["n_tokens"] = n
        kept.append(s)
    logger.info("token filter @ %d: kept %d / dropped %d", max_tokens, len(kept), len(dropped))
    if dropped:
        ds = sorted(dropped)
        logger.info("  dropped sizes: min=%d median=%d max=%d", ds[0], ds[len(ds)//2], ds[-1])
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass4_index", type=Path,
                    default=REPO / "data/teacher_guide_pass@4_correct.json")
    ap.add_argument("--init_dir", type=Path,
                    default=REPO / "results/browsecomp-plus/qwen3.5-9b-base/rollout-teacher_guide-init/run_all")
    ap.add_argument("--init_eval_dir", type=Path,
                    default=REPO / "results/browsecomp-plus/qwen3.5-9b-base/eval_bcp/rollout-teacher_guide-init/run_all")
    ap.add_argument("--output", type=Path,
                    default=REPO / "data/sft/teacher_guide.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--benchmark", default="browsecomp-plus")
    ap.add_argument("--context_window", type=int, default=131072)
    ap.add_argument("--max_tokens", type=int, default=131072,
                    help="Drop samples whose prompt+completion exceed this (0=no filter).")
    ap.add_argument("--init_target_share", type=float, default=0.20,
                    help="Token share of initial in final mix (default 0.20).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    sys_prompt_4tool = build_system_prompt(
        benchmark=args.benchmark, context_window=args.context_window, use_memory_tools=True,
    )
    tools_4tool = get_tools(benchmark=args.benchmark, use_memory_tools=True)
    logger.info("tools: %s", [t["function"]["name"] for t in tools_4tool])

    pass4 = json.loads(args.pass4_index.read_text(encoding="utf-8"))
    logger.info("pass@4 index: %d unique qids, %d trajectories",
                pass4["metadata"]["unique_correct_qids"],
                pass4["metadata"]["total_correct_trajectories"])

    teacher_guide_samples = process_teacher_guide_trajectories(
        pass4_index=pass4, sys_prompt_4tool=sys_prompt_4tool,
        tools_4tool=tools_4tool, tokenizer=tokenizer,
    )
    init_samples = process_init_correct(
        init_dir=args.init_dir, init_eval_dir=args.init_eval_dir,
        sys_prompt_4tool=sys_prompt_4tool, tools_4tool=tools_4tool, tokenizer=tokenizer,
    )

    all_samples = teacher_guide_samples + init_samples
    logger.info("total before filter/subsample: teacher_guide=%d  init=%d  combined=%d",
                len(teacher_guide_samples), len(init_samples), len(all_samples))

    if args.max_tokens > 0:
        all_samples = filter_by_token_length(all_samples, tokenizer, args.max_tokens)

    if args.init_target_share > 0:
        all_samples = _subsample_to_token_share(
            all_samples, tokenizer, "init_correct_reconstructed", args.init_target_share,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("wrote %d samples → %s", len(all_samples), args.output)

    cats = Counter(s.get("category") for s in all_samples)
    logger.info("category breakdown:")
    for cat, cnt in sorted(cats.items()):
        logger.info("  %s: %d", cat, cnt)

    sizes = [s["n_tokens"] for s in all_samples if "n_tokens" in s]
    if sizes:
        sizes.sort()
        logger.info("prompt+completion tokens: min=%d  median=%d  p95=%d  max=%d",
                    sizes[0], sizes[len(sizes)//2], sizes[int(0.95*len(sizes))], sizes[-1])


if __name__ == "__main__":
    main()
