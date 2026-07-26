"""Build per-turn SFT JSONL from CORRECT *pure-ReAct* (2-tool) initial trajectories.

Self-distillation warm-up: train Qwen3.5-9B on its OWN correct ReAct rollouts
(search + get_document only — NO memory tools). Unlike
`src.teacher_correct_oversummary.build_init_correct_sft` (which relabels initial into
the 4-tool memory schema), this keeps the 2-tool system prompt + tool set, so
the samples match the inference view of a no-memory-tool agent.

Re-uses the teacher-guided preprocess's `emit_per_turn_samples`. ReAct trajectories contain no
`manage_context` compressions, so prefix reconstruction (boundary_0=2) is a
straight slice of the live history.

Usage:
    python -m train.build_react_sft \\
        --rollouts_dir results/browsecomp-plus/qwen3.5-9b-base/rollout-teacher_guide-init/run_all \\
        --grade_path   results/browsecomp-plus/qwen3.5-9b-base/rollout-teacher_guide-init/run_all/gpt5_eval.json \\
        --output       data/sft/react_init_warmup.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.prompts import build_system_prompt
from src.tools import get_tools
from train.preprocess_teacher_guide import emit_per_turn_samples, _normalize_messages


logger = logging.getLogger("build_react_sft")


def emit_messages_sample(*, raw, sys_prompt, tools, tokenizer, qid, max_tokens):
    """Whole-trajectory conversational sample for TRL `assistant_only_loss`.

    Keeps the full message list (one sample per trajectory, loss on all assistant
    tokens via the chat template's `{% generation %}` mask). Rebuilds history[0] as
    the 2-tool system prompt and carries `tools` so TRL renders the <tools> block.
    Returns (sample | None, skip_reason | None).
    """
    raw_norm = _normalize_messages(raw)
    messages = [{"role": "system", "content": sys_prompt}] + list(raw_norm[1:])
    try:
        # Render to text then tokenize (matches the per-turn length measurement;
        # apply_chat_template(tokenize=True) miscounts for this tokenizer).
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
    except Exception as e:
        logger.warning("chat_template failed for messages qid=%s: %s", qid, e)
        return None, "render_fail"
    n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if max_tokens > 0 and n_tok > max_tokens:
        return None, "too_long"
    return {
        "messages": messages,
        "tools": tools,
        "qid": qid,
        "source": "react_init",
        "category": "react_init_correct",
        "n_tokens": n_tok,
    }, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts_dir", required=True, type=Path)
    ap.add_argument("--grade_path",   required=True, type=Path)
    ap.add_argument("--output",       required=True, type=Path)
    ap.add_argument("--tokenizer",    default="Qwen/Qwen3.5-9B")
    ap.add_argument("--benchmark",    default="browsecomp-plus")
    ap.add_argument("--context_window", type=int, default=131072)
    ap.add_argument("--max_tokens",   type=int, default=32768,
                    help="Drop samples whose rendered length exceeds this (0 = no filter)")
    ap.add_argument("--format",       choices=["prompt_completion", "messages"],
                    default="prompt_completion",
                    help="prompt_completion = per-turn unroll, loss on each completion "
                         "(TRL completion_only_loss). messages = whole trajectory in chat "
                         "format, one sample per trajectory (TRL assistant_only_loss).")
    ap.add_argument("--log_level",    default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # 2-tool ReAct: NO memory tools.
    sys_prompt_2tool = build_system_prompt(
        benchmark=args.benchmark,
        context_window=args.context_window,
        use_memory_tools=False,
    )
    tools_2tool = get_tools(benchmark=args.benchmark, use_memory_tools=False)

    correct_qids = {q["id"] for q in
                    json.loads(args.grade_path.read_text(encoding="utf-8")).get("per_question", [])
                    if q.get("correct")}
    logger.info("grade: %d correct qids", len(correct_qids))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    skipped: Counter = Counter()
    n_trajs = 0

    for fp in sorted(args.rollouts_dir.glob("run_*.json")):
        if fp.name.endswith(".partial.json"):
            continue
        qid = fp.stem.removeprefix("run_")
        if qid not in correct_qids:
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped["broken_json"] += 1
            continue
        raw = d.get("raw_history") or d.get("history") or []
        if len(raw) < 3:
            skipped["short_raw"] += 1
            continue
        asst_positions = [i for i, m in enumerate(raw) if m.get("role") == "assistant"]
        if not asst_positions:
            skipped["no_asst"] += 1
            continue
        n_trajs += 1

        if args.format == "messages":
            sample, skip = emit_messages_sample(
                raw=raw, sys_prompt=sys_prompt_2tool, tools=tools_2tool,
                tokenizer=tokenizer, qid=qid, max_tokens=args.max_tokens,
            )
            if sample is not None:
                samples.append(sample)
            else:
                skipped[skip] += 1
        else:
            new_samples, skips = emit_per_turn_samples(
                raw=raw,
                asst_positions=asst_positions,
                boundary_0=2,
                sys_prompt_4tool=sys_prompt_2tool,   # generic helper: pass the 2-tool prompt
                tools_4tool=tools_2tool,             # ...and the 2-tool set
                tokenizer=tokenizer,
                source_label="react_init",
                qid=qid,
                category="react_init_correct",
            )
            samples.extend(new_samples)
            skipped.update(skips)

    # length filter + count tokens (messages samples are already filtered + counted
    # inside emit_messages_sample; only prompt-completion needs the post-pass).
    if args.format == "prompt_completion":
        if args.max_tokens > 0:
            kept: list[dict] = []
            for s in samples:
                n_tok = len(tokenizer(s["prompt"] + s["completion"], add_special_tokens=False)["input_ids"])
                if n_tok > args.max_tokens:
                    skipped["too_long"] += 1
                    continue
                s["n_tokens"] = n_tok
                kept.append(s)
            samples = kept
        else:
            for s in samples:
                s["n_tokens"] = len(tokenizer(s["prompt"] + s["completion"], add_special_tokens=False)["input_ids"])

    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("emitted %d samples from %d correct trajs; skipped=%s",
                len(samples), n_trajs, dict(skipped))
    if samples:
        toks = sorted([s["n_tokens"] for s in samples])
        logger.info("token len: min=%d median=%d p95=%d max=%d total=%d",
                    toks[0], toks[len(toks)//2], toks[int(len(toks)*0.95)], toks[-1], sum(toks))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
