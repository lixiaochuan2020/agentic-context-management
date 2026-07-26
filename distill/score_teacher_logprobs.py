"""Stage 1 of offline on-policy distillation: score student trajectories with the teacher.

Pipeline: take the base-9B MemTool rollouts on bcp_train_680, keep the *harder* subset
(questions NOT solved 4/4 — incl. the 0/4 fails, where the teacher adds the most signal),
re-render each trajectory with the SAME renderer the SFT data uses (emit_messages_sample),
tokenize once with the assistant-token mask, and ask the teacher's vLLM `prompt_logprobs`
for its top-K distribution at every assistant position. We cache the *fully tokenized*
example + teacher top-K so Stage-2 training consumes it directly (no re-tokenization →
teacher/student token positions stay aligned by construction).

Teacher must already be serving (see scripts/serve_teacher_397b.sh) with prompt_logprobs.

Output: one .npz per trajectory at <out_dir>/<qid>_run<rep>.npz with
  input_ids   int32[L]          full token sequence (system+tools+turns)
  asst_pos    int32[M]          indices i where assistant_mask[i] is True
  tk_ids      int32[M, K]       teacher top-K token ids at each asst position
  tk_logprobs float16[M, K]     teacher top-K logprobs (natural log)
Alignment note: tk_*[m] is the teacher distribution *for* token input_ids[asst_pos[m]]
(i.e. vLLM prompt_logprobs[i]); Stage-2 supervises student logits[i-1] against it.

Run (teacher serving locally on :8900):
  python -m distill.score_teacher_logprobs \
      --runs_root <results>/browsecomp-plus/qwen3.5-9b-base \
      --tokenizer Qwen/Qwen3.5-9B \
      --teacher_api_base http://localhost:8900/v1 --teacher_model qwen3.5-397b-teacher \
      --top_k 20 --out_dir <out>/teacher_logprobs
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from src.prompts import build_system_prompt
from src.tools import get_tools
from train.build_react_sft import emit_messages_sample
from train.chat_masking import chatml_consts, assistant_positions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_teacher")


def select_trajectories(runs_root: str, reps, keep_solve_counts, run_tag="memtool-train680-run") -> list[tuple[str, int, str]]:
    """Return [(qid, rep, traj_path)] for questions whose 4-rep solve-count is in keep_solve_counts."""
    correct = defaultdict(dict)  # qid -> {rep: bool}
    traj = {}                    # (qid, rep) -> rollout json path
    for rep in reps:
        run = f"{runs_root}/{run_tag}{rep}/run_all"
        ev = f"{runs_root}/eval_bcp/{run_tag}{rep}/run_all"
        for f in glob.glob(f"{run}/run_*.json"):
            if f.endswith(".partial.json"):
                continue
            qid = Path(f).stem.split("_", 1)[1]
            traj[(qid, rep)] = f
        for f in glob.glob(f"{ev}/run_*_eval.json"):
            qid = Path(f).stem[len("run_"):-len("_eval")]
            try:
                correct[qid][rep] = bool((json.load(open(f)).get("judge_result") or {}).get("correct"))
            except Exception:
                correct[qid][rep] = False
    sel = []
    for qid, perrep in correct.items():
        if sum(perrep.values()) in keep_solve_counts:
            for rep in reps:
                if (qid, rep) in traj:
                    sel.append((qid, rep, traj[(qid, rep)]))
    return sel


def teacher_prompt_logprobs(api_base: str, model: str, input_ids: list[int], top_k: int):
    """vLLM teacher-forced scoring: top-K logprobs at each prompt position. prompt_logprobs[0] is None."""
    r = requests.post(
        f"{api_base}/completions",
        json={"model": model, "prompt": input_ids, "max_tokens": 1,
              "temperature": 0, "prompt_logprobs": top_k},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["prompt_logprobs"]


def pack_topk(plp_at_pos: dict, top_k: int):
    """{token_id_str: {logprob,...}} -> (ids[K], logprobs[K]) sorted by logprob desc, padded."""
    items = sorted(((int(t), d["logprob"]) for t, d in plp_at_pos.items()), key=lambda x: -x[1])[:top_k]
    ids = np.full(top_k, -1, np.int32)
    lps = np.full(top_k, -1e30, np.float32)
    for j, (t, lp) in enumerate(items):
        ids[j], lps[j] = t, lp
    return ids, lps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, help="…/browsecomp-plus/qwen3.5-9b-base")
    ap.add_argument("--run_tag", default="memtool-train680-run", help="rollout run-dir prefix (iter-2: memtool-train680-iter2-run)")
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--keep_solve_counts", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="solve-counts to score (default: the harder subset, excludes 4/4)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--benchmark", default="browsecomp-plus")
    ap.add_argument("--context_window", type=int, default=131072)
    ap.add_argument("--max_tokens", type=int, default=262144,
                    help="drop trajectories longer than this (256K matches the long-context SFT build "
                         "and the teacher's 262144 max_position_embeddings)")
    ap.add_argument("--teacher_api_base", required=True)
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #trajectories (0 = all); for smoke tests")
    ap.add_argument("--shard", type=int, default=0, help="this process's shard index")
    ap.add_argument("--num_shards", type=int, default=1, help="run N processes against one teacher for concurrency")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cm = chatml_consts(tok)
    sysp = build_system_prompt(benchmark=args.benchmark, context_window=args.context_window,
                               use_memory_tools=True)
    tools = get_tools(benchmark=args.benchmark, use_memory_tools=True)

    sel = select_trajectories(args.runs_root, args.reps, set(args.keep_solve_counts), run_tag=args.run_tag)
    total = len(sel)
    if args.num_shards > 1:
        sel = sel[args.shard::args.num_shards]   # strided → balanced load across shards
    if args.limit:
        sel = sel[: args.limit]
    log.info("shard %d/%d: %d of %d trajectories (solve-counts %s, reps %s)",
             args.shard, args.num_shards, len(sel), total, args.keep_solve_counts, args.reps)

    done = skipped = 0
    for qid, rep, path in tqdm(sel, desc="scoring"):
        out = os.path.join(args.out_dir, f"{qid}_run{rep}.npz")
        if os.path.exists(out):
            done += 1
            continue
        raw = (json.load(open(path)).get("raw_history") or json.load(open(path)).get("history") or [])
        s, why = emit_messages_sample(raw=raw, sys_prompt=sysp, tools=tools, tokenizer=tok,
                                      qid=qid, max_tokens=args.max_tokens)
        if s is None:
            skipped += 1
            continue
        ids = tok.apply_chat_template(s["messages"], tools=tools, tokenize=True,
                                      return_dict=True, add_generation_prompt=False)["input_ids"]
        asst_pos = [i for i in assistant_positions(ids, cm) if i > 0]
        if not asst_pos or len(ids) > args.max_tokens:
            skipped += 1
            continue
        plp = teacher_prompt_logprobs(args.teacher_api_base, args.teacher_model, ids, args.top_k)
        tk_ids = np.zeros((len(asst_pos), args.top_k), np.int32)
        tk_lps = np.zeros((len(asst_pos), args.top_k), np.float16)
        for m, i in enumerate(asst_pos):
            tk_ids[m], lps = pack_topk(plp[i], args.top_k)
            tk_lps[m] = lps.astype(np.float16)
        np.savez_compressed(out, input_ids=np.asarray(ids, np.int32),
                            asst_pos=np.asarray(asst_pos, np.int32), tk_ids=tk_ids, tk_logprobs=tk_lps)
        done += 1

    log.info("done: cached %d, skipped %d → %s", done, skipped, args.out_dir)


if __name__ == "__main__":
    main()
