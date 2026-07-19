"""
Agentic Context Management — CLI entry point.

Modes:
  run                    Standard benchmark run (browsecomp or browsecomp-plus)
  scaling-from-scratch   All queries, fresh start, forced extra turns beyond reference
  scaling-from-errors    Only wrong queries, resume from previous conversation
  eval                   Grade existing results

Directory structure:
    results/<benchmark>/<model>/<run_dir>/<run_id>/

Config loading:
  defaults (src/configs/*.py dataclasses)
    ← merged with → configs/default.yaml (or --config path)
    ← merged with → --override key=val ...
    ← merged with → explicit named CLI flags (--model, --bcp_k, etc.)

Named CLI flags below all default to None — they only override RunConfig
when the user actually passes them. The dataclass defaults are the single
source of truth.

Examples:
    # Defaults only (project YAML auto-loaded from configs/default.yaml):
    python -m src.run --mode run --run_dir baseline --client litellm

    # Override two fields via CLI:
    python -m src.run --mode run --run_dir baseline --client litellm \\
        --override retrieval.k=20 agent.context_window=131072
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.config import (
    DATA_PATH,
    GRADER_API_KEY,
    GRADER_MODEL,
    MODEL_PATH,
    RESULTS_DIR,
)
from src.client import BaseClient
from src.configs import RunConfig, load_config
from src.evaluator import aggregate_results, compute_pilot_metrics, grade_answer
from src.runner import run

logger = logging.getLogger("src")


# ── I/O helpers ───────────────────────────────────────────

def save_result(result: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("  Saved result to %s", path)


def get_question(item: dict) -> str:
    """Get question text from dataset item (standard field: 'question')."""
    return item.get("question") or item.get("problem", "")


def load_result(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Per-question config helpers ───────────────────────────

def _with_question(
    base_config: RunConfig,
    *,
    qid: str,
    correct_answer: str,
    cache_path: str | None,
    workspace_subdir: str | None = None,
    initial_history: list[dict] | None = None,
    min_turns_override: int | None = None,
) -> RunConfig:
    """Build a per-question RunConfig from base_config.

    The runtime sub-config may be tweaked (workspace_root per question,
    min_turns per scaling run); per-question top-level fields are filled in.
    """
    runtime = base_config.runtime
    if workspace_subdir is not None or min_turns_override is not None:
        kwargs = {}
        if workspace_subdir is not None:
            kwargs["workspace_root"] = workspace_subdir
        if min_turns_override is not None:
            kwargs["min_turns"] = min_turns_override
        runtime = replace(runtime, **kwargs)

    return replace(
        base_config,
        runtime=runtime,
        question_id=qid,
        correct_answer=correct_answer,
        cache_path=cache_path,
        initial_history=initial_history,
    )


# ── Unified run phase ────────────────────────────────────

def run_phase(
    client: BaseClient,
    dataset: list[dict],
    out_dir: str,
    base_config: RunConfig,
    *,
    limit: int | None = None,
    skip_existing: bool = True,
    shard: int = 0,
    num_shards: int = 1,
):
    """Run benchmark for each question in the dataset."""
    os.makedirs(out_dir, exist_ok=True)

    questions = dataset[:limit] if limit else dataset
    # Modulo-sharding: worker K of N processes questions at positions i where i % N == K.
    if num_shards > 1:
        questions = [q for i, q in enumerate(questions) if i % num_shards == shard]
    logger.info(
        "=== run | benchmark=%s | memory_tools=%s | shard=%d/%d | questions=%d ===",
        base_config.runtime.benchmark, base_config.runtime.use_memory_tools,
        shard, num_shards, len(questions),
    )

    uses_run_prefix = base_config.runtime.benchmark in ("browsecomp-plus", "deepresearch9k")

    for i, item in enumerate(questions):
        qid = item.get("id", f"q{i}")
        prefix = "run_" if uses_run_prefix else ""
        out_path = os.path.join(out_dir, f"{prefix}{qid}.json")
        cache_path = os.path.join(out_dir, f"{prefix}{qid}.partial.json")

        if skip_existing and os.path.exists(out_path):
            logger.info("[%d/%d] %s — skipping (exists)", i + 1, len(questions), qid)
            continue

        logger.info("[%d/%d] %s — running ...", i + 1, len(questions), qid)
        t0 = time.time()

        workspace_subdir = None
        if base_config.runtime.use_memory_tools:
            workspace_subdir = os.path.join(out_dir, "workspace", str(qid))
            os.makedirs(workspace_subdir, exist_ok=True)

        config = _with_question(
            base_config,
            qid=str(qid),
            correct_answer=item["answer"],
            cache_path=cache_path,
            workspace_subdir=workspace_subdir,
        )

        try:
            result = run(client, get_question(item), config, answer_type=item.get("answer_type"))
        except Exception as e:
            logger.error("[%d/%d] %s — CRASHED: %s\n%s", i + 1, len(questions), qid, e, traceback.format_exc())
            crash_result = {
                "query_id": qid, "question": get_question(item),
                "correct_answer": item["answer"],
                "elapsed_sec": round(time.time() - t0, 1),
                "status": "crashed", "error": f"{type(e).__name__}: {e}",
                "final_answer": "", "confidence": 0, "num_turns": 0,
                "token_trajectory": [], "tool_calls": [], "mem_operations": [],
                "result": [], "retrieved_docids": [],
                "history": [], "raw_history": [],
            }
            save_result(crash_result, out_path)
            continue

        result["elapsed_sec"] = round(time.time() - t0, 1)

        sample_cost = client.get_and_reset_cost() if hasattr(client, "get_and_reset_cost") else None
        save_result(result, out_path)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        answer_preview = result["final_answer"][:60]
        if sample_cost is not None:
            logger.info(
                "[%d/%d] %s — done in %.1fs, answer='%s', turns=%d, cost=$%.4f (total=$%.4f)",
                i + 1, len(questions), qid, result["elapsed_sec"],
                answer_preview, result["num_turns"],
                sample_cost, client.total_cost,
            )
        else:
            logger.info(
                "[%d/%d] %s — done in %.1fs, answer='%s', turns=%d",
                i + 1, len(questions), qid, result["elapsed_sec"],
                answer_preview, result["num_turns"],
            )


# ── Scaling-from-scratch helpers ─────────────────────────

def load_ref_turns(ref_dir: str) -> dict[str, int]:
    """Load per-query num_turns from reference run directory."""
    ref_turns: dict[str, int] = {}
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")
    for fname in os.listdir(ref_dir):
        if not fname.startswith("run_") or not fname.endswith(".json"):
            continue
        if fname.endswith(".partial.json") or fname.endswith("_eval.json"):
            continue
        qid = fname[len("run_"):-len(".json")]
        fpath = os.path.join(ref_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            ref_turns[qid] = data.get("num_turns", 0)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read reference file %s: %s", fpath, e)
    logger.info("Loaded reference turns for %d queries from %s", len(ref_turns), ref_dir)
    return ref_turns


def run_scaling_from_scratch_phase(
    client: BaseClient,
    dataset: list[dict],
    out_dir: str,
    base_config: RunConfig,
    *,
    ref_dir: str,
    extra_turns: int = 5,
    limit: int | None = None,
    skip_existing: bool = True,
):
    """All queries, fresh start, forced extra turns beyond reference run."""
    os.makedirs(out_dir, exist_ok=True)
    ref_turns = load_ref_turns(ref_dir)
    questions = dataset[:limit] if limit else dataset
    logger.info(
        "=== scaling-from-scratch | extra_turns=%d | questions=%d ===",
        extra_turns, len(questions),
    )

    # Force memory tools on for scaling.
    base_config = replace(base_config, runtime=replace(base_config.runtime, use_memory_tools=True))

    for i, item in enumerate(questions):
        qid = item.get("id", f"q{i}")
        out_path = os.path.join(out_dir, f"run_{qid}.json")
        cache_path = os.path.join(out_dir, f"run_{qid}.partial.json")

        if skip_existing and os.path.exists(out_path):
            logger.info("[%d/%d] %s — skipping (exists)", i + 1, len(questions), qid)
            continue

        prev_turns = ref_turns.get(str(qid))
        min_turns = (prev_turns + extra_turns) if prev_turns is not None else extra_turns

        logger.info("[%d/%d] %s — scaling (min_turns=%d) ...", i + 1, len(questions), qid, min_turns)
        t0 = time.time()

        workspace_subdir = os.path.join(out_dir, "workspace", str(qid))
        os.makedirs(workspace_subdir, exist_ok=True)

        config = _with_question(
            base_config,
            qid=str(qid),
            correct_answer=item["answer"],
            cache_path=cache_path,
            workspace_subdir=workspace_subdir,
            min_turns_override=min_turns,
        )

        try:
            result = run(client, get_question(item), config, answer_type=item.get("answer_type"))
        except Exception as e:
            logger.error("[%d/%d] %s — CRASHED: %s\n%s", i + 1, len(questions), qid, e, traceback.format_exc())
            crash_result = {
                "query_id": qid, "question": get_question(item),
                "correct_answer": item["answer"],
                "elapsed_sec": round(time.time() - t0, 1),
                "status": "crashed", "error": f"{type(e).__name__}: {e}",
                "final_answer": "", "confidence": 0, "num_turns": 0,
                "min_turns": min_turns, "ref_turns": prev_turns, "extra_turns": extra_turns,
                "token_trajectory": [], "tool_calls": [], "mem_operations": [],
                "result": [], "retrieved_docids": [],
                "history": [], "raw_history": [],
            }
            save_result(crash_result, out_path)
            continue

        result["elapsed_sec"] = round(time.time() - t0, 1)
        result["ref_turns"] = prev_turns
        result["extra_turns"] = extra_turns

        sample_cost = client.get_and_reset_cost() if hasattr(client, "get_and_reset_cost") else None
        save_result(result, out_path)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        logger.info(
            "[%d/%d] %s — done in %.1fs, answer='%s', turns=%d (min=%d)",
            i + 1, len(questions), qid, result["elapsed_sec"],
            result["final_answer"][:60], result["num_turns"], min_turns,
        )


# ── Scaling-from-errors helpers ──────────────────────────

def load_wrong_qids(eval_summary_path: str) -> set[str]:
    if not os.path.exists(eval_summary_path):
        raise FileNotFoundError(f"Evaluation summary not found: {eval_summary_path}")
    with open(eval_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    per_query = summary.get("per_query_metrics", [])
    wrong_qids = {str(m["query_id"]) for m in per_query if not m.get("correct", False)}
    logger.info("Loaded %d wrong query IDs from %s (out of %d total)",
                len(wrong_qids), eval_summary_path, len(per_query))
    return wrong_qids


def load_ref_result(ref_dir: str, qid: str) -> dict | None:
    fpath = os.path.join(ref_dir, f"run_{qid}.json")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if data.get("history") else None
    except (json.JSONDecodeError, OSError):
        return None


def _copy_workspace(ref_dir: str, qid: str, dest_workspace: str):
    import shutil
    import glob as _glob
    src_workspace = os.path.join(ref_dir, "workspace", qid)
    if not os.path.isdir(src_workspace):
        return
    for fpath in _glob.glob(os.path.join(src_workspace, "*")):
        dest = os.path.join(dest_workspace, os.path.basename(fpath))
        if not os.path.exists(dest):
            shutil.copy2(fpath, dest)


def run_scaling_from_errors_phase(
    client: BaseClient,
    dataset: list[dict],
    out_dir: str,
    base_config: RunConfig,
    *,
    ref_dir: str,
    eval_summary_path: str,
    extra_turns: int = 5,
    limit: int | None = None,
    skip_existing: bool = True,
):
    """Only wrong queries, resume from previous conversation, merge metrics."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    wrong_qids = load_wrong_qids(eval_summary_path)
    questions = dataset[:limit] if limit else dataset

    logger.info(
        "=== scaling-from-errors | extra_turns=%d | wrong=%d | questions=%d ===",
        extra_turns, len(wrong_qids), len(questions),
    )

    # Copy correct-answer files so output dir is a complete set
    correct_copied = 0
    for item in questions:
        qid = item.get("id", f"q{questions.index(item)}")
        if str(qid) in wrong_qids:
            continue
        src = os.path.join(ref_dir, f"run_{qid}.json")
        dst = os.path.join(out_dir, f"run_{qid}.json")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            correct_copied += 1
    logger.info("Copied %d correct-answer files from ref_dir to output", correct_copied)

    for i, item in enumerate(questions):
        qid = item.get("id", f"q{i}")
        if str(qid) not in wrong_qids:
            continue

        out_path = os.path.join(out_dir, f"run_{qid}.json")
        cache_path = os.path.join(out_dir, f"run_{qid}.partial.json")

        if skip_existing and os.path.exists(out_path):
            logger.info("[%d/%d] %s — skipping (exists)", i + 1, len(questions), qid)
            continue

        ref_data = load_ref_result(ref_dir, str(qid))
        if ref_data is None:
            src = os.path.join(ref_dir, f"run_{qid}.json")
            if os.path.exists(src) and not os.path.exists(out_path):
                shutil.copy2(src, out_path)
            continue

        ref_history = ref_data["history"]
        logger.info("[%d/%d] %s — resuming (%d history msgs, min_turns=%d) ...",
                    i + 1, len(questions), qid, len(ref_history), extra_turns)
        t0 = time.time()

        workspace_subdir = os.path.join(out_dir, "workspace", str(qid))
        os.makedirs(workspace_subdir, exist_ok=True)
        _copy_workspace(ref_dir, str(qid), workspace_subdir)

        config = _with_question(
            base_config,
            qid=str(qid),
            correct_answer=item["answer"],
            cache_path=cache_path,
            workspace_subdir=workspace_subdir,
            min_turns_override=extra_turns,
            initial_history=ref_history,
        )

        try:
            result = run(client, get_question(item), config, answer_type=item.get("answer_type"))
        except Exception as e:
            logger.error("[%d/%d] %s — CRASHED: %s\n%s",
                         i + 1, len(questions), qid, e, traceback.format_exc())
            crash_result = {
                "query_id": qid, "question": get_question(item),
                "correct_answer": item["answer"],
                "elapsed_sec": round(time.time() - t0, 1),
                "status": "crashed", "error": f"{type(e).__name__}: {e}",
                "final_answer": "", "confidence": 0, "num_turns": 0,
                "token_trajectory": [], "tool_calls": [], "mem_operations": [],
                "result": [], "retrieved_docids": [],
                "history": [], "raw_history": [],
            }
            save_result(crash_result, out_path)
            continue

        result["elapsed_sec"] = round(time.time() - t0, 1)
        result["ref_dir"] = ref_dir
        result["extra_turns"] = extra_turns

        # Incremental merge: accumulate on top of reference metrics
        result["ref_num_turns"] = ref_data.get("num_turns", 0)
        result["num_turns"] += ref_data.get("num_turns", 0)

        ref_usage = ref_data.get("usage", {})
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            result["usage"][k] = result["usage"].get(k, 0) + ref_usage.get(k, 0)

        ref_tc = ref_data.get("tool_call_counts", {})
        for k, v in ref_tc.items():
            result["tool_call_counts"][k] = result["tool_call_counts"].get(k, 0) + v

        ref_docids = set(ref_data.get("retrieved_docids", []))
        result["retrieved_docids"] = sorted(set(result.get("retrieved_docids", [])) | ref_docids)

        for key in ("token_trajectory", "tool_calls", "mem_operations", "result"):
            result[key] = ref_data.get(key, []) + result.get(key, [])

        save_result(result, out_path)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        logger.info(
            "[%d/%d] %s — done in %.1fs, answer='%s', turns=%d",
            i + 1, len(questions), qid, result["elapsed_sec"],
            result["final_answer"][:60], result["num_turns"],
        )


# ── Evaluation ────────────────────────────────────────────

def run_eval_phase(
    dataset: list[dict],
    mode_dirs: dict[str, str],
    summary_path: str,
    limit: int | None = None,
):
    """Grade existing results and aggregate metrics."""
    summary = {}
    per_sample: dict[str, dict] = {}

    for mode in mode_dirs:
        mode_dir = mode_dirs.get(mode)
        if not mode_dir or not os.path.isdir(mode_dir):
            logger.warning("  No results dir for %s", mode)
            continue

        questions = dataset[:limit] if limit else dataset
        grade_results = []
        pilot_metrics_list = []

        for i, item in enumerate(questions):
            qid = item.get("id", f"q{i}")
            result_path = os.path.join(mode_dir, f"{qid}.json")
            result = load_result(result_path)
            if result is None:
                grade = {"correct": False, "is_correct": False, "question_id": qid}
                grade_results.append(grade)
                pilot_metrics_list.append({"token_peak": 0, "tokens_cumulative_raw": 0,
                                           "output_tokens_cumulative": 0,
                                           "num_summary_calls": 0, "num_discard_calls": 0})
                if qid not in per_sample:
                    per_sample[qid] = {
                        "question_id": qid,
                        "question": get_question(item)[:200],
                        "correct_answer": item["answer"],
                    }
                per_sample[qid][mode] = {
                    "final_answer": "", "correct": False, "confidence": 0,
                    "num_turns": 0, "final_tokens": 0, "token_peak": 0,
                    "num_summary_calls": 0, "num_discard_calls": 0,
                    "elapsed_sec": 0, "mem_operations": [],
                }
                continue

            if "grade" in result:
                grade = result["grade"]
            else:
                response_text = _extract_last_response(result)
                grade = grade_answer(
                    question=get_question(item),
                    correct_answer=item["answer"],
                    response=response_text,
                )
                grade["question_id"] = qid
                result["grade"] = grade
                save_result(result, result_path)

            grade_results.append(grade)
            metrics = compute_pilot_metrics(result)
            pilot_metrics_list.append(metrics)

            if qid not in per_sample:
                per_sample[qid] = {
                    "question_id": qid,
                    "question": get_question(item)[:200],
                    "correct_answer": item["answer"],
                }
            traj = result.get("token_trajectory", [])
            if traj and isinstance(traj[0], dict):
                final_tokens = traj[-1].get("tokens_actual", 0)
            elif traj:
                final_tokens = traj[-1]
            else:
                final_tokens = 0

            per_sample[qid][mode] = {
                "final_answer": result.get("final_answer", ""),
                "correct": grade.get("correct", False),
                "confidence": result.get("confidence", 0),
                "num_turns": result.get("num_turns", 0),
                "final_tokens": final_tokens,
                "token_peak": metrics["token_peak"],
                "tokens_cumulative_raw": metrics["tokens_cumulative_raw"],
                "output_tokens_cumulative": metrics["output_tokens_cumulative"],
                "num_summary_calls": metrics["num_summary_calls"],
                "num_discard_calls": metrics["num_discard_calls"],
                "elapsed_sec": result.get("elapsed_sec", 0),
                "mem_operations": result.get("mem_operations", []),
            }

        if grade_results:
            agg = aggregate_results(grade_results)
            agg["mode"] = mode

            if pilot_metrics_list:
                n = len(pilot_metrics_list)
                for k in ("token_peak", "tokens_cumulative_raw", "output_tokens_cumulative",
                           "num_summary_calls", "num_discard_calls"):
                    agg[f"avg_{k}"] = sum(m[k] for m in pilot_metrics_list) / n

            summary[mode] = agg
            logger.info("  [eval] %s: accuracy=%.4f (%d/%d)",
                        mode, agg["accuracy"],
                        int(agg["accuracy"] * agg["num_samples"]), agg["num_samples"])

    summary["per_sample"] = list(per_sample.values())
    save_result(summary, summary_path)
    logger.info("=== Evaluation summary saved to %s ===", summary_path)
    return summary


def _extract_last_response(result: dict) -> str:
    history = result.get("history", [])
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return result.get("final_answer", "")


# ── CLI ───────────────────────────────────────────────────

# Translation table: argparse flag → RunConfig dotted path. Used only when
# the user explicitly passes the flag (default=None means "leave RunConfig
# default alone").
_CLI_TO_DOTTED: dict[str, str] = {
    "model":              "agent.model",
    "agent_api_base":     "agent.api_base",
    "benchmark":          "runtime.benchmark",
    "index_path":         "retrieval.index_path",
    "bcp_k":              "retrieval.k",
    "dr9k_k":             "retrieval.k",
    "server_url":         "retrieval.retrieval_server_url",
    "summarizer_model":   "summarizer.model",
    "summarizer_api_base": "summarizer.api_base",
}


def _cli_to_overrides(args: argparse.Namespace) -> list[str]:
    """Convert explicit (non-None) named CLI flags to OmegaConf dotlist entries."""
    overrides: list[str] = []
    for flag, dotted in _CLI_TO_DOTTED.items():
        v = getattr(args, flag, None)
        if v is None or v == "":
            continue
        overrides.append(f"{dotted}={v}")
    # store_true flag — only override when set
    if getattr(args, "use_memory_tool", False):
        overrides.append("runtime.use_memory_tools=true")
    if getattr(args, "no_query_memory", False):
        overrides.append("runtime.disable_query_memory=true")
    if getattr(args, "mc_drop_call_message", False):
        overrides.append("runtime.mc_drop_call_message=true")
    if getattr(args, "edit_reminder", False):
        overrides.append("runtime.edit_reminder=true")
    # User-supplied --override key=val entries take priority.
    overrides.extend(getattr(args, "override", None) or [])
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Context Management — experiment runner"
    )
    # Mode + orchestration (not in RunConfig).
    parser.add_argument(
        "--mode",
        choices=["run", "scaling-from-scratch", "scaling-from-errors", "eval"],
        required=True,
        help="run = standard benchmark; "
             "scaling-from-scratch = all queries with forced extra turns; "
             "scaling-from-errors = resume wrong-answer queries from previous run; "
             "eval = grade existing results",
    )
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Dataset path")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--shard", type=int, default=0,
                        help="This worker's shard index (0..num_shards-1)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel workers (default: 1)")
    parser.add_argument("--no_skip", action="store_true", help="Force re-run existing results")
    parser.add_argument("--client", choices=["local", "litellm"], default="local")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--save_training_data", action="store_true")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: $SLURM_JOB_ID or timestamp)")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Experiment directory name (e.g. 'baseline', 'with_mem')")
    parser.add_argument("--extra_turns", type=int, default=5,
                        help="(scaling modes) Extra turns beyond reference")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="(scaling modes) Reference results directory")
    parser.add_argument("--eval_summary", type=str, default=None,
                        help="(scaling-from-errors) Path to evaluation_summary.json")

    # Config plumbing.
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to project YAML (overrides dataclass defaults). "
                             "Pass empty string to disable.")
    parser.add_argument("--override", nargs="*", default=[],
                        metavar="KEY=VAL",
                        help="OmegaConf dotlist overrides applied LAST, after YAML. "
                             "Example: --override retrieval.k=20 agent.model=foo")

    # Named CLI flags that map to RunConfig paths (all default=None — when
    # not passed, the corresponding RunConfig field keeps its dataclass
    # default or YAML value). See _CLI_TO_DOTTED.
    parser.add_argument("--model", type=str, default=None,
                        help="agent.model — LiteLLM model name (e.g. openai/<served-name>)")
    parser.add_argument("--agent_api_base", type=str,
                        default=os.environ.get("AGENT_API_BASE"),
                        help="agent.api_base. Defaults to $AGENT_API_BASE.")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="runtime.benchmark — browsecomp | browsecomp-plus | deepresearch9k")
    parser.add_argument("--use_memory_tool", action="store_true",
                        help="runtime.use_memory_tools (manage_context, query_memory)")
    parser.add_argument("--no_query_memory", action="store_true",
                        help="runtime.disable_query_memory — re_mem_noquery ablation: keep "
                             "manage_context compression but drop query_memory (archived "
                             "messages become unreachable). Use with --use_memory_tool.")
    parser.add_argument("--mc_drop_call_message", action="store_true",
                        help="runtime.mc_drop_call_message — include the manage_context "
                             "assistant turn in the summarized+deleted range, leaving only "
                             "the tool-result summary in live history")
    parser.add_argument("--edit_reminder", action="store_true",
                        help="runtime.edit_reminder — for SWE-bench, append an explicit "
                             "reminder to the system prompt that the model must edit a "
                             "source file before calling submit_patch")
    parser.add_argument("--index_path", type=str, default=None,
                        help="retrieval.index_path — BM25 Lucene dir")
    parser.add_argument("--bcp_k", type=int, default=None,
                        help="retrieval.k (alias) — BCP top-k")
    parser.add_argument("--dr9k_k", type=int, default=None,
                        help="retrieval.k (alias) — DR9K top-k")
    parser.add_argument("--server_url", type=str, default=None,
                        help="retrieval.retrieval_server_url — DR9K server")
    parser.add_argument("--summarizer_model", type=str, default=None,
                        help="summarizer.model")
    parser.add_argument("--summarizer_api_base", type=str,
                        default=os.environ.get("SUMMARIZER_API_BASE"),
                        help="summarizer.api_base. Defaults to $SUMMARIZER_API_BASE.")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Build base RunConfig ─────────────────────────────
    yaml_path = args.config or None
    if yaml_path and not Path(yaml_path).is_file():
        # configs/default.yaml is the default; treat missing as opt-out.
        logger.info("Config YAML %s not found — using dataclass defaults", yaml_path)
        yaml_path = None
    overrides = _cli_to_overrides(args)
    base_config = load_config(yaml_path=yaml_path, cli_overrides=overrides)

    # context_window: when the agent client is litellm, refine from model_info
    # — but only if the user did NOT explicitly pin agent.context_window.
    if args.client == "litellm" and "agent.context_window" not in " ".join(overrides):
        try:
            import litellm
            minfo = litellm.get_model_info(base_config.agent.model)
            ctx = minfo.get("max_input_tokens", base_config.agent.context_window)
            if ctx and ctx != base_config.agent.context_window:
                from src.configs.agent import AgentConfig
                base_config = replace(
                    base_config,
                    agent=replace(base_config.agent, context_window=ctx),
                )
                logger.info("Model context window: %d tokens", ctx)
        except Exception as e:
            logger.warning("Could not get model info: %s", e)

    logger.info("Resolved RunConfig:")
    logger.info("  agent.model      = %s", base_config.agent.model)
    logger.info("  agent.api_base   = %s", base_config.agent.api_base)
    logger.info("  agent.context_window = %d", base_config.agent.context_window)
    logger.info("  summarizer.model = %s", base_config.summarizer.model)
    logger.info("  retrieval.backend = %s  k=%d", base_config.retrieval.backend, base_config.retrieval.k)
    logger.info("  runtime.benchmark = %s  use_memory_tools=%s",
                base_config.runtime.benchmark, base_config.runtime.use_memory_tools)

    # ── Load dataset ─────────────────────────────────────
    from src.data_loading import load_data

    loader_kwargs = {}
    if base_config.runtime.benchmark == "browsecomp-plus" and args.data == DATA_PATH:
        bcp_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "BrowseComp-Plus")
        data_path = os.path.join(bcp_root, "topics-qrels", "queries.tsv")
        loader_kwargs["jsonl_path"] = os.path.join(bcp_root, "data", "browsecomp_plus_decrypted.jsonl")
    else:
        data_path = args.data

    dataset = load_data(base_config.runtime.benchmark, data_path, **loader_kwargs)

    benchmark = base_config.runtime.benchmark
    if benchmark == "browsecomp-plus" and not base_config.retrieval.index_path and args.mode != "eval":
        logger.error("retrieval.index_path is required for browsecomp-plus")
        return
    if (benchmark == "deepresearch9k" and not base_config.retrieval.retrieval_server_url
            and os.environ.get("DR9K_LIVE_WEB") != "1" and args.mode != "eval"):
        logger.error("retrieval.retrieval_server_url is required for deepresearch9k (or set DR9K_LIVE_WEB=1)")
        return

    skip_existing = not args.no_skip
    run_id = args.run_id or os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run ID: %s", run_id)

    model_name = base_config.agent.model if args.client == "litellm" else MODEL_PATH
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    out_dir = os.path.join(args.results_dir, benchmark, model_short, args.run_dir, run_id)

    model_base = os.path.join(args.results_dir, benchmark, model_short)
    if args.save_training_data:
        training_data_dir = os.path.join(model_base, "training_data", args.run_dir)
        base_config = replace(
            base_config,
            runtime=replace(base_config.runtime, training_data_dir=training_data_dir),
        )

    # When running under a local model (not litellm), the agent dispatches
    # through a different code path; sync agent.model to the local checkpoint
    # for trajectory metadata.
    if args.client != "litellm":
        base_config = replace(
            base_config,
            agent=replace(base_config.agent, model=MODEL_PATH),
        )

    client = None

    def get_client() -> BaseClient:
        nonlocal client
        if client is None:
            if args.client == "litellm":
                from src.client import LiteLLMClient
                client = LiteLLMClient(base_config.agent.model, api_base=base_config.agent.api_base)
            else:
                from src.client import ModelClient
                client = ModelClient(MODEL_PATH)
        return client

    # ── Dispatch ─────────────────────────────────────────
    if args.mode == "run":
        run_phase(
            get_client(), dataset, out_dir, base_config,
            limit=args.limit,
            skip_existing=skip_existing,
            shard=args.shard,
            num_shards=args.num_shards,
        )

    elif args.mode == "scaling-from-scratch":
        if args.ref_dir is None:
            logger.error("--ref_dir is required for scaling-from-scratch")
            return
        run_scaling_from_scratch_phase(
            get_client(), dataset, out_dir, base_config,
            ref_dir=args.ref_dir,
            extra_turns=args.extra_turns,
            limit=args.limit,
            skip_existing=skip_existing,
        )

    elif args.mode == "scaling-from-errors":
        if args.ref_dir is None:
            logger.error("--ref_dir is required for scaling-from-errors")
            return
        if args.eval_summary is None:
            logger.error("--eval_summary is required for scaling-from-errors")
            return
        run_scaling_from_errors_phase(
            get_client(), dataset, out_dir, base_config,
            ref_dir=args.ref_dir,
            eval_summary_path=args.eval_summary,
            extra_turns=args.extra_turns,
            limit=args.limit,
            skip_existing=skip_existing,
        )

    elif args.mode == "eval":
        if benchmark == "browsecomp-plus":
            from src.evaluator import evaluate_browsecomp_plus
            bcp_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "BrowseComp-Plus")
            gt_path = os.path.join(bcp_root, "data", "browsecomp_plus_decrypted.jsonl")
            qrel_path = os.path.join(bcp_root, "topics-qrels", "qrel_evidence.txt")
            eval_output_dir = os.path.join(model_base, "eval_bcp", args.run_dir, run_id)
            evaluate_browsecomp_plus(
                input_dir=out_dir,
                ground_truth=gt_path,
                eval_dir=eval_output_dir,
                model=GRADER_MODEL,
                qrel_evidence_path=qrel_path,
            )
        elif benchmark == "deepsearchqa":
            from src.evaluator import evaluate_deepsearchqa, DSQA_AUTORATER_MODEL
            # The args.data path is the canonical gold source (loader and
            # grader read the same file — see data/deepsearchqa{,_single}.json).
            # eval lives inside the run dir, mirroring BCP's eval_bcp_gpt5/ convention
            eval_output_dir = os.path.join(out_dir, "eval_dsqa")
            evaluate_deepsearchqa(
                input_dir=out_dir,
                ground_truth=args.data,
                eval_dir=eval_output_dir,
                model=DSQA_AUTORATER_MODEL,
            )
        else:
            summary_path = os.path.join(model_base, "eval", args.run_dir, run_id, "summary.json")
            run_eval_phase(dataset, {"default": out_dir}, summary_path, args.limit)

    logger.info("Done.")


if __name__ == "__main__":
    main()
