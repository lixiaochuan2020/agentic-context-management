"""Grade a BrowseComp-Plus run with GPT-5 LLM-as-judge.

Thin wrapper around src.evaluator.evaluate_browsecomp_plus():
1. Loads OPENAI_API_KEY from .env.
2. Calls the existing pipeline (judges every run_*.json, writes per-query
   <stem>_eval.json + evaluation_summary.json into a sibling eval_bcp/ dir).
3. Writes a single <input_dir>/gpt5_eval.json with summary + analysis stats
   (status, tool calls, turns, tokens, wallclock) + per_question detail.

Usage:
    python scripts/grade_bcp_gpt5.py --input_dir <run dir> [--workers 16] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from src.config import GRADER_MODEL  # noqa: E402
from src.eval_analysis import compute_analysis, print_summary  # noqa: E402
from src.evaluator import evaluate_browsecomp_plus  # noqa: E402

GT_PATH = REPO / "data" / "BrowseComp-Plus" / "data" / "browsecomp_plus_decrypted.jsonl"
QREL_PATH = REPO / "data" / "BrowseComp-Plus" / "topics-qrels" / "qrel_evidence.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Run directory containing run_*.json files")
    ap.add_argument("--output", type=Path, default=None,
                    help="Eval JSON path (default: <input_dir>/gpt5_eval.json)")
    ap.add_argument("--eval_dir", type=Path, default=None,
                    help="Where per-query *_eval.json + evaluation_summary.json go "
                         "(default: <input_dir>/eval_bcp_gpt5)")
    ap.add_argument("--workers", type=int, default=16,
                    help="Concurrent GPT-5 grading calls (default: 16)")
    ap.add_argument("--model", default=None,
                    help=f"Judge model (default: {GRADER_MODEL})")
    ap.add_argument("--force", action="store_true",
                    help="Re-grade even if cached *_eval.json exists")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"not a directory: {args.input_dir}")
    if not list(args.input_dir.glob("run_*.json")):
        print(f"no run_*.json under {args.input_dir} — nothing to grade")
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (expected in .env at repo root)")
    if not GT_PATH.is_file():
        raise SystemExit(f"missing ground truth: {GT_PATH}")

    # Grader hits api.openai.com directly. AGENT_API_BASE / SUMMARIZER_API_BASE
    # are explicit kwargs on their respective clients, so no env hijacking is
    # needed here.

    eval_dir = args.eval_dir or (args.input_dir / "eval_bcp_gpt5")
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"input_dir : {args.input_dir}")
    print(f"eval_dir  : {eval_dir}")
    print(f"model     : {args.model or GRADER_MODEL}")
    print(f"workers   : {args.workers}")
    print()

    evaluate_browsecomp_plus(
        input_dir=str(args.input_dir),
        ground_truth=str(GT_PATH),
        eval_dir=str(eval_dir),
        model=args.model,
        qrel_evidence_path=str(QREL_PATH) if QREL_PATH.is_file() else None,
        force=args.force,
        max_workers=args.workers,
    )

    # ── Remap evaluation_summary.json -> gpt5_summary.json (em-summary shape) ──
    eval_summary_path = eval_dir / "evaluation_summary.json"
    if not eval_summary_path.is_file():
        raise SystemExit(f"pipeline did not produce {eval_summary_path}")
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))

    per_question: list[dict] = []
    for pq in eval_summary.get("per_query_metrics", []):
        qid = str(pq["query_id"])
        run_fp = args.input_dir / f"run_{qid}.json"
        run_data: dict = {}
        if run_fp.exists():
            try:
                run_data = json.loads(run_fp.read_text(encoding="utf-8"))
            except Exception:
                pass
        traj_status = run_data.get("status", "unknown")
        correct = bool(pq.get("correct", False))
        # Map per-traj status to gpt5_eval per_question vocabulary:
        #   complete + correct=True  → "success"
        #   complete + correct=False → "failed"
        #   max_iterations / max_context_length / crashed → pass through
        if traj_status == "complete":
            eval_status = "success" if correct else "failed"
        else:
            eval_status = traj_status
        per_question.append({
            "id": qid,
            "difficulty": -1,
            "correct": correct,
            "prediction": run_data.get("final_answer", ""),
            "gold": run_data.get("correct_answer", ""),
            "num_turns": pq.get("num_turns"),
            "elapsed_sec": pq.get("elapsed_sec"),
            "tool_call_counts": run_data.get("tool_call_counts", {}),
            "usage": run_data.get("usage", {}),
            "status": eval_status,
            "confidence": pq.get("confidence"),
        })

    n = len(per_question)
    correct = sum(1 for q in per_question if q["correct"])
    summary = {
        "input_dir": str(args.input_dir),
        "grader": "gpt-5",
        "judge_model": args.model or GRADER_MODEL,
        "total": {"n": n, "correct": correct,
                  "accuracy": (correct / n) if n else 0.0},
        "per_difficulty": {
            "unknown": {"n": n, "correct": correct,
                        "accuracy": (correct / n) if n else 0.0}
        },
        "n_missing_difficulty": n,
        "analysis": compute_analysis(per_question),
    }

    out = args.output or (args.input_dir / "gpt5_eval.json")
    out.write_text(json.dumps({"summary": summary, "per_question": per_question},
                              indent=2, ensure_ascii=False))

    print_summary(summary, f"GPT-5 GRADING — {args.input_dir.name}")
    print()
    print(f"Output:          {out}")


if __name__ == "__main__":
    main()
