"""CLI for the teacher-guided annotation step.

Walks a ReAct initial rollout dir, loads grade results (optional), picks every
trajectory that did NOT reach a correct answer (wrong-with-answer OR
no-final-answer cases), calls the teacher for one annotation per qid, and
writes the results into `src/teacher_guide/results/<tag>/annotations/`.

Step 0 (running the initial rollout) and Step 2+ (preprocess → SFT) are separate.

Example
-------
    python -m src.teacher_guide.run \\
        --run_dir results/browsecomp-plus/qwen3.5-9b-base/rollout-teacher_guide-init/run_all \\
        --eval_dir results/browsecomp-plus/qwen3.5-9b-base/grade_bcp/rollout-teacher_guide-init/run_all \\
        --out_dir  src/teacher_guide/results/teacher_guide/annotations \\
        --teacher_model gpt-5 \\
        --max_workers 4 --limit 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from src.client import LiteLLMClient
from src.teacher_guide.annotate import annotate_one, save_annotation


logger = logging.getLogger("teacher_guide.run")


def load_gold_docs_map(jsonl_path: Path | str | None) -> dict[str, list[dict]]:
    if not jsonl_path:
        return {}
    p = Path(jsonl_path)
    if not p.is_file():
        logger.warning("gold_docs jsonl not found: %s", p)
        return {}
    out: dict[str, list[dict]] = {}
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(r.get("query_id", ""))
            if qid:
                out[qid] = r.get("gold_docs") or []
    logger.info("loaded gold_docs for %d queries from %s", len(out), p)
    return out


def load_wrong_trajectories(
    run_dir: Path,
    eval_dir: Path | None,
) -> list[dict]:
    """Return trajectory dicts that did NOT end up correct.

    Includes:
      - status in {"success", "completed"} with judge_result.correct=False
        (wrong-with-answer cases — requires eval_dir)
      - status NOT in {"success", "completed"} (max_context_length /
        max_iterations / complete_no_answer / crash / legacy values — no
        final answer was produced; eval_dir not needed)
    """
    wrong: list[dict] = []
    for traj_file in sorted(run_dir.glob("run_*.json")):
        if traj_file.name.endswith(".partial.json"):
            continue
        try:
            traj = json.loads(traj_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("broken JSON in %s: %s", traj_file, e)
            continue

        qid = str(traj.get("query_id") or traj_file.stem.removeprefix("run_"))
        status = traj.get("status")

        is_correct = False
        if status == "complete" and eval_dir is not None:
            eval_file = Path(eval_dir) / f"run_{qid}_eval.json"
            if eval_file.is_file():
                try:
                    ed = json.loads(eval_file.read_text(encoding="utf-8"))
                    if (ed.get("judge_result") or {}).get("correct"):
                        is_correct = True
                except json.JSONDecodeError:
                    pass

        if is_correct:
            continue

        traj["_qid"] = qid
        wrong.append(traj)

    logger.info("found %d wrong / no-final trajectories in %s", len(wrong), run_dir)
    return wrong


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, type=Path,
                    help="ReAct initial rollout dir containing run_<qid>.json files.")
    ap.add_argument("--eval_dir", default=None, type=Path,
                    help="Grade output dir (run_<qid>_eval.json). Optional — without it, only "
                         "no-final-answer cases are picked up; completed runs are treated as correct.")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Where to write annotation JSON (one per qid).")
    ap.add_argument("--teacher_model", default="gpt-5",
                    help="Teacher model name passed to litellm (default: gpt-5).")
    ap.add_argument("--teacher_api_base", default="https://api.openai.com/v1")
    ap.add_argument("--gold_docs_path",
                    default="data/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
                    help="JSONL with gold_docs per query_id. Pass '' to disable.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N (0=all).")
    ap.add_argument("--qids", default="",
                    help="Comma-separated qid allowlist (overrides --limit ordering).")
    ap.add_argument("--max_workers", type=int, default=4,
                    help="Thread-pool size for parallel teacher calls (per-thread client).")
    ap.add_argument("--max_retries", type=int, default=2,
                    help="Per-qid retry budget on validation failure.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip qids whose annotation file already exists.")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.run_dir.is_dir():
        ap.error(f"--run_dir does not exist: {args.run_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold_docs_map = load_gold_docs_map(args.gold_docs_path or None)
    wrong = load_wrong_trajectories(args.run_dir, args.eval_dir)

    if args.qids:
        keep = {q.strip() for q in args.qids.split(",") if q.strip()}
        before = len(wrong)
        wrong = [t for t in wrong if t["_qid"] in keep]
        logger.info("--qids filter: %d → %d trajectories", before, len(wrong))
    if args.limit and args.limit > 0:
        wrong = wrong[: args.limit]
        logger.info("--limit: keeping first %d trajectories", len(wrong))

    if args.skip_existing:
        existing_ok = {p.stem.removeprefix("run_") for p in args.out_dir.glob("run_*.json")}
        before = len(wrong)
        wrong = [t for t in wrong if t["_qid"] not in existing_ok]
        logger.info("--skip_existing: dropped %d already-annotated qids", before - len(wrong))

    if not wrong:
        logger.info("nothing to annotate — exit.")
        return

    # LiteLLMClient holds mutable per-call state (_last_*); use a per-thread instance.
    tls = threading.local()

    def _get_client() -> LiteLLMClient:
        c = getattr(tls, "client", None)
        if c is None:
            c = LiteLLMClient(model=args.teacher_model, api_base=args.teacher_api_base)
            tls.client = c
        return c

    def _do(traj: dict) -> str:
        qid = traj["_qid"]
        try:
            out = annotate_one(
                client=_get_client(),
                qid=qid,
                traj=traj,
                gold_docs=gold_docs_map.get(qid),
                summary_ids=[],  # iter1: workspace empty (design decision 10)
                max_retries=args.max_retries,
            )
        except Exception as e:
            logger.exception("[%s] annotate_one crashed: %s", qid, e)
            out = {
                "query_id": qid,
                "validation": {
                    "ok": False,
                    "attempts": 0,
                    "errors": [f"crash: {type(e).__name__}: {e}"],
                    "log": [],
                },
                "decision": None,
                "teacher_raw_last": None,
            }
        path = save_annotation(out, args.out_dir)
        return f"{qid}\tdecision={out.get('decision')}\tafter_id={out.get('after_id')}\t→ {path.name}"

    logger.info("annotating %d trajectories with %d workers → %s",
                len(wrong), args.max_workers, args.out_dir)
    n_ok = n_invalid = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_do, t): t["_qid"] for t in wrong}
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                line = fut.result()
            except Exception as e:
                logger.exception("[%s] worker crashed: %s", qid, e)
                n_invalid += 1
                continue
            if "→ invalid_" in line:
                n_invalid += 1
            else:
                n_ok += 1
            print(line, flush=True)

    logger.info("done — %d valid, %d invalid (out=%s)", n_ok, n_invalid, args.out_dir)


if __name__ == "__main__":
    main()
