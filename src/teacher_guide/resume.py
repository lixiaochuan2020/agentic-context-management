"""CLI for resuming student rollouts from teacher annotations.

For each `mc`-decision annotation:
  1. load original ReAct initial trajectory
  2. splice: keep `history[:after_id+1]`, swap system prompt to 4-tool,
     append synthetic asst turn (teacher think + manage_context call)
  3. call `src.runner.run` with the spliced initial_history; runner executes
     the mc on turn 0 (summarizer produces the compression), then resumes the
     student rollout with 4 tools available

This is an exploratory validation step — measures how many wrong-initial trajs
become correct after teacher's mc intervention. Output trajectories can later
be graded with `scripts/grade_bcp_gpt5.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from src.client import LiteLLMClient
from src.configs import RunConfig, load_config
from src.runner import run as runner_run
from src.teacher_guide.splice import splice_annotation_to_initial_history


logger = logging.getLogger("teacher_guide.resume")


def _load_annotation(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("skip %s: %s", p.name, e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations_dir", required=True, type=Path,
                    help="Dir of run_<qid>.json teacher annotations.")
    ap.add_argument("--rollouts_dir", required=True, type=Path,
                    help="ReAct initial rollout dir containing run_<qid>.json.")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Where to write resumed-rollout trajectories.")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="Project YAML for runner config (default: configs/default.yaml).")
    ap.add_argument("--override", nargs="*", default=[],
                    help="OmegaConf dotlist overrides, e.g. retrieval.k=20.")
    ap.add_argument("--student_model", default=None,
                    help="agent.model (default from configs/default.yaml).")
    ap.add_argument("--agent_api_bases", default=None,
                    help="Comma-separated list of agent endpoints to round-robin across "
                         "(e.g. http://localhost:8003/v1,http://localhost:8004/v1). "
                         "Falls back to $AGENT_API_BASE / default.yaml if omitted.")
    ap.add_argument("--summarizer_model", default=None)
    ap.add_argument("--summarizer_api_base", default=os.environ.get("SUMMARIZER_API_BASE"))
    ap.add_argument("--index_path", default=None)
    ap.add_argument("--qids", default="",
                    help="Comma-separated qid allowlist.")
    ap.add_argument("--limit", type=int, default=0, help="Only first N qids (0=all).")
    ap.add_argument("--max_workers", type=int, default=8,
                    help="Thread-pool size; reuses agent endpoints round-robin.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip qids whose resumed trajectory already exists.")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.annotations_dir.is_dir():
        ap.error(f"--annotations_dir not found: {args.annotations_dir}")
    if not args.rollouts_dir.is_dir():
        ap.error(f"--rollouts_dir not found: {args.rollouts_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "workspace").mkdir(exist_ok=True)

    # ── Resolve runner config ─────────────────────────────
    yaml_path = args.config if args.config and Path(args.config).is_file() else None
    overrides: list[str] = list(args.override)
    if args.student_model:       overrides.append(f"agent.model={args.student_model}")
    if args.index_path:          overrides.append(f"retrieval.index_path={args.index_path}")
    if args.summarizer_model:    overrides.append(f"summarizer.model={args.summarizer_model}")
    if args.summarizer_api_base: overrides.append(f"summarizer.api_base={args.summarizer_api_base}")
    # Ensure 4-tool mode in case the YAML defaults differ.
    overrides.append("runtime.use_memory_tools=true")
    base_config = load_config(yaml_path=yaml_path, cli_overrides=overrides)

    api_bases = [s.strip() for s in (args.agent_api_bases or base_config.agent.api_base).split(",") if s.strip()]
    if not api_bases:
        ap.error("no agent api_base resolved (set --agent_api_bases or AGENT_API_BASE)")
    logger.info("agent api_bases: %s", api_bases)
    logger.info("student model:   %s", base_config.agent.model)

    # ── Collect work ──────────────────────────────────────
    pairs: list[tuple[str, dict, dict]] = []
    for p in sorted(args.annotations_dir.glob("run_*.json")):
        ann = _load_annotation(p)
        if ann is None or ann.get("decision") != "mc":
            continue
        qid = str(ann.get("query_id") or p.stem.removeprefix("run_"))
        traj_p = args.rollouts_dir / f"run_{qid}.json"
        if not traj_p.is_file():
            logger.warning("[%s] no source rollout — skip", qid)
            continue
        try:
            traj = json.loads(traj_p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("[%s] broken source rollout: %s", qid, e)
            continue
        pairs.append((qid, traj, ann))

    if args.qids:
        keep = {q.strip() for q in args.qids.split(",") if q.strip()}
        pairs = [t for t in pairs if t[0] in keep]
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if args.skip_existing:
        done = {p.stem.removeprefix("run_") for p in args.out_dir.glob("run_*.json")}
        before = len(pairs)
        pairs = [t for t in pairs if t[0] not in done]
        logger.info("skip_existing: dropped %d already-done", before - len(pairs))

    if not pairs:
        logger.info("nothing to do — exit.")
        return

    logger.info("resuming %d qids across %d endpoint(s) with %d workers",
                len(pairs), len(api_bases), args.max_workers)

    tls = threading.local()
    n_calls = {"n": 0, "lock": threading.Lock()}

    def _get_client(api_base: str) -> LiteLLMClient:
        cache = getattr(tls, "clients", None) or {}
        if api_base not in cache:
            cache[api_base] = LiteLLMClient(model=base_config.agent.model, api_base=api_base)
            tls.clients = cache
        return cache[api_base]

    def _do(item):
        qid, traj, ann = item
        with n_calls["lock"]:
            n_calls["n"] += 1
            idx = n_calls["n"]
        api_base = api_bases[(idx - 1) % len(api_bases)]
        out_path = args.out_dir / f"run_{qid}.json"
        cache_path = args.out_dir / f"run_{qid}.partial.json"

        spliced = splice_annotation_to_initial_history(
            traj=traj,
            annotation=ann,
            benchmark=base_config.runtime.benchmark,
            context_window=base_config.agent.context_window,
        )
        if spliced is None:
            logger.warning("[%s] splice failed (decision=%s, after_id=%s)", qid, ann.get("decision"), ann.get("after_id"))
            return (qid, "splice_failed", 0.0, None)

        workspace = args.out_dir / "workspace" / qid
        workspace.mkdir(parents=True, exist_ok=True)
        runtime = replace(
            base_config.runtime,
            use_memory_tools=True,
            workspace_root=str(workspace),
            compress_initial_history=False,
        )
        config = replace(
            base_config,
            runtime=runtime,
            question_id=qid,
            correct_answer=spliced["correct_answer"],
            cache_path=str(cache_path),
            initial_history=spliced["initial_history"],
            initial_raw_history=spliced["initial_raw_history"],
            initial_last_boundary_pos=spliced["initial_last_boundary_pos"],
            initial_summary_id=spliced["initial_summary_id"],
        )

        client = _get_client(api_base)
        t0 = time.time()
        try:
            result = runner_run(client, spliced["question"], config)
            err = None
        except Exception as e:
            logger.exception("[%s] crashed: %s", qid, e)
            result = {
                "query_id": qid, "question": spliced["question"],
                "correct_answer": spliced["correct_answer"],
                "status": "crash", "error": f"{type(e).__name__}: {e}",
                "final_answer": "", "num_turns": 0,
                "history": spliced["initial_history"],
                "raw_history": spliced["initial_raw_history"],
            }
            err = str(e)

        elapsed = round(time.time() - t0, 1)
        result["elapsed_sec"] = elapsed
        result["teacher_intervention"] = spliced["teacher_annotation"]
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if cache_path.exists():
            try: cache_path.unlink()
            except OSError: pass
        return (qid, result.get("status"), elapsed, result.get("final_answer", "")[:50] if result.get("final_answer") else "")

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_do, t): t[0] for t in pairs}
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                logger.exception("[%s] worker crashed: %s", qid, e)
                n_fail += 1
                continue
            qid_r, status, elapsed, ans = r
            if status == "complete":
                n_ok += 1
            else:
                n_fail += 1
            print(f"{qid_r}\tstatus={status}\telapsed={elapsed}s\tans={ans!r}", flush=True)

    logger.info("done — success=%d  other=%d  (out=%s)", n_ok, n_fail, args.out_dir)


if __name__ == "__main__":
    main()
