"""Shared analysis used by EM (DR9K) and GPT-5 (BCP) graders.

Each grader computes per-question results, then calls `compute_analysis()` to
augment its `summary` dict with status counts, tool-call rates, turn / token /
wallclock distributions. `print_summary()` prints a clean human-readable
table to stdout.

Per-question dicts must carry: id, correct, status, num_turns, elapsed_sec,
tool_call_counts, usage. (DR9K rows also carry `difficulty`.)
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Iterable


def _pctile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(q * (len(xs) - 1))))
    return xs[k]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean":   round(statistics.mean(xs), 2),
        "median": round(statistics.median(xs), 2),
        "p95":    round(_pctile(xs, 0.95), 2),
        "max":    round(max(xs), 2),
    }


def compute_analysis(per_question: Iterable[dict]) -> dict:
    """Return analysis-stats dict to embed inside `summary`. See module doc."""
    pq = list(per_question)
    n = len(pq)

    status = Counter(q.get("status", "unknown") for q in pq)
    # gpt5_eval per_question.status vocabulary (strict — migrated by
    # scripts/migrate_status.py):
    #   success / failed / max_iterations / max_context_length / crashed
    # "incomplete" = everything except "success".
    _NON_SUCCESS = {"failed", "max_iterations", "max_context_length", "crashed"}
    incomplete = sum(v for k, v in status.items() if k in _NON_SUCCESS)

    turns = [q["num_turns"] for q in pq if q.get("num_turns") is not None]
    elapsed = [q["elapsed_sec"] for q in pq if q.get("elapsed_sec") is not None]

    # Tool calls — denominator is N=all runs (apples-to-apples across tools).
    tool_total: Counter = Counter()
    tool_used_in: Counter = Counter()
    for q in pq:
        for t, k in (q.get("tool_call_counts") or {}).items():
            tool_total[t] += k
            tool_used_in[t] += 1
    tool_calls = {
        t: {
            "total":     tool_total[t],
            "per_run":   round(tool_total[t] / n, 3) if n else 0.0,
            "runs_used": tool_used_in[t],
        }
        for t in tool_total
    }

    usage = [q.get("usage") or {} for q in pq if q.get("usage")]
    inp = [u.get("input_tokens", 0)  for u in usage]
    out = [u.get("output_tokens", 0) for u in usage]
    tot = [u.get("total_tokens", 0)  for u in usage]
    tokens = {
        "input_mean":  round(statistics.mean(inp), 1) if inp else 0.0,
        "output_mean": round(statistics.mean(out), 1) if out else 0.0,
        "total_mean":  round(statistics.mean(tot), 1) if tot else 0.0,
        "total_sum":   sum(tot),
    }

    return {
        "status":          dict(status),
        "incomplete_rate": round(incomplete / n, 4) if n else 0.0,
        "turns":           _stats(turns),
        "tool_calls":      tool_calls,
        "tokens":          tokens,
        "wallclock_sec":   _stats(elapsed),
    }


def print_summary(summary: dict, label: str) -> None:
    """Pretty-print the augmented summary to stdout."""
    n = summary["total"]["n"]
    print()
    print("=" * 60)
    print(f"{label}")
    print("=" * 60)

    acc = summary["total"]["accuracy"]
    print(f"Accuracy:        {summary['total']['correct']}/{n} = {acc:.2%}")

    pd = summary.get("per_difficulty") or {}
    real_levels = [k for k in pd if k != "unknown"]
    if real_levels:
        for level in sorted(real_levels):
            s = pd[level]
            print(f"  {level}:           {s['correct']}/{s['n']} = {s['accuracy']:.2%}")

    a = summary.get("analysis") or {}
    st = a.get("status") or {}
    success         = st.get("success", 0)
    failed          = st.get("failed", 0)
    max_iter        = st.get("max_iterations", 0)
    max_ctx         = st.get("max_context_length", 0)
    crashed         = st.get("crashed", 0)
    print()
    print(f"Status:          success={success} ({success/n:.1%})  "
          f"failed={failed}  max_iter={max_iter}  max_ctx={max_ctx}  crashed={crashed}")
    print(f"Incomplete rate: {a.get('incomplete_rate', 0):.2%}")

    tu = a.get("turns") or {}
    print()
    print(f"Turns:           mean={tu.get('mean', 0):.1f}  "
          f"median={tu.get('median', 0):.1f}  "
          f"p95={tu.get('p95', 0):.1f}  max={tu.get('max', 0):.1f}")
    wc = a.get("wallclock_sec") or {}
    print(f"Wallclock:       mean={wc.get('mean', 0):.1f}s  "
          f"median={wc.get('median', 0):.1f}s  "
          f"p95={wc.get('p95', 0):.1f}s")

    tc = a.get("tool_calls") or {}
    if tc:
        print()
        print(f"Tool calls (per run, over all {n} runs):")
        # sort by per_run descending
        for t, v in sorted(tc.items(), key=lambda kv: -kv[1]["per_run"]):
            print(f"  {t:<16} {v['per_run']:>6.3f}  (used in {v['runs_used']:>4}/{n})")
