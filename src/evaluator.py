"""
Evaluation framework: LLM-as-judge grading + BCP evaluation pipeline + pilot metrics.

- grade_answer(): single answer grading (GPT-5 LLM-as-judge)
- grade_answer_responses_api(): grading via OpenAI Responses API
- aggregate_results(): accuracy aggregation
- compute_pilot_metrics(): ACM pilot study metrics
- evaluate_browsecomp_plus(): full BCP evaluation pipeline
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import statistics
import textwrap
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openai
from tqdm import tqdm

from src.config import GRADER_API_KEY, GRADER_MODEL

logger = logging.getLogger(__name__)


# ── EM grader (SQuAD-style normalization) ────────────────

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_RE = re.compile(r"[ -⁯⸀-⹿\\'!\"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~]")
_WHITESPACE_RE = re.compile(r"\s+")


def em_normalize(text: str) -> str:
    """SQuAD-style answer normalization: lowercase, strip punct/articles, collapse whitespace."""
    if text is None:
        return ""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _ARTICLES_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def em_grade(prediction: str, gold: str) -> bool:
    """Exact-match grade after SQuAD-style normalization."""
    return em_normalize(prediction) == em_normalize(gold)


# ── 官方 simple-evals GRADER_TEMPLATE ────────────────────

GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


# ── 单条评估 ──────────────────────────────────────────────

def grade_answer(
    question: str,
    correct_answer: str,
    response: str,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """
    使用官方 simple-evals GRADER_TEMPLATE 评估答案。

    Args:
        question: 原始问题
        correct_answer: ground truth 答案
        response: 模型的完整回复文本（含 Explanation / <answer> / Confidence）
        api_key: OpenAI API key（默认从 config 读取）
        model: grader 模型名（默认从 config 读取）

    Returns:
        {
            "correct": bool,
            "extracted_answer": str,
            "reasoning": str,
            "confidence": int,
            "raw_grading": str,
        }
    """
    api_key = api_key or GRADER_API_KEY
    model = model or GRADER_MODEL

    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )

    client = openai.OpenAI(api_key=api_key)
    grader_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": grader_prompt}],
        max_completion_tokens=1024,
    )
    grading_text = grader_response.choices[0].message.content or ""

    # 提取 correct: yes/no
    match = re.search(r"correct:\s*(yes|no)", grading_text, re.IGNORECASE)
    is_correct = match.group(1).lower() == "yes" if match else False

    # 提取 extracted_final_answer
    answer_match = re.search(
        r"extracted_final_answer:\s*(.+?)(?:\n|$)", grading_text, re.IGNORECASE
    )
    extracted_answer = answer_match.group(1).strip() if answer_match else ""

    # 提取 reasoning
    reasoning_match = re.search(
        r"reasoning:\s*(.+?)(?=\ncorrect:|$)",
        grading_text,
        re.IGNORECASE | re.DOTALL,
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # 提取 confidence
    conf_match = re.search(r"confidence:\s*(\d+)", grading_text, re.IGNORECASE)
    confidence = int(conf_match.group(1)) if conf_match else 100

    logger.info(
        "  [grader] correct=%s, extracted='%s'",
        is_correct, extracted_answer[:80],
    )

    return {
        "correct": is_correct,
        "extracted_answer": extracted_answer,
        "reasoning": reasoning,
        "confidence": confidence,
        "raw_grading": grading_text,
    }


# ── BCP 官方评估（OpenAI Responses API）─────────────────

def _parse_judge_response(judge_text: str) -> dict:
    """Parse judge model output, handling bold markdown variants from GPT-5.

    Adapted from BCP evaluate_with_openai.py to handle **field:** and
    **field**: formats in addition to plain field: format.
    """
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not judge_text:
        result["parse_error"] = True
        return result

    # extracted_final_answer (bold variants first, then plain)
    for pat in [
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ]:
        m = re.search(pat, judge_text, re.IGNORECASE | re.DOTALL)
        if m:
            result["extracted_final_answer"] = m.group(1).strip()
            break

    # reasoning
    for pat in [
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
    ]:
        m = re.search(pat, judge_text, re.IGNORECASE | re.DOTALL)
        if m:
            result["reasoning"] = m.group(1).strip()
            break

    # correct (yes/no)
    for pat in [
        r"\*\*correct:\*\*\s*(yes|no)",
        r"\*\*correct\*\*:\s*(yes|no)",
        r"correct:\s*(yes|no)",
    ]:
        m = re.search(pat, judge_text, re.IGNORECASE)
        if m:
            result["correct"] = m.group(1).lower() == "yes"
            break

    # confidence
    for pat in [
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
    ]:
        m = re.search(pat, judge_text, re.IGNORECASE)
        if m:
            result["confidence"] = min(float(m.group(1)), 100)
            break

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def grade_answer_responses_api(
    question: str,
    correct_answer: str,
    response: str,
    api_key: str | None = None,
    model: str | None = None,
    max_output_tokens: int = 1024,
) -> dict:
    """Grade using OpenAI Responses API (BCP official approach for GPT-5).

    Returns:
        {
            "correct": bool,
            "extracted_answer": str,
            "reasoning": str,
            "confidence": float,
            "raw_grading": str,
            "parse_error": bool,
        }
    """
    api_key = api_key or GRADER_API_KEY
    model = model or GRADER_MODEL

    prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )

    client = openai.OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    judge_text = resp.output_text if hasattr(resp, "output_text") else ""

    parsed = _parse_judge_response(judge_text)

    logger.info(
        "  [grader-responses] correct=%s, extracted='%s'",
        parsed["correct"],
        (parsed["extracted_final_answer"] or "")[:80],
    )

    return {
        "correct": parsed["correct"] or False,
        "extracted_answer": parsed["extracted_final_answer"] or "",
        "reasoning": parsed["reasoning"] or "",
        "confidence": parsed["confidence"] if parsed["confidence"] is not None else 100,
        "raw_grading": judge_text,
        "parse_error": parsed["parse_error"],
    }


# ── 聚合指标（官方 simple-evals 逻辑）────────────────────

def aggregate_results(results: list[dict]) -> dict:
    """
    聚合评估结果，与官方 simple-evals 计算逻辑一致:
        accuracy = sum(is_correct) / N

    Args:
        results: grade_answer() 返回值的列表

    Returns:
        {
            "accuracy": float,
            "is_correct": float,      # 官方别名
            "is_incorrect": float,     # 官方指标
            "num_samples": int,
        }
    """
    n = len(results)
    if n == 0:
        return {
            "accuracy": 0.0,
            "is_correct": 0.0,
            "is_incorrect": 0.0,
            "num_samples": 0,
        }

    is_correct_list = [1 if r["correct"] else 0 for r in results]
    is_incorrect_list = [1 if not r["correct"] else 0 for r in results]

    accuracy = sum(is_correct_list) / n

    return {
        "accuracy": accuracy,
        "is_correct": accuracy,                        # 官方别名
        "is_incorrect": sum(is_incorrect_list) / n,    # 官方指标
        "num_samples": n,
    }


# ── Pilot Study 额外指标 ─────────────────────────────────

def compute_pilot_metrics(result: dict) -> dict:
    """
    从单条实验结果计算 Pilot Study 额外指标。

    Args:
        result: run() 的返回值（unified runner output）

    Returns:
        {
            "token_peak": int,
            "token_trajectory": list[int] | list[dict],
            "num_summary_calls": int,
            "num_discard_calls": int,
            "num_manage_context_calls": int,
            "token_reduction_per_tool": list[dict],
        }
    """
    raw_trajectory = result.get("token_trajectory", [])
    tool_calls = result.get("tool_calls", [])

    # Compat: new format list[dict], legacy format list[int]
    if raw_trajectory and isinstance(raw_trajectory[0], dict):
        actual_values = [t["tokens_actual"] for t in raw_trajectory]
    else:
        actual_values = raw_trajectory

    token_peak = max(actual_values) if actual_values else 0

    # Cumulative raw tokens (only in memtool dict format)
    tokens_cumulative_raw = 0
    if raw_trajectory and isinstance(raw_trajectory[0], dict):
        last = raw_trajectory[-1]
        tokens_cumulative_raw = last.get("tokens_cumulative_raw", actual_values[-1] if actual_values else 0)
    elif actual_values:
        tokens_cumulative_raw = actual_values[-1]

    # Accumulated output tokens
    output_tokens_cumulative = 0
    if raw_trajectory and isinstance(raw_trajectory[0], dict):
        output_tokens_cumulative = raw_trajectory[-1].get("output_tokens_cumulative", 0)

    # tool call 统计
    num_summary = sum(1 for tc in tool_calls if tc.get("tool") == "summary")
    num_discard = sum(1 for tc in tool_calls if tc.get("tool") == "discard")
    num_manage_context = sum(1 for tc in tool_calls if tc.get("tool") == "manage_context")

    # 每次 tool call 前后的 token 变化
    token_reductions = []
    for tc in tool_calls:
        turn_idx = tc["turn"]
        tokens_after = tc.get("tokens_after", 0)
        tokens_before = actual_values[turn_idx - 1] if turn_idx > 0 and turn_idx - 1 < len(actual_values) else tokens_after
        token_reductions.append({
            "turn": turn_idx,
            "tool": tc["tool"],
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "delta": tokens_after - tokens_before,
        })

    return {
        "token_peak": token_peak,
        "tokens_cumulative_raw": tokens_cumulative_raw,
        "output_tokens_cumulative": output_tokens_cumulative,
        "token_trajectory": raw_trajectory,
        "num_summary_calls": num_summary,
        "num_discard_calls": num_discard,
        "num_manage_context_calls": num_manage_context,
        "token_reduction_per_tool": token_reductions,
    }


# ══════════════════════════════════════════════════════════
# BrowseComp-Plus evaluation pipeline
# ══════════════════════════════════════════════════════════

# ── BCP data helpers ─────────────────────────────────────

def load_ground_truth(jsonl_path: Path) -> dict[str, dict[str, str]]:
    """Load BCP ground truth JSONL -> {query_id: {question, answer}}."""
    gt: dict[str, dict[str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gt[str(obj["query_id"])] = {
                "question": obj["query"],
                "answer": obj["answer"],
            }
    return gt


def load_qrel_data(qrel_path: Path) -> dict[str, list[str]]:
    """Load qrel_evidence.txt (TREC format) -> {query_id: [docid, ...]}."""
    qrel_data: dict[str, list[str]] = defaultdict(list)
    if not qrel_path.exists():
        return dict(qrel_data)
    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) == 4, f"Expected 4 parts in qrel line: {line}"
            qrel_data[parts[0]].append(parts[2])
    return dict(qrel_data)


# ── Citation extraction ──────────────────────────────────

def extract_citations_from_response(response_text: str) -> list[str]:
    """Extract [docid] and fullwidth brackets citations from response text."""
    if not response_text:
        return []

    single_matches = re.findall(r"\[(\d+)\]", response_text)
    multi_matches = re.findall(r"\[([^\[\]]*?)\]", response_text)
    single_fw = re.findall(r"\u3010(\d+)\u3011", response_text)
    multi_fw = re.findall(r"\u3010([^\u3010\u3011]*?)\u3011", response_text)

    all_docids: set[str] = set()
    all_docids.update(single_matches)
    all_docids.update(single_fw)

    for match in multi_matches:
        if match in single_matches:
            continue
        all_docids.update(re.findall(r"\d+", match))

    for match in multi_fw:
        if match in single_fw:
            continue
        all_docids.update(re.findall(r"\d+", match))

    return list(all_docids)


def compute_citation_metrics(
    cited_docids: list[str], relevant_docids: list[str],
) -> dict[str, float]:
    """Citation precision/recall against qrel relevant docs."""
    metrics = {
        "num_citations": len(cited_docids),
        "num_relevant": len(relevant_docids),
        "precision": 0.0,
        "recall": 0.0,
    }
    if not cited_docids:
        return metrics

    cited_set = set(cited_docids)
    relevant_set = set(relevant_docids)
    overlap = cited_set & relevant_set

    if cited_docids:
        metrics["precision"] = len(overlap) / len(cited_docids)
    if relevant_docids:
        metrics["recall"] = len(overlap) / len(relevant_docids)

    return metrics


# ── Calibration error ────────────────────────────────────

def _calib_err(confidence, correct, p="2", beta=100):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_conf = confidence[bins[i][0]: bins[i][1]]
        bin_correct = correct[bins[i][0]: bins[i][1]]
        n_in_bin = len(bin_conf)
        if n_in_bin > 0:
            diff = np.abs(np.nanmean(bin_conf) - np.nanmean(bin_correct))
            if p == "2":
                cerr += n_in_bin / total_examples * np.square(diff)
            elif p == "1":
                cerr += n_in_bin / total_examples * diff
            elif p in ("infty", "infinity", "max"):
                cerr = np.maximum(cerr, diff)

    if p == "2":
        cerr = np.sqrt(cerr)
    return cerr


def calculate_calibration_error(
    confidences: list[float], correctness: list[bool], beta: int = 100,
) -> float:
    """Compute Expected Calibration Error (%). Requires len >= beta."""
    assert len(confidences) == len(correctness) > 0
    conf_arr = np.array(confidences) / 100.0
    corr_arr = np.array(correctness, dtype=float)
    return float(_calib_err(conf_arr, corr_arr, p="2", beta=beta) * 100)


# ── Output helpers ───────────────────────────────────────

def _save_detailed_csv(all_results: list[dict], output_dir: Path):
    """Save per-query detailed CSV (BCP official format + ACM columns)."""
    csv_path = output_dir / "detailed_judge_results.csv"
    fieldnames = [
        "query_id", "predicted_answer", "correct_answer",
        "judge_correct", "confidence", "is_completed", "parse_error",
        "json_path", "num_citations", "precision_positives", "recall_positives",
        "token_peak", "num_manage_context_calls", "num_turns", "elapsed_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            jr = r.get("judge_result", {})
            cit_raw = r.get("citations")
            cit = cit_raw if isinstance(cit_raw, dict) else {}
            metrics = cit.get("metrics") or cit.get("metrics_positives") or {}
            pred = jr.get("extracted_final_answer", "")
            if not pred:
                resp = r.get("response", "")
                pred = (resp[:200] + "...") if len(resp) > 200 else resp
            acm = r.get("acm_metrics", {})
            writer.writerow({
                "query_id": r.get("query_id", ""),
                "predicted_answer": pred,
                "correct_answer": r.get("correct_answer", ""),
                "judge_correct": jr.get("correct", ""),
                "confidence": jr.get("confidence", ""),
                "is_completed": r.get("is_completed", ""),
                "parse_error": jr.get("parse_error", False),
                "json_path": r.get("json_path", ""),
                "num_citations": len(cit.get("cited_docids", [])),
                "precision_positives": metrics.get("precision", 0),
                "recall_positives": metrics.get("recall", 0),
                "token_peak": acm.get("token_peak", ""),
                "num_manage_context_calls": acm.get("num_manage_context_calls", ""),
                "num_turns": r.get("num_turns", ""),
                "elapsed_sec": r.get("elapsed_sec", ""),
            })
    logger.info("Detailed CSV saved to %s", csv_path)


# ── Main BCP evaluation function ─────────────────────────

def _evaluate_one_bcp_file(
    json_path: Path,
    gt: dict,
    qrel_evidence: dict,
    output_dir: Path,
    model: str,
    api_key: str,
    max_output_tokens: int,
    force: bool,
) -> dict | None:
    """Grade a single run_*.json file and write its <stem>_eval.json.

    Returns the eval_result dict, or None if the file should be skipped
    (no ground truth or load error).
    """
    eval_path = output_dir / f"{json_path.stem}_eval.json"

    if eval_path.exists() and not force:
        try:
            with eval_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # fall through and re-evaluate

    try:
        with json_path.open("r", encoding="utf-8") as f:
            run_data = json.load(f)
    except Exception as e:
        logger.error("Error loading %s: %s", json_path, e)
        return None

    query_id = run_data.get("query_id")
    if not query_id or str(query_id) not in gt:
        logger.warning("No ground truth for query_id %s in %s", query_id, json_path)
        return None

    correct_answer = gt[str(query_id)]["answer"]
    gt_question = gt[str(query_id)]["question"]
    # Only "complete" trajectories (runner finished cleanly — no max_iter, no
    # context overflow, no crash) get judged for correctness. A missing
    # final_answer is still possible within "complete" and is graded as failed.
    is_completed = run_data.get("status") == "complete"

    retrieved_set = set(run_data.get("retrieved_docids", []))
    positives = qrel_evidence.get(str(query_id), [])
    retrieval_recall = (
        len(retrieved_set & set(positives)) / len(positives)
        if positives else 0.0
    )

    response = ""
    result_arr = run_data.get("result", [])
    if result_arr and result_arr[-1].get("type") == "output_text":
        response = result_arr[-1].get("output", "")

    acm_metrics = compute_pilot_metrics(run_data)
    run_status = run_data.get("status", "unknown")

    if not response or not is_completed:
        eval_result = {
            "json_path": str(json_path),
            "query_id": query_id,
            "response": response,
            "correct_answer": correct_answer,
            "is_completed": is_completed,
            "status": run_status,
            "judge_prompt": None,
            "judge_response": None,
            "judge_result": {
                "parse_error": True,
                "correct": False,
                "error": "Response incomplete or cannot be parsed",
            },
            "tool_call_counts": run_data.get("tool_call_counts", {}),
            "citations": None,
            "retrieval": {
                "recall": retrieval_recall,
                "retrieved_docids": sorted(retrieved_set),
            },
            "model_info": {"judge_model": model, "max_output_tokens": max_output_tokens},
            "acm_metrics": {
                "token_peak": acm_metrics["token_peak"],
                "tokens_cumulative_raw": acm_metrics["tokens_cumulative_raw"],
                "output_tokens_cumulative": acm_metrics["output_tokens_cumulative"],
                "num_summary_calls": acm_metrics["num_summary_calls"],
                "num_discard_calls": acm_metrics["num_discard_calls"],
                "num_manage_context_calls": acm_metrics["num_manage_context_calls"],
            },
            "num_turns": run_data.get("num_turns", 0),
            "elapsed_sec": run_data.get("elapsed_sec", 0),
        }
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
        return eval_result

    try:
        grade = grade_answer_responses_api(
            question=gt_question,
            correct_answer=correct_answer,
            response=response,
            api_key=api_key,
            model=model,
            max_output_tokens=max_output_tokens,
        )
    except Exception as e:
        logger.error("Error grading %s: %s", json_path, e)
        return None

    cited_docids = extract_citations_from_response(response)
    citation_metrics = compute_citation_metrics(cited_docids, positives)

    eval_result = {
        "json_path": str(json_path),
        "query_id": query_id,
        "question": gt_question,
        "response": response,
        "correct_answer": correct_answer,
        "is_completed": is_completed,
        "status": run_status,
        "judge_prompt": None,
        "judge_response": grade["raw_grading"],
        "judge_result": {
            "extracted_final_answer": grade["extracted_answer"],
            "reasoning": grade["reasoning"],
            "correct": grade["correct"],
            "confidence": grade["confidence"],
            "parse_error": grade.get("parse_error", False),
        },
        "tool_call_counts": run_data.get("tool_call_counts", {}),
        "citations": {
            "cited_docids": cited_docids,
            "metrics": citation_metrics,
        },
        "retrieval": {
            "retrieved_docids": sorted(retrieved_set),
            "recall": retrieval_recall,
        },
        "model_info": {"judge_model": model, "max_output_tokens": max_output_tokens},
        "acm_metrics": {
            "token_peak": acm_metrics["token_peak"],
            "tokens_cumulative_raw": acm_metrics["tokens_cumulative_raw"],
            "output_tokens_cumulative": acm_metrics["output_tokens_cumulative"],
            "num_summary_calls": acm_metrics["num_summary_calls"],
            "num_discard_calls": acm_metrics["num_discard_calls"],
            "num_manage_context_calls": acm_metrics["num_manage_context_calls"],
        },
        "num_turns": run_data.get("num_turns", 0),
        "elapsed_sec": run_data.get("elapsed_sec", 0),
    }
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    return eval_result


def evaluate_browsecomp_plus(
    input_dir: str,
    ground_truth: str,
    eval_dir: str,
    model: str | None = None,
    qrel_evidence_path: str | None = None,
    force: bool = False,
    max_output_tokens: int = 1024,
    api_key: str | None = None,
    max_workers: int = 16,
):
    """Run BCP official evaluation + ACM memory metrics.

    Args:
        input_dir: Directory with run_*.json result files.
        ground_truth: Path to browsecomp_plus_decrypted.jsonl.
        eval_dir: Output directory for evaluation results.
        model: Judge model name (default from config, e.g. gpt-5).
        qrel_evidence_path: Path to qrel_evidence.txt for retrieval/citation metrics.
        force: Re-evaluate even if *_eval.json already exists.
        max_output_tokens: Max tokens for judge response.
        api_key: OpenAI API key (default from config).
    """
    model = model or GRADER_MODEL
    api_key = api_key or GRADER_API_KEY

    input_path = Path(input_dir)
    gt_path = Path(ground_truth)
    if not input_path.is_dir():
        raise ValueError(f"Input directory {input_path} does not exist")
    if not gt_path.is_file():
        raise ValueError(f"Ground truth file {gt_path} does not exist")

    logger.info("Loading ground truth from %s", gt_path)
    gt = load_ground_truth(gt_path)

    qrel_evidence: dict[str, list[str]] = {}
    if qrel_evidence_path:
        qp = Path(qrel_evidence_path)
        logger.info("Loading qrel evidence from %s", qp)
        qrel_evidence = load_qrel_data(qp)

    output_dir = Path(eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluations will be saved to %s", output_dir)

    json_files = [
        p for p in sorted(input_path.glob("*.json"))
        if not p.name.endswith("_eval.json")
        and not p.name.endswith(".partial.json")
        and p.name != "evaluation_summary.json"
    ]
    if not json_files:
        logger.warning("No JSON result files found in %s", input_path)
        return

    logger.info("Found %d result files to evaluate", len(json_files))

    all_results: list[dict] = []
    skipped = 0

    detected_model: Optional[str] = None
    try:
        with json_files[0].open("r", encoding="utf-8") as f:
            first = json.load(f)
        detected_model = (first.get("metadata") or {}).get("model")
    except Exception:
        pass

    skipped = sum(
        1 for jp in json_files
        if (output_dir / f"{jp.stem}_eval.json").exists() and not force
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_one_bcp_file,
                jp, gt, qrel_evidence, output_dir,
                model, api_key, max_output_tokens, force,
            ): jp
            for jp in json_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = fut.result()
            except Exception as e:
                jp = futures[fut]
                logger.error("Worker error on %s: %s", jp, e)
                continue
            if result is not None:
                all_results.append(result)

    logger.info("Processed %d evaluations (%d cached, workers=%d)",
                len(all_results), skipped, max_workers)
    if not all_results:
        logger.warning("No results to aggregate")
        return

    # ── Aggregate metrics ─────────────────────────────────

    total = len(all_results)

    correct_count = sum(
        1 for r in all_results if r.get("judge_result", {}).get("correct", False)
    )
    accuracy_pct = round(correct_count / total * 100, 2) if total else 0.0

    crashed_count = sum(1 for r in all_results if r.get("status") == "crashed")
    incomplete_count = sum(
        1 for r in all_results
        if not r.get("is_completed", True) and r.get("status") != "crashed"
    )

    all_tool_counts: dict[str, float] = defaultdict(float)
    for r in all_results:
        for tool_name, count in r.get("tool_call_counts", {}).items():
            all_tool_counts[tool_name] += count
    for k in all_tool_counts:
        all_tool_counts[k] = round(all_tool_counts[k] / total, 2)

    confidences: list[float] = []
    correctness: list[bool] = []
    for r in all_results:
        jr = r.get("judge_result", {})
        if not jr.get("parse_error", False) and jr.get("correct") is not None:
            conf = jr.get("confidence")
            if conf is not None:
                confidences.append(conf)
                correctness.append(jr["correct"])

    if confidences and len(confidences) >= 100:
        calibration_err = round(calculate_calibration_error(confidences, correctness), 2)
    else:
        logger.warning(
            "%d confidences — not enough for calibration error (need >= 100)",
            len(confidences),
        )
        calibration_err = None

    recalls = [
        r.get("retrieval", {}).get("recall", 0.0)
        for r in all_results
        if qrel_evidence.get(str(r.get("query_id")), [])
    ]
    recall_pct = round(float(np.mean(recalls)) * 100, 2) if recalls else None

    with_citations = [
        r for r in all_results
        if isinstance(r.get("citations"), dict) and r["citations"].get("cited_docids")
    ]
    n_with_cit = len(with_citations)
    citation_coverage = round(n_with_cit / total * 100, 2) if total else 0.0
    avg_cit = (
        round(sum(len(r["citations"]["cited_docids"]) for r in with_citations) / n_with_cit, 2)
        if n_with_cit else 0.0
    )
    cit_prec = (
        round(sum(
            (r["citations"].get("metrics") or {}).get("precision", 0) for r in with_citations
        ) / n_with_cit * 100, 2)
        if n_with_cit else 0.0
    )
    cit_recall = (
        round(sum(
            (r["citations"].get("metrics") or {}).get("recall", 0) for r in with_citations
        ) / n_with_cit * 100, 2)
        if n_with_cit else 0.0
    )

    acm_keys = [
        "token_peak", "tokens_cumulative_raw", "output_tokens_cumulative",
        "num_summary_calls", "num_discard_calls", "num_manage_context_calls",
    ]
    acm_avgs = {}
    for k in acm_keys:
        vals = [r.get("acm_metrics", {}).get(k, 0) for r in all_results]
        acm_avgs[f"avg_{k}"] = round(sum(vals) / len(vals), 2) if vals else 0

    per_query = []
    for r in all_results:
        qid = r.get("query_id")
        jr = r.get("judge_result", {})
        ret_recall = r.get("retrieval", {}).get("recall")
        acm = r.get("acm_metrics", {})
        per_query.append({
            "query_id": qid,
            "correct": bool(jr.get("correct", False)),
            "recall": round(ret_recall * 100, 2) if ret_recall is not None else None,
            "confidence": jr.get("confidence"),
            "token_peak": acm.get("token_peak", 0),
            "tokens_cumulative_raw": acm.get("tokens_cumulative_raw", 0),
            "num_manage_context_calls": acm.get("num_manage_context_calls", 0),
            "num_summary_calls": acm.get("num_summary_calls", 0),
            "num_discard_calls": acm.get("num_discard_calls", 0),
            "num_turns": r.get("num_turns", 0),
            "elapsed_sec": r.get("elapsed_sec", 0),
        })

    summary = {
        "LLM": detected_model or "unknown",
        "Total": total,
        "Correct": correct_count,
        "Crashed": crashed_count,
        "Incomplete": incomplete_count,
        "Accuracy (%)": accuracy_pct,
        "Recall (%)": recall_pct,
        "Calibration Error (%)": calibration_err,
        "avg_tool_stats": dict(all_tool_counts),
        "Citation Summary": {
            "coverage (%)": citation_coverage,
            "avg_citations_per_response": avg_cit,
            "precision (%)": cit_prec,
            "recall (%)": cit_recall,
        },
        **acm_avgs,
        "Evaluation Date": datetime.now().date().isoformat(),
        "per_query_metrics": per_query,
    }

    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=== BCP + ACM Evaluation Summary ===")
    logger.info("  Evaluated: %d queries", total)
    logger.info("  Correct: %d | Crashed: %d | Incomplete: %d", correct_count, crashed_count, incomplete_count)
    logger.info("  Accuracy: %.2f%%", accuracy_pct)
    logger.info("  Recall: %s", f"{recall_pct:.2f}%" if recall_pct is not None else "N/A")
    logger.info("  Calibration Error: %s", f"{calibration_err:.2f}%" if calibration_err is not None else "N/A")
    logger.info("  Avg tool stats: %s", dict(all_tool_counts))
    logger.info("  Citations: coverage=%.1f%%, precision=%.1f%%, recall=%.1f%%",
                citation_coverage, cit_prec, cit_recall)
    logger.info("  ACM: avg_token_peak=%.0f, avg_manage_context=%.2f, avg_summary=%.2f, avg_discard=%.2f",
                acm_avgs.get("avg_token_peak", 0), acm_avgs.get("avg_num_manage_context_calls", 0),
                acm_avgs.get("avg_num_summary_calls", 0), acm_avgs.get("avg_num_discard_calls", 0))
    logger.info("  Summary saved to %s", summary_path)

    _save_detailed_csv(all_results, output_dir)

    return summary


# ══════════════════════════════════════════════════════════
# DeepSearchQA grader
#
# Replicates the Kaggle starter notebook (DeepSearchQA Starter Code.ipynb,
# committed at repo root). The autorater is Gemini 2.5 Flash — using a
# different autorater model "will likely result in statistically
# significant deviation in results" (HF dataset card).
#
# Both prompt strings below are reproduced VERBATIM from the notebook.
# Do NOT edit unless the Kaggle source changes — keeping scores
# comparable with the public leaderboard depends on byte equality.
# ══════════════════════════════════════════════════════════

DSQA_AUTORATER_MODEL = "gemini/gemini-2.5-flash"  # litellm prefix → Google AI Studio
DSQA_LLM_MAX_RETRIES = 5


# ── Kaggle DEEPSEARCH_QA_PROMPT (verbatim) ──────────────
_DSQA_RATER_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")


# ── Kaggle GRADER_RATING_OUTPUT_EXAMPLE (verbatim, raw string) ──────────
# `{{` `}}` survive .format() as `{` `}`. `{prompt}/{prompt_type}/{answer}/{response}`
# are .format() slots.
_DSQA_RATER_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


def _build_dsqa_rater_input(prompt: str, prompt_type: str, answer: str, response: str) -> str:
    """Render the full autorater prompt (static instructions + formatted example)."""
    return _DSQA_RATER_PROMPT + _DSQA_RATER_OUTPUT_EXAMPLE.format(
        prompt=prompt,
        prompt_type=prompt_type,
        answer=answer,
        response=response,
    )


def _parse_dsqa_json_response(raw_text: str) -> Any:
    """Strip ```json``` fence (if present) then json.loads. Returns None on failure."""
    try:
        s = raw_text.strip()
        start_marker = "```json"
        start_idx = s.find(start_marker)
        if start_idx != -1:
            s = s[start_idx + len(start_marker):].strip()
            end_idx = s.rfind("```")
            if end_idx != -1:
                s = s[:end_idx].strip()
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.info("DSQA grader JSON parse failed: %s for: %s", e, raw_text[:200])
        return None


def _dsqa_extract_correctness_details(parsed: Any) -> dict[str, bool] | None:
    try:
        details = parsed["Answer Correctness"]["Correctness Details"]
        if (
            isinstance(details, dict)
            and all(isinstance(k, str) for k in details.keys())
            and all(isinstance(v, bool) for v in details.values())
        ):
            return details
        return None
    except (KeyError, TypeError):
        return None


def _dsqa_extract_excessive(parsed: Any) -> list[str] | None:
    """Returns list[str] on success, [] when key missing (valid empty), None on malformed."""
    try:
        excessive = parsed["Answer Correctness"]["Excessive Answers"]
        if isinstance(excessive, list) and all(isinstance(x, str) for x in excessive):
            return excessive
        return None
    except KeyError:
        return []
    except TypeError:
        return None


def _dsqa_call_autorater(prompt_text: str, model: str = DSQA_AUTORATER_MODEL) -> str:
    """Call litellm for Gemini 2.5 Flash with exponential backoff jitter."""
    from litellm import completion as _litellm_completion

    last_exc: Exception | None = None
    for attempt in range(DSQA_LLM_MAX_RETRIES):
        try:
            resp = _litellm_completion(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return (resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning(
                "DSQA autorater call failed (attempt %d/%d): %s",
                attempt + 1, DSQA_LLM_MAX_RETRIES, e,
            )
            if attempt < DSQA_LLM_MAX_RETRIES - 1:
                time.sleep(1 + (2 ** (attempt + random.random())))
    raise RuntimeError(f"DSQA autorater failed after {DSQA_LLM_MAX_RETRIES} attempts") from last_exc


def dsqa_grade_one(
    *,
    example_id: str,
    problem: str,
    answer: str,
    answer_type: str,
    response: str,
    problem_category: str = "Unknown",
    model: str = DSQA_AUTORATER_MODEL,
) -> dict:
    """Grade a single DSQA item. Returns the per-item rating dict.

    Mirrors process_single_row_for_rating + _reduce_llm_response_to_item_rating
    from the Kaggle starter notebook.
    """
    rating: dict[str, Any] = {
        "example_id": example_id,
        "query": problem,
        "response": response,
        "category_type": problem_category,
        "answer_type": answer_type,
        "expected_correct_answer": answer,
        "answer_correctness_explanation": None,
        "expected_correct_answer_list": None,
        "response_wrong_answers_list": None,
        "grader_ratings_list": None,
        "empty_model_response": False,
        "empty_auto_rater_response": False,
        "invalid_auto_rater_response": False,
        "rating_response": "",
        "rating_prompt": "",
        "error_message": None,
    }

    if not response:
        rating["empty_model_response"] = True
        rating["error_message"] = "AI response was empty."
        return rating

    prompt_text = _build_dsqa_rater_input(problem, answer_type, answer, response)
    rating["rating_prompt"] = prompt_text

    try:
        raw = _dsqa_call_autorater(prompt_text, model=model)
    except Exception as e:  # noqa: BLE001
        rating["error_message"] = f"autorater call failed: {e}"
        return rating

    rating["rating_response"] = raw
    if not raw:
        rating["empty_auto_rater_response"] = True
        rating["error_message"] = "Auto-rater response was empty."
        return rating

    parsed = _parse_dsqa_json_response(raw)
    if parsed is None:
        rating["invalid_auto_rater_response"] = True
        rating["error_message"] = "Invalid JSON from autorater."
        return rating

    node = parsed.get("Answer Correctness") if isinstance(parsed, dict) else None
    if not isinstance(node, dict):
        rating["invalid_auto_rater_response"] = True
        rating["error_message"] = "Missing 'Answer Correctness' node."
        return rating

    explanation = node.get("Explanation")
    if not isinstance(explanation, str):
        rating["invalid_auto_rater_response"] = True
        rating["error_message"] = "Missing/malformed Explanation."
        return rating
    rating["answer_correctness_explanation"] = explanation

    details = _dsqa_extract_correctness_details(parsed)
    if details is None:
        rating["invalid_auto_rater_response"] = True
        rating["error_message"] = "Invalid Correctness Details."
        return rating
    rating["expected_correct_answer_list"] = list(details.keys())
    rating["grader_ratings_list"] = list(details.values())

    excessive = _dsqa_extract_excessive(parsed)
    if excessive is None:
        rating["invalid_auto_rater_response"] = True
        rating["error_message"] = "Invalid Excessive Answers."
        return rating
    if excessive:
        rating["response_wrong_answers_list"] = excessive

    return rating


def _dsqa_per_item_metric(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def _dsqa_ci_str(count: int, total: int, z: float = 1.96) -> str:
    """Wilson normal-approx CI string, matching the Kaggle notebook format."""
    if total == 0:
        return f"N/A ({count}/{total})"
    count = max(0, min(count, total))
    p = count / total
    margin = z * math.sqrt(p * (1.0 - p) / total)
    s = f"{round(p * 100, 2):.2f} ± {round(margin * 100, 2):.2f} ({count}/{total})"
    if total <= 5:
        s += " (CI not robust for n<=5)"
    return s


def _aggregate_dsqa_ratings(ratings: list[dict]) -> dict:
    """Aggregate item ratings → ProjectRating-shaped dict.

    Faithfully replicates aggregate_ratings from the Kaggle starter notebook.
    """
    if not ratings:
        return {}

    total = len(ratings)
    num_empty_model = num_empty_rater = num_invalid_rater = 0
    num_evaluated = 0
    num_all_correct = 0
    num_fully_incorrect = 0
    num_correct_with_excessive = 0
    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"evaluated": 0, "all_correct": 0})
    per_item_p: list[float] = []
    per_item_r: list[float] = []
    per_item_f1: list[float] = []

    for r in ratings:
        if r.get("invalid_auto_rater_response"):
            num_invalid_rater += 1
            continue
        if r.get("empty_auto_rater_response"):
            num_empty_rater += 1
            continue
        if r.get("empty_model_response"):
            num_empty_model += 1
            continue

        grader_list = r.get("grader_ratings_list")
        if grader_list is None:
            continue

        num_evaluated += 1
        cat = r.get("category_type") or "Unknown"
        category_stats[cat]["evaluated"] += 1

        num_correct = sum(1 for x in grader_list if x)
        tp = num_correct
        fn = len(grader_list) - num_correct
        has_expected = bool(grader_list)

        all_expected_correct = has_expected and num_correct == len(grader_list)
        if has_expected and num_correct == 0:
            num_fully_incorrect += 1

        excessive = r.get("response_wrong_answers_list") or []
        has_excessive = bool(excessive)
        fp = len(excessive)
        if has_excessive and (all_expected_correct or not has_expected):
            num_correct_with_excessive += 1

        is_all_correct = (all_expected_correct or not has_expected) and not has_excessive
        if is_all_correct:
            num_all_correct += 1
            category_stats[cat]["all_correct"] += 1

        m = _dsqa_per_item_metric(tp, fp, fn)
        per_item_p.append(m["precision"])
        per_item_r.append(m["recall"])
        per_item_f1.append(m["f1_score"])

    # Status decomposition from runner.py status field (mirrors BCP's
    # Total/Correct/Crashed/Incomplete in the BCP eval summary).
    n_crashed = sum(1 for r in ratings if r.get("run_status") == "crashed")
    n_incomplete = sum(
        1 for r in ratings
        if r.get("run_status") in ("max_iterations", "max_context_length")
    )
    accuracy_pct = round(num_all_correct / total * 100, 2) if total else 0.0

    summary: dict[str, Any] = {
        # ── BCP-style top-line decomposition ──
        "Total": total,
        "Correct": num_all_correct,
        "Crashed": n_crashed,
        "Incomplete": n_incomplete,
        "Accuracy (%)": accuracy_pct,
        # ── Existing DSQA fields ──
        "num_total_ratings": total,
        "num_empty_model_response": num_empty_model,
        "num_empty_auto_rater_response": num_empty_rater,
        "num_invalid_auto_rater_response": num_invalid_rater,
        "num_valid_ratings": total - num_empty_model - num_empty_rater - num_invalid_rater,
        "num_answer_correctness_evaluated": num_evaluated,
        "pct_empty_model_response": round(num_empty_model * 100.0 / total, 2),
        "pct_empty_auto_rater_response": round(num_empty_rater * 100.0 / total, 2),
        "pct_invalid_auto_rater_response": round(num_invalid_rater * 100.0 / total, 2),
        "pct_w_ci_all_answers_correct": "",
        "pct_w_ci_fully_incorrect_items": "",
        "pct_w_ci_correct_with_excessive_answers": "",
        "precision": "",
        "recall": "",
        "f1_score": "",
        "per_category": {},
    }

    if num_evaluated > 0:
        summary["pct_w_ci_all_answers_correct"] = _dsqa_ci_str(num_all_correct, num_evaluated)
        summary["pct_w_ci_fully_incorrect_items"] = _dsqa_ci_str(num_fully_incorrect, num_evaluated)
        summary["pct_w_ci_correct_with_excessive_answers"] = _dsqa_ci_str(num_correct_with_excessive, num_evaluated)
        summary["precision"] = f"{np.mean(per_item_p):.2%}"
        summary["recall"] = f"{np.mean(per_item_r):.2%}"
        summary["f1_score"] = f"{np.mean(per_item_f1):.2%}"
        summary["per_category"] = {
            cat: {
                "evaluated": s["evaluated"],
                "all_correct": s["all_correct"],
                "all_correct_pct": (s["all_correct"] / s["evaluated"]) if s["evaluated"] else 0.0,
            }
            for cat, s in sorted(category_stats.items())
        }

    # ── peak_tokens (max tokens_actual per question, aggregated) ──
    peaks = [int(r["peak_tokens"]) for r in ratings
             if isinstance(r.get("peak_tokens"), (int, float)) and r["peak_tokens"] > 0]
    if peaks:
        summary["peak_tokens"] = {
            "mean":   round(float(np.mean(peaks)), 1),
            "median": int(np.median(peaks)),
            "p95":    int(np.percentile(peaks, 95)),
            "max":    int(max(peaks)),
            "n":      len(peaks),
        }
    else:
        summary["peak_tokens"] = {"mean": 0, "median": 0, "p95": 0, "max": 0, "n": 0}

    # ── tool_calls: per-tool aggregates across all questions ──
    per_q_counts: dict[str, list[int]] = defaultdict(list)
    for r in ratings:
        tc = r.get("tool_call_counts") or {}
        if not isinstance(tc, dict):
            continue
        # Fill zeros for tools this question did NOT call, using union of tool
        # names seen so far. Two-pass below handles tools that appear later.
        for name, n in tc.items():
            try:
                per_q_counts[name].append(int(n))
            except (TypeError, ValueError):
                continue
    # Pad each tool's list with zeros so the per-q mean reflects all ratings.
    n_ratings = len(ratings)
    for name, lst in per_q_counts.items():
        if len(lst) < n_ratings:
            lst.extend([0] * (n_ratings - len(lst)))

    if per_q_counts:
        summary["tool_calls"] = {
            "total_by_tool":         {n: int(sum(v)) for n, v in sorted(per_q_counts.items())},
            "mean_per_q":            {n: round(float(np.mean(v)), 2) for n, v in sorted(per_q_counts.items())},
            "median_per_q":          {n: int(statistics.median(v)) for n, v in sorted(per_q_counts.items())},
            "questions_using_tool":  {n: int(sum(1 for x in v if x > 0)) for n, v in sorted(per_q_counts.items())},
        }
    else:
        summary["tool_calls"] = {
            "total_by_tool": {}, "mean_per_q": {},
            "median_per_q": {}, "questions_using_tool": {},
        }

    return summary


def _load_dsqa_ground_truth(path: str | Path) -> dict[str, dict]:
    """Load DSQA gold dict keyed by str example_id."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(item["example_id"]): item for item in raw}


def _compute_peak_tokens(run_data: dict) -> int:
    """Max `tokens_actual` across the token trajectory (0 if absent)."""
    traj = run_data.get("token_trajectory") or []
    peaks: list[int] = []
    for t in traj:
        if isinstance(t, dict):
            v = t.get("tokens_actual")
        else:
            v = t
        if isinstance(v, (int, float)) and v > 0:
            peaks.append(int(v))
    return max(peaks) if peaks else 0


def _extract_dsqa_response(run_data: dict) -> str:
    """Return the agent's final answer text for grading.

    Prefers run_data['final_answer'] (the parsed <answer>...</answer>
    contents). Falls back to the last assistant turn's text if missing —
    that path corresponds to runs that crashed or timed out.
    """
    fa = run_data.get("final_answer")
    if isinstance(fa, str) and fa.strip():
        return fa
    history = run_data.get("history") or []
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _evaluate_one_dsqa_file(
    run_path: Path,
    gold: dict[str, dict],
    output_dir: Path,
    model: str,
    force: bool,
) -> dict | None:
    eval_path = output_dir / f"{run_path.stem}_eval.json"
    if eval_path.exists() and not force:
        try:
            cached = json.loads(eval_path.read_text(encoding="utf-8"))
            # Backfill runtime stats that older eval files may lack — avoids
            # forcing a Gemini re-grade just to add peak_tokens.
            if "peak_tokens" not in cached or "tool_call_counts" not in cached:
                with run_path.open("r", encoding="utf-8") as f:
                    run_data = json.load(f)
                cached.setdefault("tool_call_counts", run_data.get("tool_call_counts", {}))
                cached["peak_tokens"] = _compute_peak_tokens(run_data)
                with eval_path.open("w", encoding="utf-8") as f:
                    json.dump(cached, f, indent=2, ensure_ascii=False)
            return cached
        except Exception:
            pass  # fall through to re-grade

    with run_path.open("r", encoding="utf-8") as f:
        run_data = json.load(f)

    qid = str(run_data.get("query_id") or run_path.stem.replace("run_", "", 1))
    gold_item = gold.get(qid)
    if gold_item is None:
        logger.warning("No gold for query_id %s — skipping", qid)
        return None

    response = _extract_dsqa_response(run_data)
    rating = dsqa_grade_one(
        example_id=qid,
        problem=gold_item["problem"],
        answer=gold_item.get("answer", "") or "",
        answer_type=gold_item.get("answer_type", "Single Answer"),
        response=response,
        problem_category=gold_item.get("problem_category", "Unknown"),
        model=model,
    )

    payload = {
        **rating,
        "run_status": run_data.get("status", "unknown"),
        "num_turns": run_data.get("num_turns"),
        "elapsed_sec": run_data.get("elapsed_sec"),
        "tool_call_counts": run_data.get("tool_call_counts", {}),
        "peak_tokens": _compute_peak_tokens(run_data),
    }
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def evaluate_deepsearchqa(
    input_dir: str,
    ground_truth: str,
    eval_dir: str,
    *,
    summary_dir: str | None = None,
    model: str = DSQA_AUTORATER_MODEL,
    max_workers: int = 5,
    force: bool = False,
) -> dict:
    """Run the official DSQA autorater over all run files in input_dir.

    Writes per-item *_eval.json under eval_dir. The aggregated
    evaluation_summary.json goes into summary_dir (default:
    <input_dir>/eval_summary), kept apart from the 900-file
    per-item directory so it stays easy to find.
    """
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "DSQA grader needs GEMINI_API_KEY (Google AI Studio). "
            "Get one at https://aistudio.google.com/apikey and export it "
            "(or place in .env at repo root)."
        )

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input directory {input_path} does not exist")
    gold = _load_dsqa_ground_truth(ground_truth)
    logger.info("Loaded %d gold items from %s", len(gold), ground_truth)

    output_dir = Path(eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("DSQA evaluations will be saved to %s", output_dir)

    # Runner writes <qid>.json (matches BCP convention). Exclude grader
    # outputs and known sidecar files.
    run_files = [
        p for p in sorted(input_path.glob("*.json"))
        if not p.name.endswith("_eval.json")
        and not p.name.endswith(".partial.json")
        and p.name not in {"evaluation_summary.json", "dsqa_summary.json"}
    ]
    if not run_files:
        logger.warning("No run files found in %s", input_path)
        return {}

    logger.info("Grading %d run files (workers=%d)", len(run_files), max_workers)
    ratings: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_one_dsqa_file, jp, gold, output_dir, model, force,
            ): jp for jp in run_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="DSQA"):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                jp = futures[fut]
                logger.error("Worker error on %s: %s", jp, e)
                continue
            if r is not None:
                ratings.append(r)

    summary = _aggregate_dsqa_ratings(ratings)
    summary["input_dir"] = str(input_path)
    summary["ground_truth"] = str(ground_truth)
    summary["judge_model"] = model

    summary_root = Path(summary_dir) if summary_dir else (input_path / "eval_summary")
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("DSQA — precision=%s, recall=%s, f1=%s, all_correct=%s",
                summary.get("precision"), summary.get("recall"),
                summary.get("f1_score"), summary.get("pct_w_ci_all_answers_correct"))
    logger.info("DSQA summary saved to %s", summary_path)
    return summary
