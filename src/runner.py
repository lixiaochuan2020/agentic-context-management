"""Unified runner for all benchmark modes.

Replaces normal_runner.py, memtool_runner.py, and bcp_runner.py.
Benchmark selection (`browsecomp` vs `browsecomp-plus`) drives which
prompts, tools, search backend, and answer detection format are used.
Memory tools are controlled independently via `use_memory_tools`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from src.config import MAX_TURNS, SEARCH_SEP_NUM_RESULTS, MAX_PAGE_CHARS_SEP
from src.history import HistoryManager
from src.client import BaseClient
from src.prompts import build_system_prompt as _build_system_prompt
from src.prompts import (
    _SUMMARY_INSTRUCTION,
    _SUMMARY_INSTRUCTION_NOQUERY,
    BENCHMARK_PARTS,
    QUERY_MEMORY_PROMPT,
    build_initial_prompt,
    STOP_PROMPT,
    BCP_STOP_PROMPT,
    BCP_CONTINUE_PROMPT,
    BCP_COMPRESS_PROMPT,
    DSQA_STOP_PROMPT,
)
from src.tools import get_tools, execute_search, execute_open
from src.training_data import save_training_sample
from src.configs import RunConfig

try:
    from litellm.exceptions import ContextWindowExceededError
except ImportError:
    ContextWindowExceededError = None  # type: ignore

logger = logging.getLogger(__name__)


# RunConfig now lives in src.configs (4 sub-configs + per-question fields).
# All defaults are defined there; this module just consumes them.


# ══════════════════════════════════════════════════════════
# BCP searcher initialization
# ══════════════════════════════════════════════════════════

def _init_bcp_searcher(index_path: str):
    """Initialize BM25Searcher + Qwen tokenizer for BCP corpus search."""
    import sys
    import argparse as _ap
    import importlib.util
    import types

    bcp_searcher_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "BrowseComp-Plus", "searcher"
    )
    searchers_dir = os.path.join(bcp_searcher_path, "searchers")

    if "searchers" not in sys.modules:
        _pkg = types.ModuleType("searchers")
        _pkg.__path__ = [searchers_dir]
        _pkg.__package__ = "searchers"
        sys.modules["searchers"] = _pkg

    def _load_submod(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "searchers"
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_submod("searchers.base", os.path.join(searchers_dir, "base.py"))
    _bm25_mod = _load_submod("searchers.bm25_searcher", os.path.join(searchers_dir, "bm25_searcher.py"))
    BM25Searcher = _bm25_mod.BM25Searcher

    from transformers import AutoTokenizer

    searcher = BM25Searcher(_ap.Namespace(index_path=index_path))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return searcher, tokenizer


# ══════════════════════════════════════════════════════════
# DR9K retrieval client (thin HTTP wrapper duck-typed to BCP's BM25Searcher)
# ══════════════════════════════════════════════════════════

class _DR9KClient:
    """HTTP client for the wiki-18 retrieval server (retrieval_server/server.py).

    Exposes the same .search(query, k) and .get_document(docid) method shapes
    as BCP's BM25Searcher, so the existing _execute_bcp_search and
    _execute_get_document helpers work unchanged.
    """

    def __init__(self, server_url: str, timeout: float = 60.0):
        import requests
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def search(self, query: str, k: int = 5) -> list[dict]:
        r = self.session.post(
            f"{self.server_url}/search",
            json={"query": query, "k": k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        # BCP's BM25Searcher.search returns [{docid, text, score}, ...]. Mirror that.
        return [
            {"docid": h["docid"], "text": h["text"], "score": h["score"]}
            for h in r.json()["hits"]
        ]

    def get_document(self, docid: str) -> dict | None:
        r = self.session.post(
            f"{self.server_url}/get_document",
            json={"docid": str(docid)},
            timeout=self.timeout,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        body = r.json()
        return {"docid": body["docid"], "text": body["text"]}


def _init_dr9k_client(server_url: str):
    """Initialize HTTP client + Qwen tokenizer for DR9K wiki-18 search."""
    from transformers import AutoTokenizer
    client = _DR9KClient(server_url)
    # Same tokenizer as BCP for snippet truncation in _execute_bcp_search
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return client, tokenizer


class _DR9KLiveWebClient:
    """Live-web searcher for deepresearch9k — a drop-in for _DR9KClient.

    search() hits Serper (Google) and returns organic hits with docid = URL;
    get_document(url) fetches the page via the shared execute_open() helper.
    Returns the same {docid,text,score} / {docid,text} shapes as _DR9KClient,
    so _execute_bcp_search / _execute_get_document (incl. snippet + 8192-token
    truncation) work unchanged — only the *source* of documents changes.
    """

    def __init__(self, timeout: float = 20.0):
        import requests
        from src.config import SERPER_API_KEY
        self._requests = requests
        self._key = SERPER_API_KEY
        self.timeout = timeout

    def search(self, query: str, k: int = 5) -> list[dict]:
        from src.tools import SERPER_URL
        if not self._key:
            logger.error("DR9K live-web: SERPER_API_KEY not set")
            return []
        try:
            r = self._requests.post(
                SERPER_URL,
                headers={"X-API-KEY": self._key, "Content-Type": "application/json"},
                json={"q": query, "num": k}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # noqa: BLE001 — surface as empty, agent retries
            logger.warning("DR9K live search failed for %r: %s", query[:60], e)
            return []
        hits = []
        for rank, o in enumerate(data.get("organic", [])[:k], 1):
            link = o.get("link", "")
            if not link:
                continue
            title, snip = o.get("title", ""), o.get("snippet", "")
            hits.append({"docid": link, "text": f"{title}\n{snip}".strip(),
                         "score": 1.0 / rank})
        return hits

    def get_document(self, docid: str) -> dict | None:
        # Fetch generously (~40k chars); _execute_get_document then applies the
        # SAME 8192-token cap used on the wiki-18 path.
        text = execute_open(str(docid), max_chars=40000)
        if text.startswith("[OPEN ERROR"):
            return None
        return {"docid": str(docid), "text": text}


def _init_dr9k_live_client():
    """Live-web (Serper) DR9K client + Qwen tokenizer (same tokenizer as wiki-18)."""
    from transformers import AutoTokenizer
    client = _DR9KLiveWebClient()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return client, tokenizer


# ══════════════════════════════════════════════════════════
# Chat-template prompt rendering (for traj inspection)
# ══════════════════════════════════════════════════════════

# Cache the chat-template tokenizer; both base and SFT Qwen3-4B-Thinking
# share the same tokenizer + chat_template, so a single load works for all
# qwen3-4b-thinking-* served-model-names.
_PROMPT_TOKENIZER = None
_PROMPT_TOKENIZER_PATH = "Qwen/Qwen3-4B-Thinking-2507"


def _get_prompt_tokenizer():
    global _PROMPT_TOKENIZER
    if _PROMPT_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _PROMPT_TOKENIZER = AutoTokenizer.from_pretrained(_PROMPT_TOKENIZER_PATH)
        except Exception as e:
            logger.warning("Could not load prompt tokenizer (%s); rendered_prompt will be None: %s",
                           _PROMPT_TOKENIZER_PATH, e)
            _PROMPT_TOKENIZER = False  # sentinel: load failed, don't retry
    return _PROMPT_TOKENIZER if _PROMPT_TOKENIZER else None


def _render_chat_prompt(messages: list[dict], tools: list[dict]) -> str | None:
    """Render the prompt the model actually sees, including the auto-injected
    <tools>{schemas}</tools> block. Used to populate `rendered_prompt` in run
    outputs so trajectories carry the canonical prompt for inspection."""
    tok = _get_prompt_tokenizer()
    if tok is None:
        return None
    try:
        # transformers' apply_chat_template wants the inner JSON Schema
        # (without the OpenAI {type:function, function:...} envelope).
        tools_inner = []
        for t in tools or []:
            if isinstance(t, dict) and "function" in t:
                tools_inner.append(t["function"])
            elif isinstance(t, dict):
                tools_inner.append(t)
        return tok.apply_chat_template(
            messages, tools=tools_inner or None,
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:
        logger.warning("apply_chat_template failed; rendered_prompt will be None: %s", e)
        return None


# ══════════════════════════════════════════════════════════
# Summarizer helpers (single copy, no more duplication)
# ══════════════════════════════════════════════════════════

_SUMMARIZER_MAX_TOKENS = 4_096


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> CoT blocks from a summarizer's output.

    Safety net for the case where someone points the summarizer at a
    thinking model. No-op for non-thinking models like gpt-5.4-mini.
    """
    return _THINK_RE.sub("", text or "").strip()


_MEMORY_RE = re.compile(r"<memory>(.*?)</memory>", re.DOTALL | re.IGNORECASE)


def _extract_memory(text: str) -> tuple[str, bool]:
    """Extract content inside the first <memory>...</memory> block.

    Returns (content, parse_ok). `parse_ok=True` requires all of:
      1. A properly bounded <memory>...</memory> pair was found.
      2. The captured inner content is ≥50 chars (rejects literal '...').
      3. The captured inner contains NO `<memory>` and NO `</think>` —
         these are restart-mode artifacts (model emitted multiple
         `<memory>` opens / `</think>` resets before a final close, and
         the non-greedy regex spanned all of them).

    `parse_ok=False` means the fallback path was taken — we still return
    `text.strip()` as best-effort content so the caller has something to
    insert into the tool_response. Downstream consumers should filter on
    `parse_ok` rather than re-parse the saved body.

    Summarizer is instructed to wrap output in <memory> tags so we can
    strip off any pre-amble / chain-of-thought / repeated-instruction
    bullets the model emits before the actual memory entry."""
    if not text:
        return "", False
    m = _MEMORY_RE.search(text)
    if m:
        content = m.group(1).strip()
        if (len(content) >= 50
                and "<memory>" not in content
                and "</think>" not in content):
            return content, True
    return text.strip(), False


def call_summarizer(
    client: BaseClient,
    query: str,
    context_turns: list[dict],
    disable_query_memory: bool = False,
) -> tuple[str, bool]:
    """Call the summarizer on the given slice of messages, return memory text
    and a parse-ok flag from `_extract_memory`.

    Retries once on `parse_ok=False` (model failed to emit a clean
    `<memory>...</memory>` pair). After 2 total attempts, returns whatever
    the last attempt produced — caller filters on `parse_ok`.

    Serializes `context_turns` to a single text block and feeds it through
    the `_SUMMARY_INSTRUCTION` template as one user message — same pattern as
    `_handle_query_memory` / `QUERY_MEMORY_PROMPT`.
    """
    conversation_text = "\n\n".join(
        f"[{m.get('role', '?')}] {m.get('content', '') or ''}"
        for m in context_turns
    )
    instruction = _SUMMARY_INSTRUCTION_NOQUERY if disable_query_memory else _SUMMARY_INSTRUCTION
    prompt = instruction.format(
        question=query,
        conversation=conversation_text,
    )
    msgs = [{"role": "user", "content": prompt}]

    last_content, last_ok = "", False
    for attempt in range(2):  # 1 try + 1 retry
        result = client.generate(msgs, max_new_tokens=_SUMMARIZER_MAX_TOKENS)
        if not result:
            logger.warning("call_summarizer attempt %d empty text (finish=%s)",
                           attempt, getattr(client, "_last_finish_reason", "?"))
            last_content, last_ok = "", False
            continue
        content, ok = _extract_memory(_strip_thinking(result))
        last_content, last_ok = content, ok
        if ok:
            if attempt > 0:
                logger.info("call_summarizer recovered on attempt %d", attempt)
            break
        logger.warning("call_summarizer attempt %d: <memory> tag not detected cleanly, retry",
                       attempt)
    return last_content, last_ok


# ══════════════════════════════════════════════════════════
# BCP search execution helpers
# ══════════════════════════════════════════════════════════

_HINT_SEEN_IN_CONTEXT = (
    "\n[seen earlier; this preview is truncated. Call get_document(docid) "
    "for the full text if needed.]"
)
_HINT_SEEN_COMPRESSED = (
    "\n[seen earlier but compressed by manage_context; this preview is truncated. "
    "Call query_memory(summary_id, query) to retrieve the relevant earlier summary.]"
)
# re_mem_noquery ablation: no query_memory — point back to get_document, which
# still re-fetches the full document from the corpus.
_HINT_SEEN_COMPRESSED_NOQUERY = (
    "\n[seen earlier but compressed by manage_context; this preview is truncated. "
    "Call get_document for the full text if you need it again.]"
)


def _execute_bcp_search(searcher, tokenizer, query: str, k: int,
                        snippet_new_tokens: int = 512,
                        snippet_seen_tokens: int = 128,
                        visited: set[str] | None = None,
                        in_context: set[str] | None = None,
                        disable_query_memory: bool = False,
                        ) -> tuple[str, list[str]]:
    """Run retrieval (BM25 or dense, duck-typed) and return (text, docids).

    3-tier snippet length:
      (a) new docs   → snippet_new_tokens preview (default 512)
      (b) seen & still in_context → snippet_seen_tokens + hint to get_document
      (c) seen but compressed     → snippet_seen_tokens + hint to query_memory
    """
    hits = searcher.search(query, k=k)
    parts = []
    docids = []
    for i, hit in enumerate(hits, 1):
        docid = str(hit["docid"])
        docids.append(docid)
        text = hit.get("text", "")

        if visited is None or docid not in visited:
            cap = snippet_new_tokens
            hint = ""
        elif in_context is not None and docid in in_context:
            cap = snippet_seen_tokens
            hint = _HINT_SEEN_IN_CONTEXT
        else:
            cap = snippet_seen_tokens
            hint = (_HINT_SEEN_COMPRESSED_NOQUERY if disable_query_memory
                    else _HINT_SEEN_COMPRESSED)

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > cap:
            text = tokenizer.decode(tokens[:cap], skip_special_tokens=True)
        parts.append(f"[{i}] docid={docid} score={hit.get('score', 0):.2f}\n{text}{hint}")

        if visited is not None:
            visited.add(docid)
        if in_context is not None:
            in_context.add(docid)

    result = "\n\n".join(parts) if parts else "No results found."
    return result, docids


def _execute_get_document(searcher, docid: str,
                          tokenizer=None,
                          max_tokens: int = 8192,
                          visited: set[str] | None = None,
                          in_context: set[str] | None = None,
                          ) -> str:
    """Retrieve a document by docid, capped at `max_tokens` tokens."""
    doc = searcher.get_document(docid)
    if doc is None:
        return json.dumps({"error": f"Document with docid '{docid}' not found"})
    text = doc["text"]
    if tokenizer is not None and max_tokens > 0:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            text += f"\n\n[... document truncated to {max_tokens} tokens ...]"
    if visited is not None:
        visited.add(str(docid))
    if in_context is not None:
        in_context.add(str(docid))
    return f"docid={doc['docid']}\n\n{text}"


# ══════════════════════════════════════════════════════════
# manage_context handler
# ══════════════════════════════════════════════════════════

def compress_messages(
    messages: list[dict],
    start_pos: int,
    end_pos: int,
    question: str,
    summarizer_client: BaseClient,
    summary_dir: str | None,
    summary_id: int,
    disable_query_memory: bool = False,
) -> tuple[str, int, list[dict], bool] | None:
    """Pure compression: summarize messages[start_pos:end_pos], save the
    archived range to summary_{N}.json, return (formatted tool_response,
    new summary_id, archived_messages, parse_ok).

    `parse_ok` is the flag from `_extract_memory` — True iff the summarizer
    emitted a properly bounded `<memory>...</memory>` block with ≥50 chars
    of inner content; False means the fallback path was taken (raw text).

    Does NOT mutate `messages` — callers handle deletion themselves.
    Returns None if the range is empty (nothing to compress).

    Shared by runner._handle_manage_context (live mc during rollout) and
    teacher_guided_rollout._execute_guided_action (mc inside an off-trajectory guided turn).
    """
    if start_pos >= end_pos:
        return None
    full_range = [
        {k: v for k, v in m.items() if k != "_idx"}
        for m in messages[start_pos:end_pos]
    ]
    txt, parse_ok = call_summarizer(summarizer_client, question, full_range,
                                    disable_query_memory=disable_query_memory)
    new_summary_id = summary_id + 1
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
        entries = []
        for m in full_range:
            entry = {"role": m.get("role", "?"), "content": m.get("content", "")}
            if "tool_calls" in m:
                entry["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                entry["tool_call_id"] = m["tool_call_id"]
            entries.append(entry)
        with open(os.path.join(summary_dir, f"summary_{new_summary_id}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    feedback = f"[summary_id: {new_summary_id}] {txt}"
    return feedback, new_summary_id, full_range, parse_ok


def _handle_manage_context(
    client, hm, question, args, tools, summary_dir,
    training_data_dir, question_id, correct_answer,
    mode, turn, model_name, benchmark,
    mem_operations, cm_call_count,
    last_boundary_pos: int,
    summary_id: int,
    tool_call_id: str | None = None,
    summarizer_client: BaseClient | None = None,
    drop_call_message: bool = False,
    disable_query_memory: bool = False,
) -> tuple[str, int, int, bool | None]:
    """Execute manage_context. Compresses
    hm.messages[last_boundary_pos : current_assistant_pos)
    where current_assistant_pos = len(hm.messages) - 1 (the assistant message
    that issued this tool call has already been appended).

    When `drop_call_message=True`, the assistant turn that issued mc is itself
    included in the summarized + deleted range — its <think> content goes into
    the summarizer input and the message is removed from live history. Only
    the tool-result summary remains.

    Returns (feedback, new_last_boundary_pos, new_summary_id_counter, parse_ok).
    `parse_ok` is True/False if a summary was generated, or None if no
    compression actually happened (empty range).
    On error / empty range, last_boundary_pos and summary_id are unchanged.
    """
    context_before = hm.get_messages_for_model()
    _tbc = client.count_tokens(context_before, tools=tools)

    current_assistant_pos = len(hm.messages) - 1  # the manage_context tool-call message
    start_pos = last_boundary_pos
    # half-open: [start_pos, end_pos). When drop_call_message=True the mc
    # assistant turn itself joins the deleted range.
    end_pos = current_assistant_pos + 1 if drop_call_message else current_assistant_pos

    summ_client = summarizer_client or client
    result = compress_messages(
        messages=hm.messages,
        start_pos=start_pos, end_pos=end_pos,
        question=question, summarizer_client=summ_client,
        summary_dir=summary_dir, summary_id=summary_id,
        disable_query_memory=disable_query_memory,
    )
    if result is None:
        feedback = (
            "Error: nothing to compress. There are no messages between your "
            "previous manage_context call and this one."
        )
        hm.add_message_dict(client.build_tool_message(feedback, tool_call_id=tool_call_id))
        return feedback, last_boundary_pos, summary_id, None
    feedback, new_summary_id, _archived, parse_ok = result

    removed = hm.compress_range(start_pos, end_pos)
    hm._tokens_before_compression = _tbc
    hm.add_message_dict(client.build_tool_message(feedback, tool_call_id=tool_call_id))

    # The new boundary is the position just AFTER the tool result we just
    # appended. Next manage_context will compress everything from there
    # (exclusive of that tool result) up to the next manage_context request.
    new_last_boundary_pos = len(hm.messages)

    mem_operations.append({
        "tool": "manage_context",
        "summary_id": new_summary_id,
        "compressed_message_count": len(removed),
    })

    if training_data_dir:
        context_after = hm.get_messages_for_model()
        tokens_after = client.count_tokens(context_after, tools=tools)
        # tool_result_content is the raw summarizer output (without the
        # `[summary_id: N]` prefix that compress_messages adds).
        txt = feedback.removeprefix(f"[summary_id: {new_summary_id}] ")
        save_training_sample(
            out_dir=training_data_dir, question_id=question_id,
            call_index=cm_call_count, question=question,
            correct_answer=correct_answer, mode=mode, turn=turn,
            context_before=context_before, tokens_before=_tbc,
            tool_name="manage_context", tool_args=args, success=True,
            feedback=feedback, tool_result_content=txt,
            context_after=context_after, tokens_after=tokens_after,
            metadata={"benchmark": benchmark, "model": model_name},
        )

    logger.info(
        "  manage_context() — summary_id=%d, compressed %d messages, parse_ok=%s",
        new_summary_id, len(removed), parse_ok,
    )
    return feedback, new_last_boundary_pos, new_summary_id, parse_ok


# ══════════════════════════════════════════════════════════
# query_memory handler
# ══════════════════════════════════════════════════════════

def _handle_query_memory(
    client: BaseClient,
    args: dict,
    summary_dir: str | None,
    question: str,
    mem_operations: list[dict],
    summarizer_client: BaseClient | None = None,
) -> str:
    """Execute query_memory(summary_id, query). Reads
    {summary_dir}/summary_{summary_id}.json and asks the summarizer LLM to
    extract content matching `query` from the original messages."""
    summary_id = args.get("summary_id")
    query_text = args.get("query", "")

    if summary_id is None or not query_text:
        return "Error: missing required parameters (summary_id, query)."

    if summary_dir is None:
        return "Error: query_memory is unavailable (no summary directory configured)."

    summary_file = os.path.join(summary_dir, f"summary_{summary_id}.json")
    if not os.path.exists(summary_file):
        return f"Error: summary_id {summary_id} not found."

    with open(summary_file, encoding="utf-8") as f:
        msgs = json.load(f)

    history = "\n".join(
        f"[{m.get('role', '?')}] {m.get('content', '')}"
        for m in msgs
    )

    summ_client = summarizer_client or client
    result = _strip_thinking(summ_client.generate([
        {"role": "user", "content": QUERY_MEMORY_PROMPT.format(
            summary_id=summary_id, history=history, query=query_text)}
    ]))
    feedback = f"[query_memory: summary_id={summary_id}]\n\n{result}"
    mem_operations.append({
        "tool": "query_memory",
        "summary_id": summary_id,
        "query": query_text,
    })
    logger.info("  query_memory(summary_id=%s, '%s')", summary_id, query_text[:60])
    return feedback


# ══════════════════════════════════════════════════════════
# Unified run function
# ══════════════════════════════════════════════════════════

def run(client: BaseClient, question: str, config: RunConfig,
        *, answer_type: str | None = None) -> dict[str, Any]:
    """Unified agentic loop for browsecomp / browsecomp-plus / deepresearch9k / deepsearchqa.

    answer_type is only consumed by benchmarks whose answer format varies
    per question (currently: deepsearchqa with "Single Answer" / "Set Answer").
    Other benchmarks ignore it.
    """
    is_bcp = (config.runtime.benchmark == "browsecomp-plus")
    is_dr9k = (config.runtime.benchmark == "deepresearch9k")
    is_dsqa = (config.runtime.benchmark == "deepsearchqa")
    # DR9K shares BCP's search+get_document tool surface and answer format;
    # they only differ in retrieval backend.
    uses_local_retrieval = is_bcp or is_dr9k
    tools = get_tools(config.runtime.benchmark, use_memory_tools=config.runtime.use_memory_tools,
                      disable_query_memory=config.runtime.disable_query_memory)

    # ── Summarizer client ──────────────────────────────────
    # By default, distinct from the agent client (manage_context /
    # query_memory routed at api.openai.com). When the summarizer model
    # matches the agent's model_name, share the agent client to avoid
    # duplicating credentials / endpoints (e.g. bedrock runs).
    summarizer_client: BaseClient | None = None
    if config.runtime.use_memory_tools and config.summarizer.model:
        if config.summarizer.model == config.agent.model:
            summarizer_client = client
        else:
            from src.client import LiteLLMClient
            summarizer_client = LiteLLMClient(
                config.summarizer.model,
                api_base=(config.summarizer.api_base or None),
            )

    # ── Benchmark-driven selection ─────────────────────────
    parts = BENCHMARK_PARTS[config.runtime.benchmark]
    _has_answer_fn = parts["has_answer"]
    _extract_answer_fn = parts["extract_answer"]
    _extract_conf_fn = parts["extract_confidence"]
    if is_dsqa:
        _stop_prompt = DSQA_STOP_PROMPT
    elif uses_local_retrieval:
        _stop_prompt = BCP_STOP_PROMPT
    else:
        _stop_prompt = STOP_PROMPT

    # ── Retrieval backend init (only when needed) ─────────
    bcp_searcher = bcp_tokenizer = None
    dr9k_client = dr9k_tokenizer = None
    if is_bcp:
        if config.retrieval.index_path is None:
            raise ValueError("--index_path is required for browsecomp-plus")
        # Qwen3-8B-Embedding is opt-in via $BCP_RETRIEVER=qwen3_8b_embedding
        # (or config.retrieval.backend). Falls back to BM25 otherwise. BM25 Lucene
        # is still loaded for document-text lookup either way.
        import os as _os
        _retriever = _os.environ.get("BCP_RETRIEVER", config.retrieval.backend).lower()
        bcp_searcher, bcp_tokenizer = _init_bcp_searcher(config.retrieval.index_path)
        if _retriever == "qwen3_8b_embedding":
            _shards = (config.retrieval.qwen3_8b_embedding_shards_dir
                       or _os.environ.get("BCP_QWEN3_8B_EMBEDDING_SHARDS_DIR"))
            if not _shards:
                raise ValueError("qwen3_8b_embedding retriever requires shards_dir "
                                 "(--qwen3_8b_embedding_shards_dir or "
                                 "$BCP_QWEN3_8B_EMBEDDING_SHARDS_DIR)")
            from src.qwen3_8b_embedding_retriever import get_or_build
            from src.embed_client import make_embedder_from_env
            _embed_fn = make_embedder_from_env()
            bcp_searcher = get_or_build(
                shards_dir=_shards,
                text_searcher=bcp_searcher,  # delegates get_document() to BM25 Lucene
                embed_fn=_embed_fn,
            )
            logger.info("BCP retriever = Qwen3-8B-Embedding (shards=%s)", _shards)
        else:
            logger.info("BCP retriever = BM25")
    elif is_dr9k:
        if os.environ.get("DR9K_LIVE_WEB") == "1":
            logger.info("DR9K retriever = LIVE WEB (Serper search + URL fetch)")
            dr9k_client, dr9k_tokenizer = _init_dr9k_live_client()
        else:
            if not config.retrieval.retrieval_server_url:
                raise ValueError("--server_url is required for deepresearch9k (or set DR9K_LIVE_WEB=1)")
            dr9k_client, dr9k_tokenizer = _init_dr9k_client(config.retrieval.retrieval_server_url)

    # ── Build history ──────────────────────────────────────
    if config.initial_history is not None:
        hm = HistoryManager.from_saved_history(
            config.initial_history,
            raw_history=config.initial_raw_history,
        )
        logger.info("  continue mode: loaded %d messages", len(hm.messages))
    else:
        hm = HistoryManager()
        system_msg = _build_system_prompt(
            config.runtime.benchmark,
            context_window=config.agent.context_window,
            use_memory_tools=config.runtime.use_memory_tools,
            disable_query_memory=config.runtime.disable_query_memory,
            answer_type=answer_type,
        )
        hm.add_message("system", system_msg)
        hm.add_message("user", question)

    # ── manage_context state ───────────────────────────────
    # last_boundary_pos: index in hm.messages where the next manage_context
    #   compression should start (half-open interval up to the manage_context
    #   tool-call message itself). Default = len(hm.messages) right after
    #   the initial messages are added — so the system prompt and the
    #   original user question are preserved (fresh mode), and any loaded
    #   continue-mode history is treated as already-finalized context.
    #   Teacher-guided rollout sets compress_initial_history=True to allow
    #   compressing the loaded history (boundary stays at 1, i.e. compress
    #   from the message right after msg[0]/system).
    # summary_id_counter: monotonically increasing; first manage_context
    #   produces summary_id=1. In continue mode, scan summary_dir for the
    #   highest existing summary_{N}.json and continue from there.
    last_boundary_pos: int = len(hm.messages)
    summary_id_counter: int = 0
    summary_dir = config.runtime.workspace_root  # field name kept for backward compat
    if config.initial_history is not None:
        # Explicit overrides win (e.g. teacher_guided_rollout passes both).
        if config.initial_last_boundary_pos is not None:
            last_boundary_pos = config.initial_last_boundary_pos
        elif config.runtime.compress_initial_history:
            last_boundary_pos = 1
        if config.initial_summary_id is not None:
            summary_id_counter = config.initial_summary_id
        else:
            # Scan existing summary files in workspace to continue summary_id numbering.
            if summary_dir and os.path.isdir(summary_dir):
                import glob as _glob
                _existing = _glob.glob(os.path.join(summary_dir, "summary_*.json"))
                for _path in _existing:
                    try:
                        n = int(os.path.basename(_path)
                                .removeprefix("summary_").removesuffix(".json"))
                        if n > summary_id_counter:
                            summary_id_counter = n
                    except ValueError:
                        pass
            # Also scan the history text for [summary_id: N] markers so that new
            # summaries never reuse an ID that appears in the loaded history.
            if config.runtime.compress_initial_history:
                import re as _re
                _sid_re = _re.compile(r"\[summary_id:\s*(\d+)\]")
                for _m in hm.messages:
                    _c = _m.get("content") or ""
                    for _match in _sid_re.finditer(_c):
                        _n = int(_match.group(1))
                        if _n > summary_id_counter:
                            summary_id_counter = _n

    # Capture the turn-0 rendered prompt (with auto-injected <tools>) so the
    # saved trajectory carries the canonical view the model actually sees.
    # Commented out to keep saved trajectories smaller — re-enable if needed
    # for debugging the chat-template-rendered prompt sent to the model.
    # rendered_prompt = _render_chat_prompt(hm.get_messages_for_model(), tools)

    # ── Tracking structures ────────────────────────────────
    token_trajectory: list[dict] = []
    tool_call_log: list[dict] = []
    mem_operations: list[dict] = []
    normalized_results: list[dict] = []
    # Per-summary parse-ok flags (one entry per manage_context call that
    # actually produced a summary). `parse_ok=True` means the summarizer's
    # raw output had a properly bounded `<memory>...</memory>` block with
    # ≥50 chars of inner content; `False` means the fallback path was taken.
    summary_parse_ok: list[dict] = []
    retrieved_docids: set[str] = set()
    # Per-rollout doc tracking for 3-tier search truncation. `visited` grows
    # monotonically; `in_context` is reset on every manage_context call.
    visited_docids: set[str] = set()
    in_context_docids: set[str] = set()
    tool_call_counts: dict[str, int] = {}
    cm_call_count: int = 0
    suppressed_answer_count: int = 0
    compress_prompt_count: int = 0
    MAX_SUPPRESSED_ANSWERS = 50
    MAX_COMPRESS_RETRIES = 10
    final_answer = ""
    confidence = 100
    error: str | None = None
    # Status string written into the trajectory JSON. One of:
    #   "success"            — final_answer extracted (grader decides correctness)
    # Status conventions (kept in sync with experiments/INDEX.md):
    #   "complete"           — rollout terminated cleanly (no crash / no
    #                          max-iter / no context-overflow). Whether a
    #                          final_answer was extracted is irrelevant here;
    #                          missing answers are picked up by the grader
    #                          and marked "failed" in the eval JSON.
    #   "max_iterations"     — for-loop walked to max_iter without converging
    #   "max_context_length" — ContextWindowExceededError during generate()
    #   "crashed"            — other generate() exception
    status = "complete"

    # Token accounting
    _prev_actual = client.count_tokens(hm.get_messages_for_model(), tools=tools)
    _cumulative_raw = _prev_actual
    _output_tokens_baseline = client.snapshot_output_tokens() if hasattr(client, "snapshot_output_tokens") else 0
    _total_input_tokens = 0
    _total_output_tokens = 0
    # generation-efficiency timing: _gen_time_sec = Σ LLM-call durations (pure generation,
    # excludes retrieval/tool waits); span = first-gen-start → last-gen-end (incl inter-turn waits).
    _gen_time_sec = 0.0
    _gen_first_start: float | None = None
    _gen_last_end: float | None = None

    # current_context_tokens — what we project to the model as
    # "[CURRENT CONTEXT TOKEN: N]" at the end of its most recent response.
    # Conservative upper bound on the NEXT prompt size, including the
    # rollout.max_new_tokens reservation vLLM sets aside for output. This
    # way N can be compared directly against context_window; when N ≈
    # context_window the model knows the next generate() will fail and
    # must compress now.
    #   N = count_tokens(hm.get_messages_for_model(), tools) + max_new_tokens
    # Recomputed from the actual post-turn message list at the end of every
    # loop iteration, so compression (manage_context) is reflected
    # immediately in the marker the NEXT turn sees.
    current_context_tokens: int | None = None

    def _build_result(current_turn: int) -> dict[str, Any]:
        mode_tag = f"{config.runtime.benchmark}_memtool" if config.runtime.use_memory_tools else f"{config.runtime.benchmark}_normal"
        return {
            "metadata": {"model": config.agent.model, "mode": mode_tag, "api": "litellm"},
            "query_id": config.question_id,
            "tool_call_counts": dict(tool_call_counts),
            "usage": {
                "input_tokens": _total_input_tokens,
                "output_tokens": _total_output_tokens,
                "total_tokens": _total_input_tokens + _total_output_tokens,
            },
            "efficiency": {
                "gen_time_sec": round(_gen_time_sec, 3),
                "gen_span_sec": (round(_gen_last_end - _gen_first_start, 3)
                                 if _gen_first_start is not None and _gen_last_end is not None else None),
                "output_tokens": _total_output_tokens,
                "total_tokens": _total_input_tokens + _total_output_tokens,
                "avg_gen_speed_tok_s": (round(_total_output_tokens / _gen_time_sec, 1)
                                        if _gen_time_sec > 0 else None),
            },
            "status": status,
            "retrieved_docids": sorted(retrieved_docids),
            "result": normalized_results,
            "question": question,
            "correct_answer": config.correct_answer,
            "final_answer": final_answer,
            "confidence": confidence,
            "token_trajectory": token_trajectory,
            "tool_calls": tool_call_log,
            "mem_operations": mem_operations,
            "history": hm.get_messages_for_model(),
            "raw_history": hm.get_raw_history(),
            # "rendered_prompt": rendered_prompt,
            "summary_parse_ok": summary_parse_ok,
            "num_turns": current_turn + 1,
            "min_turns": config.runtime.min_turns,
            "suppressed_answer_count": suppressed_answer_count,
            "continue_mode": config.initial_history is not None,
            "error": error,
        }

    def _write_cache(current_turn: int):
        if not config.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(config.cache_path), exist_ok=True)
            partial = _build_result(current_turn)
            partial["status"] = "in_progress" if status == "complete" else status
            with open(config.cache_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("  failed to write trajectory cache: %s",
                           traceback.format_exc())

    # ── Continue mode: pre-loop context management ─────────
    if config.initial_history is not None:
        _last_asst_text = ""
        for _m in reversed(hm.messages):
            if _m.get("role") == "assistant":
                _last_asst_text = _m.get("content") or ""
                break

        _init_tokens = client.count_tokens(hm.get_messages_for_model(), tools=tools)
        logger.info("  continue mode: initial tokens=%d (%.1f%% of %d)",
                     _init_tokens, _init_tokens / config.agent.context_window * 100, config.agent.context_window)

        if _init_tokens >= int(config.agent.context_window * 0.95) and config.runtime.use_memory_tools:
            logger.info("  continue mode: context nearly full, injecting compress prompt")
            hm.add_message("user", BCP_COMPRESS_PROMPT if uses_local_retrieval else STOP_PROMPT)
        elif _has_answer_fn(_last_asst_text):
            logger.info("  continue mode: previous answer detected, injecting continue prompt")
            hm.add_message("user", BCP_CONTINUE_PROMPT if uses_local_retrieval else
                           "Your previous answer was incorrect. Continue investigating with different search queries.")

        _prev_actual = client.count_tokens(hm.get_messages_for_model(), tools=tools)
        _cumulative_raw = _prev_actual

    # ── Main loop ──────────────────────────────────────────
    # BCP / DR9K / DSQA honour config.runtime.max_iterations. browsecomp
    # (real web, no SFT cap) stays uncapped — turns there are bounded by
    # the model and context window naturally.
    uses_iter_cap = uses_local_retrieval or is_dsqa
    max_iter = config.runtime.max_iterations if uses_iter_cap else MAX_TURNS
    turn = -1
    for turn in range(max_iter):
        # Resume path: on turn 0, if initial_history ended with an assistant
        # message that carries unexecuted tool_calls (e.g. teacher-guided
        # iter rollout hands off the guided turn this way), use those tool_calls as if
        # we'd just generated them — no client.generate() call. The assistant
        # is already in hm.messages so we don't re-add it.
        resume_pending = (
            turn == 0
            and config.initial_history is not None
            and hm.messages
            and hm.messages[-1].get("role") == "assistant"
            and hm.messages[-1].get("tool_calls")
        )
        if resume_pending:
            pending_asst = hm.messages[-1]
            response_text = pending_asst.get("content") or ""
            all_tool_calls = []
            for tc in pending_asst.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {}
                all_tool_calls.append({
                    "id": tc.get("id"),
                    "name": fn.get("name") or tc.get("name"),
                    "arguments": args,
                })
            logger.info(
                "  turn=0 (resume): %d pending tool_call(s) from initial_history",
                len(all_tool_calls),
            )
        else:
            msgs = hm.get_messages_for_model(token_hint=current_context_tokens)
            logger.info("  turn=%d, messages=%d", turn, len(msgs))

            try:
                _g0 = time.perf_counter()
                if _gen_first_start is None:
                    _gen_first_start = time.time()
                response = client.generate(msgs, tools=tools, max_new_tokens=config.agent.rollout.max_new_tokens)
                _gen_time_sec += time.perf_counter() - _g0
                _gen_last_end = time.time()
            except Exception as exc:
                if ContextWindowExceededError and isinstance(exc, ContextWindowExceededError):
                    logger.error("  turn=%d — ContextWindowExceededError, stopping.", turn)
                    error = "context_window_exceeded"
                    status = "max_context_length"
                    # Try to extract answer from last assistant message
                    for m in reversed(hm.get_messages_for_model()):
                        if m.get("role") == "assistant":
                            content = m.get("content", "")
                            if _has_answer_fn(content):
                                final_answer = _extract_answer_fn(content)
                                confidence = _extract_conf_fn(content)
                            break
                else:
                    logger.error("  turn=%d — crashed during generate(): %s\n%s",
                                 turn, exc, traceback.format_exc())
                    error = f"crash: {type(exc).__name__}: {exc}"
                    status = "crashed"
                _write_cache(turn)
                break

            # Track usage
            _last_prompt = getattr(client, "_last_prompt_tokens", None)
            _last_completion = getattr(client, "_last_completion_tokens", None)
            if _last_prompt is not None:
                _total_input_tokens += _last_prompt
            if _last_completion is not None:
                _total_output_tokens += _last_completion

            all_tool_calls = client.parse_all_tool_calls()
            response_text = response if isinstance(response, str) else ""

            # Record text output in normalized_results
            if response_text and not all_tool_calls:
                normalized_results.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": response_text,
                })

            hm.add_message_dict(client.build_assistant_message(response))

        if all_tool_calls:
            for tc in all_tool_calls:
                tc_id = tc["id"]
                name = tc["name"]
                args = tc["arguments"]
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

                # ── search ──────────────────────────────────
                if name == "search":
                    query_str = args.get("query", "")
                    if not query_str:
                        result_str = "Error: missing required parameter 'query'."
                        docids = []
                    elif is_bcp:
                        result_str, docids = _execute_bcp_search(
                            bcp_searcher, bcp_tokenizer, query_str,
                            config.retrieval.k,
                            snippet_new_tokens=config.retrieval.snippet_new_tokens,
                            snippet_seen_tokens=128,
                            visited=visited_docids,
                            in_context=in_context_docids,
                            disable_query_memory=config.runtime.disable_query_memory)
                        retrieved_docids.update(docids)
                    elif is_dr9k:
                        result_str, docids = _execute_bcp_search(
                            dr9k_client, dr9k_tokenizer, query_str,
                            config.retrieval.k,
                            snippet_new_tokens=config.retrieval.snippet_new_tokens,
                            snippet_seen_tokens=128,
                            visited=visited_docids,
                            in_context=in_context_docids,
                            disable_query_memory=config.runtime.disable_query_memory)
                        retrieved_docids.update(docids)
                    else:
                        result_str = execute_search(query_str, num_results=SEARCH_SEP_NUM_RESULTS)
                        docids = []
                    hm.add_message_dict(client.build_tool_message(result_str, tool_call_id=tc_id))
                    normalized_results.append({
                        "type": "tool_call",
                        "tool_name": "search",
                        "arguments": json.dumps(args),
                        "output": result_str,
                    })
                    logger.info("  search('%s') — %d chars", query_str[:60], len(result_str))

                # ── get_document (BCP / DR9K) ───────────────
                elif name == "get_document":
                    docid = args.get("docid", "")
                    if not docid:
                        result_str = "Error: missing required parameter 'docid'."
                    else:
                        searcher = dr9k_client if is_dr9k else bcp_searcher
                        _tok = dr9k_tokenizer if is_dr9k else bcp_tokenizer
                        result_str = _execute_get_document(
                            searcher, docid,
                            tokenizer=_tok, max_tokens=8192,
                            visited=visited_docids,
                            in_context=in_context_docids)
                        retrieved_docids.add(str(docid))
                    hm.add_message_dict(client.build_tool_message(result_str, tool_call_id=tc_id))
                    normalized_results.append({
                        "type": "tool_call",
                        "tool_name": "get_document",
                        "arguments": json.dumps(args),
                        "output": result_str,
                    })
                    logger.info("  get_document('%s') — %d chars", docid, len(result_str))

                # ── open (BrowseComp only) ──────────────────
                elif name == "open":
                    url = args.get("url", "")
                    if not url:
                        result_str = "Error: missing required parameter 'url'."
                    else:
                        result_str = execute_open(url, max_chars=MAX_PAGE_CHARS_SEP)
                    hm.add_message_dict(client.build_tool_message(result_str, tool_call_id=tc_id))
                    normalized_results.append({
                        "type": "tool_call",
                        "tool_name": "open",
                        "arguments": json.dumps(args),
                        "output": result_str,
                    })
                    logger.info("  open('%s') — %d chars", url[:60], len(result_str))

                # ── manage_context ──────────────────────────
                elif name == "manage_context":
                    prev_summary_id = summary_id_counter
                    result_str, last_boundary_pos, summary_id_counter, _parse_ok = _handle_manage_context(
                        client, hm, question, args, tools, summary_dir,
                        config.runtime.training_data_dir, config.question_id,
                        config.correct_answer,
                        f"{config.runtime.benchmark}_memtool", turn,
                        config.agent.model, config.runtime.benchmark,
                        mem_operations, cm_call_count,
                        last_boundary_pos=last_boundary_pos,
                        summary_id=summary_id_counter,
                        tool_call_id=tc_id,
                        summarizer_client=summarizer_client,
                        drop_call_message=config.runtime.mc_drop_call_message,
                        disable_query_memory=config.runtime.disable_query_memory,
                    )
                    # Only record when a fresh summary was actually generated
                    # (summary_id advanced). Empty-range / error paths return
                    # parse_ok=None and don't bump summary_id_counter.
                    if _parse_ok is not None and summary_id_counter > prev_summary_id:
                        summary_parse_ok.append({
                            "summary_id": summary_id_counter,
                            "parse_ok": _parse_ok,
                        })
                    cm_call_count += 1
                    # Docs prior to this mc are no longer in working memory.
                    in_context_docids.clear()
                    normalized_results.append({
                        "type": "tool_call",
                        "tool_name": "manage_context",
                        "arguments": json.dumps(args),
                        "output": result_str,
                    })

                # ── query_memory ────────────────────────────
                elif name == "query_memory":
                    if config.runtime.disable_query_memory:
                        # re_mem_noquery ablation: the tool is not offered and
                        # archived messages are discarded — never retrieve them,
                        # even if the model hallucinates the call.
                        feedback = ("Error: query_memory is not available. Compressed "
                                    "messages have been discarded and cannot be retrieved; "
                                    "rely on your summaries and on fresh searches.")
                    else:
                        feedback = _handle_query_memory(
                            client, args, summary_dir, question, mem_operations,
                            summarizer_client=summarizer_client)
                    hm.add_message_dict(client.build_tool_message(feedback, tool_call_id=tc_id))
                    normalized_results.append({
                        "type": "tool_call",
                        "tool_name": "query_memory",
                        "arguments": json.dumps(args),
                        "output": feedback,
                    })

                else:
                    hm.add_message_dict(client.build_tool_message(
                        f"Error: unknown tool '{name}'.", tool_call_id=tc_id))
                    logger.warning("  unknown tool '%s'", name)

        else:
            # No tool call — check for truncation
            if (hasattr(client, 'last_finish_reason')
                    and client.last_finish_reason == "length"):
                logger.warning("  turn=%d — truncated, injecting retry", turn)
                hm.add_message("user",
                    "Your previous response was truncated. "
                    "Please retry with a shorter tool call or provide your answer directly.")

        # ── Token accounting ───────────────────────────────
        token_count = client.count_tokens(hm.get_messages_for_model(), tools=tools)
        tokens_before_compression = getattr(hm, '_tokens_before_compression', None)
        if tokens_before_compression is not None:
            raw_delta = tokens_before_compression - _prev_actual
            hm._tokens_before_compression = None
        else:
            raw_delta = token_count - _prev_actual
        _cumulative_raw += max(raw_delta, 0)

        # [CURRENT CONTEXT TOKEN] projection for the NEXT turn: always derive
        # from the actual post-turn message list (which reflects compression
        # immediately) using the same tokenizer/chat-template count we just
        # computed (token_count) plus the max_new_tokens reservation. This
        # makes the marker the model reads next turn a faithful, conservative
        # upper bound on the prompt size of that next generate() call.
        current_context_tokens = token_count + config.agent.rollout.max_new_tokens

        _output_tokens_now = client.snapshot_output_tokens() if hasattr(client, "snapshot_output_tokens") else 0
        _output_tokens_cumulative = _output_tokens_now - _output_tokens_baseline

        token_trajectory.append({
            "turn": turn,
            "tokens_delta": token_count - _prev_actual,
            "tokens_cumulative_raw": _cumulative_raw,
            "tokens_actual": token_count,
            "output_tokens_cumulative": _output_tokens_cumulative,
        })
        _prev_actual = token_count

        if all_tool_calls:
            for tc in all_tool_calls:
                tool_call_log.append({
                    "turn": turn,
                    "tool": tc["name"],
                    "args": tc["arguments"],
                    "tokens_after": token_count,
                })

        logger.info(
            "  turn=%d, tokens=%d, output_tokens=%d",
            turn, token_count, _output_tokens_cumulative,
        )

        # ── End-condition logic ────────────────────────────
        _has_answer = _has_answer_fn(response_text if not all_tool_calls else "")

        # Reset compress counter when context drops below 90%
        if token_count < int(config.agent.context_window * 0.90):
            compress_prompt_count = 0

        # Scaling: suppress premature final answer when below min_turns
        if (_has_answer
                and config.runtime.min_turns > 0
                and turn < config.runtime.min_turns
                and suppressed_answer_count < MAX_SUPPRESSED_ANSWERS):
            suppressed_answer_count += 1
            logger.info(
                "  SCALING: suppressed final answer at turn=%d "
                "(min_turns=%d, suppressed_count=%d)",
                turn, config.runtime.min_turns, suppressed_answer_count,
            )
            continue_msg = BCP_CONTINUE_PROMPT if is_bcp else \
                "Your proposed answer has been noted, but you have not reached the minimum research turns. Continue investigating."
            hm.add_message("user", continue_msg)
            _write_cache(turn)
            continue

        # Normal final answer detection
        if _has_answer:
            final_answer = _extract_answer_fn(response_text)
            confidence = _extract_conf_fn(response_text)
            logger.info("  final_answer: '%s' (conf=%d%%)", final_answer[:80], confidence)
            _write_cache(turn)
            break

        # Scaling/Continue: context nearly full — force compress instead of stopping
        elif (config.runtime.use_memory_tools
              and token_count >= int(config.agent.context_window * 0.95)
              and ((config.runtime.min_turns > 0 and turn < config.runtime.min_turns)
                   or config.initial_history is not None)
              and compress_prompt_count < MAX_COMPRESS_RETRIES):
            compress_prompt_count += 1
            logger.info(
                "  SCALING: context 95%% full at turn=%d, "
                "forcing compress (min_turns=%d, attempt=%d)",
                turn, config.runtime.min_turns, compress_prompt_count,
            )
            compress_msg = BCP_COMPRESS_PROMPT if is_bcp else \
                "WARNING: Your context is nearly full. Call manage_context NOW to free space."
            hm.add_message("user", compress_msg)
            _write_cache(turn)
            continue

        # Context nearly full
        elif token_count >= int(config.agent.context_window * 0.95):
            logger.info("  nearing context window, STOP_PROMPT ...")
            hm.add_message("user", _stop_prompt)
            _g0 = time.perf_counter()
            if _gen_first_start is None:
                _gen_first_start = time.time()
            stop_resp = client.generate(
                hm.get_messages_for_model(), tools=tools, max_new_tokens=config.agent.rollout.max_new_tokens)
            _gen_time_sec += time.perf_counter() - _g0
            _gen_last_end = time.time()
            hm.add_message_dict(client.build_assistant_message(stop_resp))

            stop_text = stop_resp if isinstance(stop_resp, str) else ""
            normalized_results.append({
                "type": "output_text", "tool_name": None,
                "arguments": None, "output": stop_text,
            })

            token_count = client.count_tokens(hm.get_messages_for_model(), tools=tools)
            stop_delta = token_count - _prev_actual
            _cumulative_raw += max(stop_delta, 0)
            _output_tokens_now = client.snapshot_output_tokens() if hasattr(client, "snapshot_output_tokens") else 0
            _output_tokens_cumulative = _output_tokens_now - _output_tokens_baseline
            token_trajectory.append({
                "turn": turn + 1, "tokens_delta": stop_delta,
                "tokens_cumulative_raw": _cumulative_raw,
                "tokens_actual": token_count,
                "output_tokens_cumulative": _output_tokens_cumulative,
            })
            _prev_actual = token_count

            if _has_answer_fn(stop_text):
                final_answer = _extract_answer_fn(stop_text)
                confidence = _extract_conf_fn(stop_text)
            _write_cache(turn)
            break

        _write_cache(turn)
    else:
        status = "max_iterations"  # for-loop walked to max_iter without converging

    return _build_result(turn)
