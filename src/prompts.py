"""Prompts and answer detection.

Tools are NOT described here. Tool surface lives in `src/tools.py` as JSON
Schemas; vLLM / `tokenizer.apply_chat_template` injects them into a system
message at render time. This file contains only behavioral guidance: 
- header
- context-window hints
- search strategy
- memory management strategy
- answer format.

Public surface:
- build_initial_prompt(benchmark, question, ...) — first user message
  (header, question, context window hints, search strategy, memory management strategy, answer format). No tool descriptions.
- BENCHMARK_PARTS — per-benchmark dispatch (search note, answer format, answer detectors).
- Scaling / runtime-injected prompts: STOP_PROMPT, BCP_STOP_PROMPT,
  BCP_CONTINUE_PROMPT, BCP_COMPRESS_PROMPT, QUERY_MEMORY_PROMPT.
- Answer-detection helpers (XML format and BCP text format).
"""
from __future__ import annotations

import re

# ══════════════════════════════════════════════════════════
# 1. Shared header / instructions (free mode only)
# ══════════════════════════════════════════════════════════

_HEADER = """\
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the tools multiple times."""

_HEADER_WITH_MEMORY = """\
You are a deep research agent. You need to answer the given question by interacting with a search engine and managing your context memory. Your in-context information serves as short-term memory; previously compressed segments live in long-term memory.

After each tool result, the system appends "[CURRENT CONTEXT TOKEN: N]" at the end of the tool response — this tells you your current context usage. Read it; do NOT emit it yourself.

The "manage_context" tool takes no arguments. When you call it, the system compresses everything in your conversation since your previous manage_context call (or since the start of the investigation if this is your first call) up to (but not including) the message that issued the call. The system prompt and the original question are always preserved. The original messages in the compressed range are saved to disk; the tool returns a summary of paths explored, reasoning, and conclusions, prefixed with "[summary_id: N]".

Use the "query_memory" tool to retrieve detailed information from any prior summary's original messages by referencing the summary_id."""

# ── re_mem_noquery ablation header ──────────────────────────
# Same manage_context compression as _HEADER_WITH_MEMORY, but NO query_memory:
# the raw archived messages are unreachable, so no mention of long-term memory
# or a retrieval tool. Selected when use_memory_tools=True AND
# disable_query_memory=True. The original headers above are left untouched.
_HEADER_MEMTOOL_NOQUERY = """\
You are a deep research agent. You need to answer the given question by interacting with a search engine and managing your context. Keep your working context focused: as it fills with material you have already processed, compress the parts you no longer need verbatim into concise summaries.

After each tool result, the system appends "[CURRENT CONTEXT TOKEN: N]" at the end of the tool response — this tells you your current context usage. Read it; do NOT emit it yourself.

The "manage_context" tool takes no arguments. When you call it, the system compresses everything in your conversation since your previous manage_context call (or since the start of the investigation if this is your first call) up to (but not including) the message that issued the call. The system prompt and the original question are always preserved. The compressed messages are replaced by a summary of paths explored, reasoning, and conclusions, prefixed with "[summary_id: N]". Compression is permanent — once compressed, the original messages are gone and only the summary remains, so rely on your summaries and on fresh searches."""

_CONTEXT_WINDOW_LINE = """\
Your context window is {context_window} tokens. Plan your searches accordingly — once context is full the conversation ends immediately. You must provide your answer in the required format before reaching the context limit."""

# Inserted only when use_memory_tools=True; pushes the agent to keep searching,
# compress when stuck, and reach back into prior summaries via query_memory.
_MEMORY_MANAGEMENT_STRATEGY = """\
Strategy:
- If you are not confident in a result, search more — issue additional queries with different terms to corroborate or contradict your best candidate. Do not settle on a low-confidence answer when more searches are still cheap.
- Watch the [CURRENT CONTEXT TOKEN: N] marker at the end of each tool result. When it climbs and your recent rounds contain dead ends or duplicates, call "manage_context" to compress them into a [summary_id: N] entry, freeing space. Do not write this marker in your own responses; it is an environmental signal you consume, not produce.
- If a prior summary looks relevant to a new search direction, call the "query_memory" tool to pull detailed content out of that summary's original messages.
- Calling the "manage_context" tool or the "query_memory" tool does not end the investigation. After these tools return, continue searching — they exist to make room for and surface more evidence, not to wrap up. Only commit to a final answer when you have confident evidence."""

# ── re_mem_noquery ablation strategy ────────────────────────
# Same as _MEMORY_MANAGEMENT_STRATEGY minus the query_memory bullet; bullet on
# not-ending refers only to manage_context. Selected with _HEADER_MEMTOOL_NOQUERY.
_MEMORY_MANAGEMENT_STRATEGY_NOQUERY = """\
Strategy:
- If you are not confident in a result, search more — issue additional queries with different terms to corroborate or contradict your best candidate. Do not settle on a low-confidence answer when more searches are still cheap.
- Watch the [CURRENT CONTEXT TOKEN: N] marker at the end of each tool result. When it climbs and your recent rounds contain dead ends or duplicates, call "manage_context" to compress them into a [summary_id: N] entry, freeing space. Compression is lossy and permanent, so make sure the returned summary captures every fact, docid, and lead you might still need. Do not write this marker in your own responses; it is an environmental signal you consume, not produce.
- Calling the "manage_context" tool does not end the investigation. After it returns, continue searching — it exists to make room for more evidence, not to wrap up. Only commit to a final answer when you have confident evidence."""

# ══════════════════════════════════════════════════════════
# 2. Per-benchmark parts (no tool descriptions — tools come from src/tools.py)
# ══════════════════════════════════════════════════════════

# ── Answer formats ────────────────────────────────────────

_XML_ANSWER_FORMAT = """\
Your response should be in the following format:
<explanation>{{your explanation for your final answer}}</explanation>
<answer>{{your succinct, final answer}}</answer>
<confidence>{{your confidence score between 0 and 100 for your answer}}</confidence>"""


_BCP_ANSWER_FORMAT = """\
Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation \
section only, you should cite your evidence documents inline by enclosing \
their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


# ── Search strategy ──────────────────────────────────────────

_SEARCH_STRATEGY = """\
IMPORTANT: Search snippets are often incomplete. Before answering, always examine at least one full page of content (either by opening a URL when available or by reading the full page text provided in search results)."""

_BCP_SEARCH_STRATEGY = """\
IMPORTANT: Search snippets are often incomplete. Before answering, always retrieve and examine at least one full document using get_document."""


# ── DeepSearchQA prompts ─────────────────────────────────
# DSQA grades against exhaustive answer SETS (65% are Set Answer). The
# agent must enumerate every required item inside a single <answer> tag,
# comma- or newline-separated, otherwise recall drops to zero on a miss.
# answer_format is a callable in BENCHMARK_PARTS — see _dsqa_answer_format.

_DSQA_ANSWER_FORMAT_SINGLE = """\
This question expects a SINGLE answer. Your response should be in the following format:
<explanation>{{your reasoning summarizing the key evidence}}</explanation>
<answer>{{your single, final answer}}</answer>
<confidence>{{your confidence score between 0 and 100}}</confidence>"""

_DSQA_ANSWER_FORMAT_SET = """\
This question expects a SET of answers (multiple distinct items). Your response should be in the following format:
<explanation>{{your reasoning summarizing the key evidence per item}}</explanation>
<answer>{{ALL required items, comma- or newline-separated. Do NOT omit any item — missing items hurt recall. Do NOT add items you cannot verify — extras hurt precision.}}</answer>
<confidence>{{your confidence score between 0 and 100}}</confidence>"""


def _dsqa_answer_format(answer_type: str | None) -> str:
    """Pick DSQA answer format based on per-question answer_type."""
    if answer_type == "Set Answer":
        return _DSQA_ANSWER_FORMAT_SET
    return _DSQA_ANSWER_FORMAT_SINGLE


_DSQA_SEARCH_STRATEGY = """\
IMPORTANT: DeepSearchQA questions span 17 domains and grade against exhaustive answer sets. Search snippets alone are unreliable — issue several diverse queries to surface candidates, then open at least one full page per candidate before committing. For Set Answer questions you must keep searching until you are confident no required item is missing."""


# ══════════════════════════════════════════════════════════
# 3. Answer-detection helpers
# ══════════════════════════════════════════════════════════

def has_final_answer(response_text: str) -> bool:
    """XML format: detect <answer>...</answer>."""
    return "<answer>" in response_text and "</answer>" in response_text


def extract_final_answer(response_text: str) -> str:
    """XML format: last <answer>...</answer> contents."""
    matches = re.findall(
        r"<answer>(.*?)</answer>", response_text, re.IGNORECASE | re.DOTALL
    )
    return matches[-1].strip() if matches else ""


def extract_confidence(response_text: str) -> int:
    """XML format: last <confidence>...</confidence> integer (0-100)."""
    matches = re.findall(
        r"<confidence>\s*(\d+)\s*%?\s*</confidence>", response_text, re.IGNORECASE
    )
    return int(matches[-1]) if matches else 100


def has_bcp_final_answer(response_text: str) -> bool:
    """BCP text format: detect 'Exact Answer:' line."""
    return "Exact Answer:" in response_text


def extract_bcp_final_answer(response_text: str) -> str:
    """BCP text format: text after the last 'Exact Answer:' line.

    Tolerates markdown bold wrapping like `**Exact Answer:** value` by
    eating optional `*` chars between the colon and the actual answer.
    """
    matches = re.findall(
        r"Exact Answer:\s*\**\s*(.+?)(?:\n|$)", response_text, re.DOTALL
    )
    return matches[-1].strip().rstrip("*").strip() if matches else ""


def extract_bcp_confidence(response_text: str) -> int:
    """BCP text format: last 'Confidence: NN' integer."""
    matches = re.findall(
        r"Confidence:\s*(\d+)\s*%?", response_text, re.IGNORECASE
    )
    return int(matches[-1]) if matches else 100


# ══════════════════════════════════════════════════════════
# 4. Benchmark dispatch table
# ══════════════════════════════════════════════════════════

# Each entry: answer_format, search_strategy, and the answer-detection trio
# used at runtime. Tool descriptions live in src/tools.py JSON Schemas;
# the chat template injects them at render time.
BENCHMARK_PARTS: dict[str, dict] = {
    "browsecomp": {
        "answer_format": _XML_ANSWER_FORMAT,
        "search_strategy": _SEARCH_STRATEGY,
        "has_answer": has_final_answer,
        "extract_answer": extract_final_answer,
        "extract_confidence": extract_confidence,
    },
    "browsecomp-plus": {
        "answer_format": _BCP_ANSWER_FORMAT,
        "search_strategy": _BCP_SEARCH_STRATEGY,
        "has_answer": has_bcp_final_answer,
        "extract_answer": extract_bcp_final_answer,
        "extract_confidence": extract_bcp_confidence,
    },
    "deepresearch9k": {
        "answer_format": _BCP_ANSWER_FORMAT,
        "search_strategy": _BCP_SEARCH_STRATEGY,
        "has_answer": has_bcp_final_answer,
        "extract_answer": extract_bcp_final_answer,
        "extract_confidence": extract_bcp_confidence,
    },
    "deepsearchqa": {
        # callable: answer_format(answer_type) → str. build_system_prompt
        # checks callable() to decide whether to invoke or use as-is.
        "answer_format": _dsqa_answer_format,
        "search_strategy": _DSQA_SEARCH_STRATEGY,
        "has_answer": has_final_answer,            # XML <answer>...</answer> — generic helper, not BCP-scoped
        "extract_answer": extract_final_answer,
        "extract_confidence": extract_confidence,
    },
}


# ══════════════════════════════════════════════════════════
# 5. Initial-prompt builder (free mode, single source of truth)
# ══════════════════════════════════════════════════════════

def build_system_prompt(
    benchmark: str,
    *,
    context_window: int = 200_000,
    use_memory_tools: bool = False,
    disable_query_memory: bool = False,
    answer_type: str | None = None,
) -> str:
    """Assemble the system message: all general instructions, no question.

    Tools are NOT described here — the chat template injects their JSON
    Schemas (from src/tools.py) into the system message. Strategy guidance
    is included only when use_memory_tools=True.

    For benchmarks whose answer format varies per question (DSQA: Single
    vs Set Answer), BENCHMARK_PARTS[benchmark]["answer_format"] is a
    callable answer_format(answer_type) → str. Other benchmarks store
    a plain string and answer_type is ignored.

    Caller (runner) should add this as `role:"system"`, then add the
    question as a separate `role:"user"` message.
    """
    parts = BENCHMARK_PARTS.get(benchmark)
    if parts is None:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. "
            f"Available: {list(BENCHMARK_PARTS.keys())}"
        )

    if use_memory_tools:
        header = _HEADER_MEMTOOL_NOQUERY if disable_query_memory else _HEADER_WITH_MEMORY
    else:
        header = _HEADER

    sections = [
        header,
        "",
        _CONTEXT_WINDOW_LINE.format(context_window=context_window),
    ]
    if use_memory_tools:
        sections.append("")
        sections.append(
            _MEMORY_MANAGEMENT_STRATEGY_NOQUERY if disable_query_memory
            else _MEMORY_MANAGEMENT_STRATEGY
        )
    sections.append(parts["search_strategy"])
    sections.append("")
    answer_format = parts["answer_format"]
    if callable(answer_format):
        answer_format = answer_format(answer_type)
    sections.append(answer_format)
    return "\n".join(sections)


def build_initial_prompt(
    benchmark: str,
    question: str,
    *,
    context_window: int = 200_000,
    use_memory_tools: bool = False,
    disable_query_memory: bool = False,
    answer_type: str | None = None,
) -> str:
    """Legacy: combine system instructions + question into one string.

    Kept for src/augment_replay.py which constructs a single user
    message for replay. New code (runner.py) uses build_system_prompt
    + a separate user-role question message.
    """
    system = build_system_prompt(
        benchmark,
        context_window=context_window,
        use_memory_tools=use_memory_tools,
        disable_query_memory=disable_query_memory,
        answer_type=answer_type,
    )
    return f"{system}\n\nQuestion: {question}"


# ══════════════════════════════════════════════════════════
# 6. Scaling / runtime-injected prompts
# ══════════════════════════════════════════════════════════

QUERY_MEMORY_PROMPT = """\
Saved messages under summary_id={summary_id}:

{history}

Recall request — extract content relevant to: {query}

Output format (strict):
- Put any internal reasoning inside <think>...</think> — these will be stripped.
- After </think>, write the recall as compact bullets only. No prose preamble. No restatement of the query. No address to the reader (no "the user", "the agent", "you").
- Each bullet must carry concrete identifiers (docids, URLs, numbers, names) verbatim — never paraphrase numerical evidence.
- Group into up to three sections (omit any that are empty):
  - **Relevant findings:** facts that bear on the query, with supporting docids/URLs.
  - **Dead ends:** queries / docids / hypotheses tried that produced nothing.
  - **Open / unresolved:** most promising direction still to verify.
- If nothing in the saved messages is relevant, output exactly: `(nothing relevant under summary_id={summary_id})`."""

# Used by the manage_context summarizer. Template placeholders:
#   {question}     — the agent's original investigation target
#   {conversation} — the slice of messages being compressed, pre-serialized
_SUMMARY_INSTRUCTION = """\
Original question: {question}

Conversation to compress:

{conversation}

Compress the conversation above into a working-memory entry. This entry
replaces the archived messages in the agent's working context; future
calls to query_memory(summary_id, query) will search this text.

Coverage:
- Knowledge state — facts established with [docid] citations, candidate
  answers and the evidence supporting or contradicting each, hypotheses
  already ruled out with the [docid] that eliminated them, and remaining
  open sub-questions.
- Thoughts — a concise distillation of the most recent reasoning chain
  from the latest assistant turns (what direction the agent has converged
  on and why), AND the concrete next step the agent should take after
  this compression (specific search vocabulary / document to fetch /
  memory query — not a generic "continue searching").

Output exactly one <memory>...</memory> block with the two sections in
this order:

<memory>
## Knowledge state
<facts, candidates, eliminated hypotheses, open sub-questions, in short prose>

## Thoughts
<a few sentences distilling the latest reasoning thread, followed by 1-2
sentences naming the concrete next step>
</memory>

- ≤ 4096 tokens total inside the block, so try to be concise but still cover all the important information.
- Preserve identifiers verbatim — docid, named entities, dates, numbers.
- Do not enumerate every search query or docid you tried.
- After closing the chat-template-opened `</think>`, your very next token
  must be `<memory>`. Do NOT emit any preamble between `</think>` and
  `<memory>` — no `Thinking Process:` heading, no numbered planning list,
  no `Analysis:` / `Plan:` / `Let me ...` lead-in, no re-stated rules,
  no re-quoted question. Plan inside `<think>` if you need to plan;
  the visible output starts directly with `<memory>` and ends with
  `</memory>`."""


# re_mem_noquery variant of _SUMMARY_INSTRUCTION (F1): identical to the above
# except the two query_memory references are removed (the archived messages are
# discarded and cannot be retrieved). Content guidance is otherwise unchanged so
# the compression policy matches the anchor. Selected when disable_query_memory=True.
_SUMMARY_INSTRUCTION_NOQUERY = """\
Original question: {question}

Conversation to compress:

{conversation}

Compress the conversation above into a working-memory entry. This entry
replaces the archived messages in the agent's working context.

Coverage:
- Knowledge state — facts established with [docid] citations, candidate
  answers and the evidence supporting or contradicting each, hypotheses
  already ruled out with the [docid] that eliminated them, and remaining
  open sub-questions.
- Thoughts — a concise distillation of the most recent reasoning chain
  from the latest assistant turns (what direction the agent has converged
  on and why), AND the concrete next step the agent should take after
  this compression (specific search vocabulary / document to fetch —
  not a generic "continue searching").

Output exactly one <memory>...</memory> block with the two sections in
this order:

<memory>
## Knowledge state
<facts, candidates, eliminated hypotheses, open sub-questions, in short prose>

## Thoughts
<a few sentences distilling the latest reasoning thread, followed by 1-2
sentences naming the concrete next step>
</memory>

- ≤ 4096 tokens total inside the block, so try to be concise but still cover all the important information.
- Preserve identifiers verbatim — docid, named entities, dates, numbers.
- Do not enumerate every search query or docid you tried.
- After closing the chat-template-opened `</think>`, your very next token
  must be `<memory>`. Do NOT emit any preamble between `</think>` and
  `<memory>` — no `Thinking Process:` heading, no numbered planning list,
  no `Analysis:` / `Plan:` / `Let me ...` lead-in, no re-stated rules,
  no re-quoted question. Plan inside `<think>` if you need to plan;
  the visible output starts directly with `<memory>` and ends with
  `</memory>`."""


# ── STOP prompts: injected when context window is exhausted ──────────────

STOP_PROMPT = """\
The investigation must now conclude. You have reached the context window limit.

IMPORTANT: Do NOT call any tools. Do NOT attempt any further searches. You MUST respond with text only.

Please review ALL the search results and reasoning from your previous rounds, then provide your best final answer.

You MUST respond in this exact format (text only, no tool calls):
<explanation>{{your explanation summarizing the key evidence}}</explanation>
<answer>{{your succinct, final answer}}</answer>
<confidence>{{your confidence score between 0 and 100}}</confidence>"""


BCP_STOP_PROMPT = """\
The investigation must now conclude. You have reached the context window limit.

IMPORTANT: Do NOT call any tools. Do NOT attempt any further searches. You MUST respond with text only.

Please review ALL the search results and reasoning from your previous rounds, then provide your best final answer.

You MUST respond in this exact format (text only, no tool calls):
Explanation: {your explanation summarizing the key evidence, with [docid] citations}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0 and 100}%"""


DSQA_STOP_PROMPT = """\
The investigation must now conclude. You have reached the context window limit.

IMPORTANT: Do NOT call any tools. Do NOT attempt any further searches. You MUST respond with text only.

Please review ALL the search results and reasoning from your previous rounds, then provide your best final answer.

You MUST respond in this exact format (text only, no tool calls):
<explanation>{your explanation summarizing the key evidence}</explanation>
<answer>{your final answer — for Set Answer questions enumerate EVERY required item inside this single tag, comma- or newline-separated}</answer>
<confidence>{your confidence score between 0 and 100}</confidence>"""


# ── Scaling-mode prompts: injected when min_turns hasn't been reached ────

BCP_CONTINUE_PROMPT = """\
Your proposed answer has been noted, but your investigation is not yet complete. 
You have not reached the minimum number of research turns for this query.

Do NOT provide a final answer yet. Instead, continue your investigation:
- Consider alternative search queries and angles you have not yet tried.
- Look for corroborating or contradicting evidence from different sources.
- If you have already found strong evidence, search for edge cases or exceptions that might change the answer.

Resume your research now. Use the search and get_document tools to gather more evidence."""


BCP_COMPRESS_PROMPT = """\
WARNING: Your context is nearly full, but your investigation is not yet complete. You have not reached the minimum number of research turns for this query.

You MUST immediately call manage_context() to compress everything since your previous manage_context call into a single summary, freeing space. The tool takes no arguments — the system picks the range automatically.

Do NOT provide a final answer yet. Call manage_context() NOW, then resume searching."""
