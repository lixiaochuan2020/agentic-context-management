"""Teacher annotation prompt for the v5_gpt5_teacher pipeline.

The teacher's only job is to identify the EARLIEST point in a ReAct trajectory
where the student should have invoked `manage_context` (mc) or `query_memory`
(qm), and to write a first-person `<think>` rationale for that call.

See `experiments/20260517.md` § Exp2 for the full design.
"""
from __future__ import annotations

import json


_TEACHER_ANNOTATE_INSTRUCTION = """\
You are reviewing an agent's research trajectory on a hard fact-finding question. The agent
failed: either it produced a wrong answer, exhausted its turn budget, or its working context
overflowed before reaching an answer.

Your single job: identify the EARLIEST point in the trajectory where the agent should have
invoked a context-management operation — either `manage_context` (compress working memory)
or `query_memory` (retrieve from prior summaries) — and write the first-person rationale the
agent will appear to have produced for that call.

# QUESTION
{question}

# GOLDEN ANSWER  (FOR YOUR REFERENCE ONLY — NEVER reveal or hint at this)
{correct_answer}

# GOLDEN EVIDENCE  (FOR YOUR REFERENCE ONLY — never cite docids from these, quote, or paraphrase)
{gold_docs_block}

# AGENT TRAJECTORY
Each message is prefixed with `[id=N, role=R]`. Assistant `tool_calls` and tool responses are
inlined under the relevant message.

{numbered_history}

# WORKSPACE STATE
summary_ids currently stored from prior manage_context calls: {summary_ids}

# AVAILABLE ACTIONS

- `mc` — compress all messages since the last compression boundary into a summary stored in
  long-term memory. Use when:
    * the agent is cycling through repeated queries without finding new evidence
    * long tool responses (search results, fetched documents) accumulated in working context
      are no longer load-bearing
    * working context is approaching token saturation
- `qm` — query a prior summary by natural-language query. ONLY VALID IF the summary_ids list
  above is non-empty.
- `no_action_needed` — emit this if no mc/qm intervention prior to the trajectory's failure
  point would have materially helped.

# CONSTRAINTS ON `after_id`

- Must satisfy `0 <= after_id <= last_id`.
- The message at id=after_id must have role in {{system, user, tool}}. Inserting after an
  assistant turn that has pending tool_calls would orphan those calls.
- Pick the SMALLEST valid `after_id` where the action would materially help. If multiple
  positions are equally valid, pick the earliest.

# CONSTRAINTS ON `qm`

- If summary_ids above is empty, you MUST NOT choose `qm`. Pick `mc` or `no_action_needed`.

# OUTPUT FORMAT — STRICT JSON, ONE OBJECT, NO PROSE BEFORE OR AFTER

```json
{{
  "decision": "mc" | "qm" | "no_action_needed",
  "after_id": <integer; use -1 if decision is no_action_needed>,
  "think": "<first-person rationale, see rules below>",
  "qm_query": "<natural-language query; required iff decision='qm'>"
}}
```

# RULES FOR `think`

`think` is the first-person reasoning the agent will appear to have produced just before
calling mc/qm. It must read as authentic agent reasoning.

REQUIREMENTS:
1. First-person only ("I notice…", "my searches…", "my working context…"). No third-person
   self-reference, no naming or alluding to any external party.
2. Ground the rationale ONLY in observable trajectory signals, e.g.:
    * the same or near-identical query keywords repeated across multiple search turns
    * accumulated input tokens / context-window pressure
    * the last N turns produced no new docids and no new evidence
    * the retrieved docid set is small and cycling
    * (for qm only) a relevant earlier summary exists that has not been re-queried
3. STRICTLY FORBIDDEN:
    * mentioning, naming, or describing any part of the golden answer
    * naming any docid the agent has NOT retrieved in this trajectory
    * any of these words: `coach`, `coaches`, `coaching`, `feedback`, `review`, `reviewer`,
      `reviewed`, `external`, `advised`, `advisor`, `guidance`, `told me`, `someone said`,
      `instructed`
4. Length: between 200 and 1200 characters.

Output the JSON object only. No preamble, no closing remarks, no prose outside the JSON.
"""


def render_gold_docs_block(gold_docs, max_chars_per_doc: int = 2000) -> str:
    """Render gold_docs as a flat block: docid + url + first ~512 tokens of text."""
    if not gold_docs:
        return "(no gold docs available for this question)"
    parts = []
    for d in gold_docs:
        docid = d.get("docid", "?")
        url = (d.get("url") or "").strip()
        text = (d.get("text") or "").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "\n... [truncated]"
        head = f"docid={docid}"
        if url:
            head += f"\nurl: {url}"
        parts.append(f"{head}\ntext:\n{text}")
    return "\n\n---\n\n".join(parts)


def render_numbered_history(history, max_tool_chars: int = 4000) -> str:
    """Render a chat history with `[id=N, role=R]` prefixes for teacher annotation.

    Tool messages (search results / fetched docs) are truncated to `max_tool_chars`
    characters to keep the prompt bounded. Assistant messages render both visible
    content and any structured `tool_calls`.
    """
    lines: list[str] = []
    for i, m in enumerate(history):
        role = m.get("role", "?")
        head = f"[id={i}, role={role}]"
        if role == "assistant":
            content = (m.get("content") or "").strip()
            tool_calls = m.get("tool_calls") or []
            body: list[str] = []
            if content:
                body.append(content)
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name") or "?"
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        pass
                body.append(f"tool_call: {name}({json.dumps(args, ensure_ascii=False)})")
            lines.append(f"{head}\n" + ("\n".join(body) if body else "(empty)"))
        elif role == "tool":
            content = (m.get("content") or "").strip()
            tool_name = m.get("name") or ""
            if len(content) > max_tool_chars:
                content = content[:max_tool_chars] + (
                    f"\n... [truncated {len(content) - max_tool_chars} chars]"
                )
            tag = head + (f" name={tool_name}" if tool_name else "")
            lines.append(f"{tag}\n{content}")
        else:
            content = m.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(
                    (b.get("text") or str(b)) if isinstance(b, dict) else str(b)
                    for b in content
                )
            lines.append(f"{head}\n{content}")
    return "\n\n".join(lines)
