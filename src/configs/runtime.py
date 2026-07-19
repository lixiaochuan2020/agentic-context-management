from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeConfig:
    """Per-rollout runtime limits and modes."""

    # "browsecomp" | "browsecomp-plus" | "deepresearch9k"
    benchmark: str = "browsecomp-plus"

    # Hard cap on agent turns (each turn = one assistant generation).
    max_iterations: int = 100

    # When True: expose manage_context + query_memory tools and create a
    # per-question workspace for summary_*.json offload files.
    use_memory_tools: bool = False

    # When True (and use_memory_tools=True): the "re_mem_noquery" ablation —
    # keep the agent-initiated manage_context compression, but drop the
    # query_memory tool so the raw archived messages behind each summary are
    # unreachable (the summary is all that remains). Uses the no-query prompt
    # variants (no mention of long-term memory / query_memory). Isolates
    # retrieval fidelity from the compression policy for the reviewer ablation.
    disable_query_memory: bool = False

    # Root of per-question workspaces (workspace_root/{qid}/summary_*.json).
    # When None, no workspace is created (and memory tools are disabled even
    # if use_memory_tools=True).
    workspace_root: Optional[str] = None

    # Root for per-question training-data captures (training_data_dir/{qid}.json).
    training_data_dir: Optional[str] = None

    # Force at least this many turns before allowing a final answer (scaling
    # / continue mode).
    min_turns: int = 0

    # When True, manage_context can compress the entire initial_history (from
    # message 1), not just messages produced after it. Used by teacher-guided
    # rollouts where the iter_N transcript is treated as fully compressible.
    compress_initial_history: bool = False

    # When True, manage_context also includes the assistant turn that issued
    # the mc call in the compressed range — its <think> reasoning becomes part
    # of the summarizer input and the message itself is deleted along with
    # the rest of the range. Only the summary `tool` message remains in live
    # history, avoiding per-call (asst_tool_call, tool_result) residue. The
    # tool result keeps its original tool_call_id; Qwen3-style chat templates
    # render `tool` role messages as <tool_response>...</tool_response> inside
    # a synthesized user turn and do not require the originating assistant.
    mc_drop_call_message: bool = True

    # When True, append an explicit reminder to the SWE system prompt that
    # the model must edit at least one source file via str_replace_editor
    # (with command=str_replace or insert) before calling submit_patch.
    # Added to recover SWE-bench rollouts that previously submitted empty
    # diffs because the SFT'd model investigated extensively but never
    # attempted an edit. Off by default — opt in per-run.
    edit_reminder: bool = False
