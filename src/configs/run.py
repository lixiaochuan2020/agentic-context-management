from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .agent import AgentConfig
from .summarizer import SummarizerConfig
from .retrieval import RetrievalConfig
from .runtime import RuntimeConfig


@dataclass
class RunConfig:
    """Top-level rollout config = 4 sub-configs + per-question fields.

    Sub-configs (agent / summarizer / retrieval / runtime) hold every
    rollout-wide default. Per-question fields (question_id, correct_answer,
    cache_path, initial_history) are injected by phase functions via
    `dataclasses.replace(base_config, question_id=..., ...)` inside the
    per-question loop.
    """

    agent: AgentConfig = field(default_factory=AgentConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Per-question fields — replaced per rollout. Not user-configurable via YAML.
    question_id: str = ""
    correct_answer: str = ""
    cache_path: Optional[str] = None
    initial_history: Optional[list[dict]] = None
    # If a caller (e.g. teacher_guided_rollout) preloads initial_history that
    # has had compression applied off-trajectory, it can pass the unmodified
    # raw history here so hm.raw_messages keeps the full archive.
    initial_raw_history: Optional[list[dict]] = None
    # Caller may set the starting last_boundary_pos and summary_id counter
    # (e.g. to continue from an initial trajectory that already had mc calls).
    # If None, runner uses defaults: boundary = len(hm.messages), summary_id
    # auto-scanned from workspace + history markers.
    initial_last_boundary_pos: Optional[int] = None
    initial_summary_id: Optional[int] = None
