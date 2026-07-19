"""Single source of truth for run-time configuration.

All defaults live on the dataclasses in this module. No other code (CLI
parsers, phase functions, helpers) is allowed to repeat a default. Inner
layers receive the value from the caller; if a CLI flag is not passed it
becomes `None` and the dataclass default fills in.

Top-level entry: `RunConfig`, composed of `AgentConfig`,
`SummarizerConfig`, `RetrievalConfig`, `RuntimeConfig`, plus per-question
fields the phase function injects per rollout.

Use `load_config(...)` to build a `RunConfig` from dataclass defaults +
optional `configs/default.yaml` + CLI dotlist overrides + explicit named
CLI args.
"""
from .agent import AgentConfig
from .summarizer import SummarizerConfig
from .retrieval import RetrievalConfig
from .runtime import RuntimeConfig
from .run import RunConfig
from .loader import load_config

__all__ = [
    "AgentConfig",
    "SummarizerConfig",
    "RetrievalConfig",
    "RuntimeConfig",
    "RunConfig",
    "load_config",
]
