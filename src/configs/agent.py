from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RolloutConfig:
    """Per-turn rollout parameters that interact with the vLLM token budget."""

    # Output reservation per generate() call. vLLM enforces
    #   prompt_tokens + max_new_tokens <= max_model_len
    # so the effective prompt cap is max_model_len - max_new_tokens.
    # The [CURRENT CONTEXT TOKEN: N] marker we show the model adds this
    # value to the actually-used tokens so N is a conservative upper
    # bound the model can compare directly against context_window.
    max_new_tokens: int = 16_384


@dataclass
class AgentConfig:
    """The model under evaluation / training (the deep-research agent itself)."""

    model: str = "gpt-5.4-mini"
    api_base: Optional[str] = None
    context_window: int = 200_000
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
