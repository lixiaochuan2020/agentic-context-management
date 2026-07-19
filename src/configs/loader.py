"""Build a fully-resolved RunConfig from dataclass defaults + YAML + CLI overrides.

Merge order (later wins):
  1. RunConfig dataclass defaults  (the source of truth)
  2. YAML file at `yaml_path` (e.g. configs/default.yaml)
  3. dotlist `cli_overrides`        (e.g. ["retrieval.k=20", "agent.model=gpt-5"])

Caller is responsible for translating named CLI flags into dotlist entries
before calling load_config(...). This keeps the loader purely about merging
typed structures.
"""
from __future__ import annotations
import os
from pathlib import Path

from omegaconf import OmegaConf

from .run import RunConfig


def load_config(
    yaml_path: str | os.PathLike | None = None,
    cli_overrides: list[str] | None = None,
) -> RunConfig:
    schema = OmegaConf.structured(RunConfig)

    if yaml_path:
        p = Path(yaml_path)
        if not p.is_file():
            raise FileNotFoundError(f"config yaml not found: {p}")
        yaml_cfg = OmegaConf.load(p)
    else:
        yaml_cfg = OmegaConf.create({})

    dotlist_cfg = OmegaConf.from_dotlist(list(cli_overrides or []))

    merged = OmegaConf.merge(schema, yaml_cfg, dotlist_cfg)
    # `to_object` validates against the dataclass schema and returns a real
    # RunConfig instance (not an OmegaConf DictConfig).
    return OmegaConf.to_object(merged)
