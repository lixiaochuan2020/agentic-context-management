"""Dataset loaders for each benchmark.

Each loader normalizes data to a standard format:
    [{id: str, question: str, answer: str, ...}]

Usage:
    from src.data_loading import load_data
    dataset = load_data("browsecomp-plus", "data/bcp_100.json")
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Registry: benchmark name -> loader function
_LOADERS: dict[str, callable] = {}


def register(benchmark: str):
    """Decorator to register a loader for a benchmark."""
    def wrapper(fn):
        _LOADERS[benchmark] = fn
        return fn
    return wrapper


def load_data(benchmark: str, data_path: str, **kwargs) -> list[dict]:
    """Load dataset for a benchmark.

    Args:
        benchmark: benchmark name (must be registered)
        data_path: path to the data file
        **kwargs: benchmark-specific options (e.g., jsonl_path for BCP)

    Returns:
        List of dicts, each with at least {id, question, answer}.
    """
    loader = _LOADERS.get(benchmark)
    if loader is None:
        raise ValueError(
            f"No loader registered for benchmark '{benchmark}'. "
            f"Available: {sorted(_LOADERS.keys())}"
        )
    dataset = loader(data_path, **kwargs)
    logger.info("Loaded %d questions for benchmark '%s' from %s", len(dataset), benchmark, data_path)
    return dataset


# Import loaders so they register themselves
from src.data_loading import browsecomp, browsecomp_plus  # noqa: E402, F401
