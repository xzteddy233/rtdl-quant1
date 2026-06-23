from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(config).__name__}")
    return config


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Persist a YAML mapping and return the resolved output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(config), file, sort_keys=False, allow_unicode=True)
    return output.resolve()
