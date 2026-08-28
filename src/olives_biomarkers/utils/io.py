"""Small I/O helpers shared across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class JsonIO:
    """Read/write JSON with directory creation and NumPy-safe encoding."""

    @staticmethod
    def _default(obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return str(obj)

    @classmethod
    def write(cls, data: Any, path: str | Path, indent: int = 2) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=indent, default=cls._default), encoding="utf-8")
        return out

    @staticmethod
    def read(path: str | Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))


class YamlIO:
    """Read/write YAML configuration files."""

    @staticmethod
    def read(path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        return yaml.safe_load(text) or {}

    @staticmethod
    def write(data: dict[str, Any], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return out


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` as a directory if needed and return it."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
