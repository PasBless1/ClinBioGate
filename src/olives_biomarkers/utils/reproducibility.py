"""Seeding and run provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SeedManager:
    """Sets and records seeds across Python, NumPy and PyTorch."""

    def __init__(self, seed: int = 42, deterministic: bool = True) -> None:
        self.seed = int(seed)
        self.deterministic = deterministic

    def apply(self) -> int:
        """Seed every RNG we rely on and return the seed used."""
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if self.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        return self.seed

    def worker_init_fn(self, worker_id: int) -> None:
        """DataLoader worker seeding so augmentation streams are reproducible."""
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed % (2**32))
        except ImportError:
            pass


@dataclass
class RunMetadata:
    """Everything needed to reproduce a single run."""

    run_id: str
    experiment: str
    seed: int
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform_name: str = field(default_factory=platform.platform)
    device: str = "cpu"
    config: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    manifest_hash: str | None = None

    DEFAULT_PACKAGES = ("numpy", "pandas", "torch", "torchvision", "sklearn", "pyarrow")

    @classmethod
    def collect_package_versions(cls, packages: tuple[str, ...] | None = None) -> dict[str, str]:
        """Record installed versions of packages that can change results."""
        import importlib

        names = packages if packages is not None else cls.DEFAULT_PACKAGES
        versions: dict[str, str] = {}
        for name in names:
            try:
                module = importlib.import_module(name)
                versions[name] = getattr(module, "__version__", "unknown")
            except ImportError:
                versions[name] = "not installed"
        return versions

    @staticmethod
    def current_git_commit(repo_root: str | Path) -> str | None:
        """Short git hash, or None when not in a git repository."""
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    @staticmethod
    def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
        """Stable content hash used to pin a run to an exact data manifest."""
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            chunk = handle.read(chunk_size)
            while chunk:
                digest.update(chunk)
                chunk = handle.read(chunk_size)
        return digest.hexdigest()[:16]

    def to_json(self, path: str | Path) -> Path:
        """Write the metadata beside the run's artefacts."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return out
