"""Runtime environment detection and path resolution.

The same code must run in three places: a local VS Code session, a Colab
runtime driven from VS Code, and a plain Colab notebook. ``RuntimeEnvironment``
is the single place that knows the difference.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeEnvironment:
    """Describes where the code is running and where the data lives.

    Attributes:
        repo_root: Directory holding ``pyproject.toml``; relative paths resolve from here.
        is_colab: Whether the interpreter is a Google Colab runtime.
        device: Best available torch device string.
        drive_mounted: Whether Google Drive has been mounted (Colab only).
    """

    repo_root: Path
    is_colab: bool = False
    device: str = "cpu"
    drive_mounted: bool = False
    details: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def detect(cls, repo_root: str | Path | None = None) -> RuntimeEnvironment:
        """Detect the current runtime and return a populated environment."""
        root = Path(repo_root) if repo_root is not None else cls._find_repo_root()
        env = cls(repo_root=root, is_colab=cls._in_colab(), device=cls._best_device())
        env.details = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "repo_root": str(root),
            "device": env.device,
            "is_colab": str(env.is_colab),
        }
        env.details.update(cls._torch_details())
        return env

    @staticmethod
    def _find_repo_root(start: Path | None = None) -> Path:
        """Walk upwards from ``start`` until a directory with pyproject.toml is found."""
        here = (start or Path.cwd()).resolve()
        for candidate in [here, *here.parents]:
            if (candidate / "pyproject.toml").exists():
                return candidate
        return here

    @staticmethod
    def _in_colab() -> bool:
        return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))

    @staticmethod
    def _best_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _torch_details() -> dict[str, str]:
        try:
            import torch
        except ImportError:
            return {"torch": "not installed"}
        info = {"torch": torch.__version__, "cuda_available": str(torch.cuda.is_available())}
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["gpu_memory_gb"] = format(total, ".1f")
        return info

    # ------------------------------------------------------------------
    # behaviour
    # ------------------------------------------------------------------
    def mount_drive(self, mount_point: str = "/content/drive") -> bool:
        """Mount Google Drive when running on Colab. No-op elsewhere."""
        if not self.is_colab:
            return False
        point = Path(mount_point)
        if point.exists() and any(point.iterdir()):
            self.drive_mounted = True
            return True
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mount_point)
        self.drive_mounted = True
        return True

    def resolve(self, path: str | Path) -> Path:
        """Resolve a possibly-relative path against the repository root."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.repo_root / candidate)

    def resolve_data_root(
        self,
        configured_root: str | Path,
        colab_drive_subpath: str | None = None,
        drive_mount: str = "/content/drive",
    ) -> Path:
        """Pick the data root that actually exists for this runtime.

        Order: an existing repo-relative path, then the Colab Drive location.
        Returns the best candidate even when nothing exists, so callers can raise
        an actionable "expected data at ..." message.
        """
        local = self.resolve(configured_root)
        if local.exists():
            return local
        if self.is_colab and colab_drive_subpath:
            return Path(drive_mount) / colab_drive_subpath
        return local

    def ensure_importable(self) -> None:
        """Put ``<repo_root>/src`` on ``sys.path`` for notebook-driven imports."""
        src = str(self.repo_root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

    def disk_free_gb(self, path: str | Path | None = None) -> float:
        """Free space in GB at ``path`` (defaults to the repository root)."""
        target = self.resolve(path) if path is not None else self.repo_root
        while not target.exists() and target != target.parent:
            target = target.parent
        return shutil.disk_usage(target).free / 1e9

    def git_commit(self) -> str | None:
        """Short git commit hash of the repository, or None outside a git repo."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    def summary(self) -> str:
        """Human-readable one-block summary for notebook headers."""
        rows = dict(self.details)
        rows["free_disk_gb"] = format(self.disk_free_gb(), ".1f")
        rows["git_commit"] = self.git_commit() or "not a git repo"
        return "\n".join(f"{key:>16}: {value}" for key, value in rows.items())
