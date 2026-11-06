"""Platform detection and workspace path management for Google Colab, Kaggle, and Local."""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class PlatformType(str, Enum):
    COLAB = "colab"
    KAGGLE = "kaggle"
    LOCAL = "local"


@dataclass(frozen=True)
class PlatformPaths:
    """Standardized working directories for model cache, inputs, outputs, and frames."""
    base_dir: Path
    models_dir: Path
    input_dir: Path
    output_dir: Path
    frames_dir: Path
    checkpoints_dir: Path

    def ensure_directories(self) -> None:
        """Create all required directories if they don't already exist."""
        for path in (
            self.base_dir,
            self.models_dir,
            self.input_dir,
            self.output_dir,
            self.frames_dir,
            self.checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class PlatformEnvironment:
    """Encapsulates execution platform details and filesystem layouts."""

    def __init__(self, platform_type: PlatformType, paths: PlatformPaths) -> None:
        self.platform_type = platform_type
        self.paths = paths
        self.paths.ensure_directories()

    @property
    def is_colab(self) -> bool:
        return self.platform_type == PlatformType.COLAB

    @property
    def is_kaggle(self) -> bool:
        return self.platform_type == PlatformType.KAGGLE

    @property
    def is_local(self) -> bool:
        return self.platform_type == PlatformType.LOCAL

    def __repr__(self) -> str:
        return f"<PlatformEnvironment platform={self.platform_type.value} base_dir='{self.paths.base_dir}'>"


def detect_platform() -> PlatformType:
    """Identify whether the environment is Google Colab, Kaggle, or a Local workstation."""
    # Check Google Colab
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return PlatformType.COLAB

    # Check Kaggle
    if (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.path.exists("/kaggle/working")
    ):
        return PlatformType.KAGGLE

    return PlatformType.LOCAL


def get_workspace_paths(
    platform: Optional[PlatformType] = None,
    custom_base_dir: Optional[str | Path] = None,
) -> PlatformPaths:
    """Compute and return standardized workspace paths based on the active platform."""
    detected = platform or detect_platform()

    if custom_base_dir is not None:
        base = Path(custom_base_dir).resolve()
    elif detected == PlatformType.COLAB:
        base = Path("/content/visionscale_work")
    elif detected == PlatformType.KAGGLE:
        base = Path("/kaggle/working/visionscale_work")
    else:
        # Default local workspace
        base = Path.cwd() / "visionscale_workspace"

    # Temporary frames can be in /tmp or system temp to leverage fast local storage
    if detected in (PlatformType.COLAB, PlatformType.KAGGLE):
        frames_dir = Path("/tmp/visionscale_frames")
    else:
        frames_dir = base / "temp_frames"

    return PlatformPaths(
        base_dir=base,
        models_dir=base / "weights",
        input_dir=base / "inputs",
        output_dir=base / "outputs",
        frames_dir=frames_dir,
        checkpoints_dir=base / "checkpoints",
    )


def initialize_platform(custom_base_dir: Optional[str | Path] = None) -> PlatformEnvironment:
    """Convenience factory to detect platform and ensure all workspace directories are ready."""
    plt_type = detect_platform()
    paths = get_workspace_paths(platform=plt_type, custom_base_dir=custom_base_dir)
    return PlatformEnvironment(platform_type=plt_type, paths=paths)
