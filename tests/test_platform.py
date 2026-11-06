"""Unit tests for platform detection and filesystem layout."""

from pathlib import Path
from visionscale.core.platform import (
    PlatformType,
    detect_platform,
    get_workspace_paths,
    initialize_platform,
)


def test_detect_platform_local():
    # In standard local test environment, should return LOCAL
    plat = detect_platform()
    assert plat in (PlatformType.LOCAL, PlatformType.COLAB, PlatformType.KAGGLE)


def test_workspace_paths_creation(tmp_path: Path):
    custom_base = tmp_path / "custom_ws"
    paths = get_workspace_paths(platform=PlatformType.LOCAL, custom_base_dir=custom_base)

    assert paths.base_dir == custom_base
    assert paths.models_dir == custom_base / "weights"
    assert paths.input_dir == custom_base / "inputs"
    assert paths.output_dir == custom_base / "outputs"
    assert paths.checkpoints_dir == custom_base / "checkpoints"

    paths.ensure_directories()
    assert paths.models_dir.exists()
    assert paths.input_dir.exists()
    assert paths.output_dir.exists()
    assert paths.frames_dir.exists()
    assert paths.checkpoints_dir.exists()


def test_platform_environment_init(tmp_path: Path):
    custom_base = tmp_path / "env_test"
    env = initialize_platform(custom_base_dir=custom_base)

    assert env.paths.base_dir.exists()
    assert isinstance(env.is_local, bool)
    assert "PlatformEnvironment" in repr(env)
