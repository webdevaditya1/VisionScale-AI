"""Unit tests for configuration system and target dimension calculations."""

from visionscale.config import (
    ContentCategory,
    ResolutionTarget,
    VisionScaleConfig,
)


def test_config_defaults():
    cfg = VisionScaleConfig()
    assert cfg.processing.target_resolution == ResolutionTarget.P1080
    assert cfg.processing.content_category == ContentCategory.AUTO
    assert cfg.processing.enable_face_restoration is False
    assert cfg.processing.enable_interpolation is False
    assert cfg.processing.target_fps is None  # Must preserve source FPS
    assert cfg.encoding.video_codec == "libx264"
    assert cfg.encoding.audio_codec == "copy"
    assert cfg.encoding.crf == 18


def test_compute_target_dimensions_1080p():
    cfg = VisionScaleConfig()
    cfg.processing.target_resolution = ResolutionTarget.P1080

    # 16:9 720p input -> 1920x1080
    tw, th = cfg.compute_target_dimensions(1280, 720)
    assert (tw, th) == (1920, 1080)

    # 4:3 640x480 input -> 1440x1080
    tw, th = cfg.compute_target_dimensions(640, 480)
    assert (tw, th) == (1440, 1080)
    # Both must be even numbers
    assert tw % 2 == 0
    assert th % 2 == 0


def test_compute_target_dimensions_4k():
    cfg = VisionScaleConfig()
    cfg.processing.target_resolution = ResolutionTarget.P4K

    # 16:9 1080p input -> 3840x2160
    tw, th = cfg.compute_target_dimensions(1920, 1080)
    assert (tw, th) == (3840, 2160)

    # 21:9 ultrawide 2560x1080 -> height 2160, width 5120
    tw, th = cfg.compute_target_dimensions(2560, 1080)
    assert th == 2160
    assert tw % 2 == 0


def test_config_from_dict():
    data = {
        "processing": {
            "target_resolution": "4K",
            "content_category": "anime",
            "enable_face_restoration": True,
            "use_fp16": False,
        },
        "encoding": {
            "crf": 16,
            "preset": "slow",
        },
    }
    cfg = VisionScaleConfig.from_dict(data)
    assert cfg.processing.target_resolution == ResolutionTarget.P4K
    assert cfg.processing.content_category == ContentCategory.ANIME
    assert cfg.processing.enable_face_restoration is True
    assert cfg.processing.use_fp16 is False
    assert cfg.encoding.crf == 16
    assert cfg.encoding.preset == "slow"
