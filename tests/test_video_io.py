"""Integration and unit tests for video probing, frame streaming, reconstruction, and agents."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from visionscale.agents.base_agent import AgentContext
from visionscale.agents.model_selector import ModelSelectorAgent
from visionscale.agents.quality_checker import QualityCheckerAgent
from visionscale.agents.video_analyzer import VideoAnalyzerAgent
from visionscale.agents.video_exporter import VideoExporterAgent
from visionscale.config import ResolutionTarget, VisionScaleConfig
from visionscale.core.hardware import detect_hardware
from visionscale.io.frame_extractor import FrameExtractor, FrameStream
from visionscale.io.video_probe import probe_video
from visionscale.io.video_writer import FFmpegVideoWriter, compile_video_from_frames


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Generate a valid, compact synthetic test video (15 frames, 320x240 @ 30 FPS)."""
    video_path = tmp_path / "test_input.mp4"
    width, height, fps = 320, 240, 30.0
    num_frames = 15

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        for i in range(num_frames):
            # Create a moving gradient bar
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 15) % 255  # Blue channel
            frame[:, :, 1] = 120             # Green channel
            frame[:, :, 2] = 200             # Red channel
            # Draw a circle
            cv2.circle(frame, (20 + i * 15, 120), 30, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()

    assert video_path.exists() and video_path.stat().st_size > 0
    return video_path


def test_probe_video(sample_video: Path):
    meta = probe_video(sample_video)
    assert meta.width == 320
    assert meta.height == 240
    assert abs(meta.fps - 30.0) < 1.0
    assert meta.total_frames >= 14  # Codec container padding may vary slightly
    assert meta.duration_seconds > 0.3
    assert meta.file_size_bytes > 0


def test_frame_stream_and_batches(sample_video: Path):
    stream = FrameStream(sample_video)
    frames = list(stream)
    assert len(frames) >= 14
    idx, frame = frames[0]
    assert idx == 0
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8

    # Test batch iterator
    batch_stream = FrameStream(sample_video).batches(batch_size=4)
    batches = list(batch_stream)
    assert len(batches) >= 3
    assert len(batches[0]) == 4


def test_frame_extractor_to_disk(sample_video: Path, tmp_path: Path):
    out_dir = tmp_path / "disk_frames"
    extractor = FrameExtractor(out_dir)
    extracted = extractor.extract_all_to_disk(sample_video, max_frames=5)
    assert len(extracted) == 5
    for p in extracted:
        assert p.exists()
        assert p.suffix == ".png"

    extractor.clear()
    assert len(list(out_dir.glob("frame_*.*"))) == 0


def test_ffmpeg_video_writer(sample_video: Path, tmp_path: Path):
    out_video = tmp_path / "test_reconstructed.mp4"
    width, height, fps = 320, 240, 30.0

    # Stream frames through FFmpegVideoWriter
    with FFmpegVideoWriter(out_video, width=width, height=height, fps=fps) as writer:
        for idx, frame in FrameStream(sample_video):
            writer.write_frame(frame)

    assert out_video.exists()
    assert out_video.stat().st_size > 1000

    # Probe the written video to ensure compliance
    meta = probe_video(out_video)
    assert meta.width == width
    assert meta.height == height
    assert abs(meta.fps - fps) < 1.0


def test_quality_checker_agent():
    qc = QualityCheckerAgent()
    good_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    # Add texture so variance > 0
    good_frame[0:50, 0:50] = 200
    assert qc.inspect_frame(good_frame, expected_width=1920, expected_height=1080) is True

    # Bad frame: wrong dimensions
    assert qc.inspect_frame(good_frame, expected_width=1280, expected_height=720) is False

    # Bad frame: invalid dtype
    float_frame = good_frame.astype(np.float32)
    assert qc.inspect_frame(float_frame, expected_width=1920, expected_height=1080) is False


def test_agent_pipeline_orchestration(sample_video: Path, tmp_path: Path):
    cfg = VisionScaleConfig()
    cfg.processing.target_resolution = ResolutionTarget.P1080
    hw = detect_hardware()

    context = AgentContext()
    context.set("video_input_path", str(sample_video))

    # 1. Video Analyzer Agent
    analyzer = VideoAnalyzerAgent(cfg)
    context = analyzer.run(context)
    meta = context.get("video_metadata")
    assert meta.width == 320
    target_w, target_h = context.get("target_dimensions")
    # For 320x240 (4:3) targeting 1080p -> target is 1440x1080
    assert target_h == 1080
    assert target_w == 1440

    # 2. Model Selector Agent
    selector = ModelSelectorAgent(cfg, hw)
    context = selector.run(context)
    selected = context.get("selected_models")
    assert len(selected) >= 1
    assert context.get("primary_upscaler_id") in ("realesrgan-x4plus", "realesrgan-x4plus-anime")

    # 3. Video Exporter Agent
    exporter = VideoExporterAgent(cfg, output_dir=tmp_path / "exports")
    context = exporter.run(context)
    out_path = Path(context.get("video_output_path"))
    assert out_path.name.endswith("_upscaled_1080p.mp4")
