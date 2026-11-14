"""End-to-end, frame-streamed video enhancement pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from visionscale.agents.base_agent import AgentContext
from visionscale.agents.model_selector import ModelSelectorAgent
from visionscale.agents.video_analyzer import VideoAnalyzerAgent
from visionscale.agents.video_exporter import VideoExporterAgent
from visionscale.config import VisionScaleConfig
from visionscale.core.hardware import HardwareInfo, detect_hardware
from visionscale.core.memory import MemoryManager
from visionscale.core.platform import initialize_platform
from visionscale.io.frame_extractor import FrameStream
from visionscale.io.video_writer import FFmpegVideoWriter
from visionscale.models.realesrgan import RealESRGANEnhancer
from visionscale.models.registry import MODEL_REGISTRY

logger = logging.getLogger("visionscale.pipeline")

ProgressCallback = Callable[[int, int], None]


def enhance_video(
    input_path: str | Path,
    config: Optional[VisionScaleConfig] = None,
    output_dir: Optional[str | Path] = None,
    hardware: Optional[HardwareInfo] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Enhance a video with Real-ESRGAN and stream the result to an MP4 file."""
    source_path = Path(input_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    active_config = config or VisionScaleConfig()
    active_hardware = hardware or detect_hardware()
    device = "cpu" if active_config.processing.force_cpu else active_hardware.recommended_device
    use_fp16 = (
        active_config.processing.use_fp16
        and device.startswith("cuda")
        and active_hardware.recommended_fp16
    )
    platform = initialize_platform(custom_base_dir=active_config.workspace_dir)
    weights_dir = platform.paths.models_dir
    destination_dir = Path(output_dir) if output_dir else platform.paths.output_dir

    context = AgentContext()
    context.set("video_input_path", str(source_path))
    context = VideoAnalyzerAgent(active_config).run(context)
    context = ModelSelectorAgent(active_config, active_hardware).run(context)

    if context.get("face_restoration_model_id") or context.get("interpolation_model_id"):
        raise NotImplementedError(
            "Face restoration and frame interpolation are not implemented in the video inference pipeline."
        )

    model_id = context.get("primary_upscaler_id")
    if model_id not in ("realesrgan-x4plus", "realesrgan-x4plus-anime"):
        raise ValueError(f"No inference implementation is available for model '{model_id}'.")

    target_width, target_height = context.get("target_dimensions")
    metadata = context.get("video_metadata")
    selector = VideoExporterAgent(active_config, destination_dir)
    context = selector.run(context)
    output_path = Path(context.get("video_output_path"))

    model = RealESRGANEnhancer(
        metadata=MODEL_REGISTRY[model_id],
        device=device,
        use_fp16=use_fp16,
        weights_dir=weights_dir,
    )
    memory_manager = MemoryManager(device)
    profile = memory_manager.compute_profile(
        frame_width=metadata.width,
        frame_height=metadata.height,
        scale=RealESRGANEnhancer.UPSCALE_FACTOR,
        prefer_fp16=use_fp16,
    )
    tile_size = active_config.processing.tile_size or profile.tile_size
    tile_pad = active_config.processing.tile_pad
    max_frames = None
    if active_config.processing.max_video_duration_seconds is not None:
        max_frames = max(
            1,
            int(metadata.fps * active_config.processing.max_video_duration_seconds),
        )
    expected_frames = min(metadata.total_frames, max_frames) if max_frames else metadata.total_frames

    frames_written = 0
    try:
        last_reported_percent = -1

        def report_weight_progress(percentage: float, message: str) -> None:
            nonlocal last_reported_percent
            rounded_percentage = int(percentage)
            if rounded_percentage >= last_reported_percent + 10 or rounded_percentage == 100:
                logger.info("%s (%.1f%%)", message, percentage)
                last_reported_percent = rounded_percentage

        model.ensure_weights(progress_callback=report_weight_progress)
        model.load()
        with FFmpegVideoWriter(
            output_path=output_path,
            width=target_width,
            height=target_height,
            fps=metadata.fps,
            source_audio_path=source_path,
            encoding_config=active_config.encoding,
        ) as writer:
            for frame_index, frame in FrameStream(source_path, max_frames=max_frames):
                current_tile = tile_size
                while True:
                    try:
                        enhanced = model.enhance_frame(
                            frame,
                            tile_size=current_tile,
                            tile_pad=tile_pad,
                        )
                        break
                    except RuntimeError as exc:
                        if (
                            not device.startswith("cuda")
                            or "out of memory" not in str(exc).lower()
                            or current_tile <= 128
                        ):
                            raise
                        current_tile = max(128, int(current_tile * 0.7))
                        logger.warning(
                            "CUDA out of memory on frame %d; retrying with tile size %d.",
                            frame_index,
                            current_tile,
                        )
                        memory_manager.clear_cache()

                if enhanced.shape[:2] != (target_height, target_width):
                    import cv2

                    enhanced = cv2.resize(
                        enhanced,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                if enhanced.dtype != np.uint8 or enhanced.ndim != 3 or enhanced.shape[2] != 3:
                    raise ValueError(f"Enhancer returned an invalid frame at index {frame_index}.")

                writer.write_frame(enhanced)
                frames_written += 1
                if progress_callback:
                    progress_callback(frames_written, expected_frames)

        if frames_written == 0:
            raise RuntimeError(f"No frames were decoded from {source_path}.")

        output_metadata = selector.verify_exported_video(output_path, metadata)
        if (output_metadata.width, output_metadata.height) != (target_width, target_height):
            raise RuntimeError(
                f"Export resolution mismatch: expected {target_width}x{target_height}, "
                f"got {output_metadata.width}x{output_metadata.height}."
            )
        return output_path
    finally:
        model.unload()
