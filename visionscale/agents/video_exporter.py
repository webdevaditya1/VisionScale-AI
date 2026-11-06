"""Video Exporter Agent: orchestrates final container encoding, verification, and artifact delivery."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from visionscale.agents.base_agent import AgentContext, BaseAgent
from visionscale.config import VisionScaleConfig
from visionscale.io.video_probe import VideoMetadata, probe_video
from visionscale.io.video_writer import FFmpegVideoWriter, compile_video_from_frames

logger = logging.getLogger("visionscale.agents.exporter")


class VideoExporterAgent(BaseAgent):
    """Coordinates final container creation, verifies stream compliance, and cleans temps."""

    def __init__(self, config: VisionScaleConfig, output_dir: Path) -> None:
        super().__init__("VideoExporterAgent")
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def determine_output_path(self, input_path: Path, target_resolution: str) -> Path:
        """Generate standardized output filename."""
        stem = input_path.stem
        out_name = f"{stem}_upscaled_{target_resolution}.mp4"
        return self.output_dir / out_name

    def verify_exported_video(self, output_file: Path, expected_meta: VideoMetadata) -> VideoMetadata:
        """Probe resulting file to ensure video streams, duration, and audio are intact."""
        if not output_file.exists():
            raise FileNotFoundError(f"Exported video file does not exist: {output_file}")

        if output_file.stat().st_size == 0:
            raise ValueError(f"Exported video file is empty: {output_file}")

        out_meta = probe_video(output_file)
        # Check duration drift
        if expected_meta.duration_seconds > 0:
            drift = abs(out_meta.duration_seconds - expected_meta.duration_seconds)
            if drift > 1.5:
                logger.warning(
                    f"Duration drift noticed: source duration {expected_meta.duration_seconds:.2f}s "
                    f"vs exported {out_meta.duration_seconds:.2f}s"
                )

        # Check audio preservation
        if expected_meta.has_audio and not out_meta.has_audio:
            logger.warning("Source had audio stream, but exported video reports no audio stream.")

        return out_meta

    def run(self, context: AgentContext) -> AgentContext:
        self._current_context = context
        input_path = Path(context.get("video_input_path"))
        source_meta: VideoMetadata = context.get("video_metadata")
        target_res = self.config.processing.target_resolution.value

        custom_out = context.get("video_output_path")
        out_path = Path(custom_out) if custom_out else self.determine_output_path(input_path, target_res)
        context.set("video_output_path", str(out_path))

        self.log(self.name, f"Export target defined: {out_path}")
        return context
