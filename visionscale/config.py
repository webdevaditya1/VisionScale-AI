"""VisionScale AI Global Configuration System.

Controls pipeline defaults, targets (1080p, 4K), encoding parameters,
and agent execution policies.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ResolutionTarget(str, Enum):
    P1080 = "1080p"
    P4K = "4k"
    ORIGINAL = "original"

    @classmethod
    def from_str(cls, val: str) -> ResolutionTarget:
        normalized = val.strip().lower()
        if normalized in ("4k", "2160p", "uhd"):
            return cls.P4K
        elif normalized in ("1080p", "fhd"):
            return cls.P1080
        elif normalized in ("original", "source"):
            return cls.ORIGINAL
        return cls(normalized)


class ContentCategory(str, Enum):
    AUTO = "auto"
    REAL_WORLD = "real_world"
    ANIME = "anime"
    ARCHIVAL = "archival"


@dataclass
class EncodingConfig:
    """FFmpeg export encoding settings."""
    video_codec: str = "libx264"  # Colab/Kaggle standard h264 for universal browser playback
    audio_codec: str = "copy"      # Preserve original stream without re-encoding when possible
    audio_fallback_codec: str = "aac"
    crf: int = 18                  # Visual lossless range (17-20)
    preset: str = "medium"         # Speed/efficiency balance
    pix_fmt: str = "yuv420p"       # Maximum compatibility for Jupyter/browsers/mobile


@dataclass
class ProcessingConfig:
    """Runtime processing and model parameters."""
    target_resolution: ResolutionTarget = ResolutionTarget.P1080
    content_category: ContentCategory = ContentCategory.AUTO
    enable_face_restoration: bool = False
    enable_interpolation: bool = False
    target_fps: Optional[float] = None  # None = strictly preserve source FPS
    force_cpu: bool = False
    preferred_device: str = "cuda:0"
    use_fp16: bool = True
    batch_size: int = 1
    tile_size: int = 0  # 0 indicates auto-calculated by MemoryManager
    tile_pad: int = 10
    save_temp_frames_to_disk: bool = True
    keep_temp_frames: bool = False
    max_video_duration_seconds: Optional[int] = None  # Optional limit for Colab timeouts


@dataclass
class VisionScaleConfig:
    """Master configuration structure for VisionScale AI."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    workspace_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: dict) -> VisionScaleConfig:
        """Construct config from dictionary with safety defaults."""
        proc_data = data.get("processing", {})
        enc_data = data.get("encoding", {})

        target_res = proc_data.get("target_resolution", "1080p")
        if isinstance(target_res, str):
            target_res = ResolutionTarget.from_str(target_res)
        elif not isinstance(target_res, ResolutionTarget):
            target_res = ResolutionTarget.P1080

        category = proc_data.get("content_category", "auto")
        if isinstance(category, str):
            category = ContentCategory(category.lower())
        elif not isinstance(category, ContentCategory):
            category = ContentCategory.AUTO

        processing = ProcessingConfig(
            target_resolution=target_res,
            content_category=category,
            enable_face_restoration=proc_data.get("enable_face_restoration", False),
            enable_interpolation=proc_data.get("enable_interpolation", False),
            target_fps=proc_data.get("target_fps", None),
            force_cpu=proc_data.get("force_cpu", False),
            preferred_device=proc_data.get("preferred_device", "cuda:0"),
            use_fp16=proc_data.get("use_fp16", True),
            batch_size=proc_data.get("batch_size", 1),
            tile_size=proc_data.get("tile_size", 0),
            tile_pad=proc_data.get("tile_pad", 10),
            save_temp_frames_to_disk=proc_data.get("save_temp_frames_to_disk", True),
            keep_temp_frames=proc_data.get("keep_temp_frames", False),
            max_video_duration_seconds=proc_data.get("max_video_duration_seconds", None),
        )

        encoding = EncodingConfig(
            video_codec=enc_data.get("video_codec", "libx264"),
            audio_codec=enc_data.get("audio_codec", "copy"),
            audio_fallback_codec=enc_data.get("audio_fallback_codec", "aac"),
            crf=enc_data.get("crf", 18),
            preset=enc_data.get("preset", "medium"),
            pix_fmt=enc_data.get("pix_fmt", "yuv420p"),
        )

        ws_dir = data.get("workspace_dir")
        return cls(
            processing=processing,
            encoding=encoding,
            workspace_dir=Path(ws_dir) if ws_dir else None,
        )

    @classmethod
    def from_env(cls) -> VisionScaleConfig:
        """Load overrides from environment variables if present."""
        target_res = os.environ.get("VISIONSCALE_TARGET_RES", "1080p")
        category = os.environ.get("VISIONSCALE_CATEGORY", "auto")
        force_cpu = os.environ.get("VISIONSCALE_FORCE_CPU", "0") in ("1", "true", "True")

        return cls(
            processing=ProcessingConfig(
                target_resolution=ResolutionTarget.from_str(target_res),
                content_category=ContentCategory(category.lower()),
                force_cpu=force_cpu,
            ),
            encoding=EncodingConfig(),
        )

    def compute_target_dimensions(self, input_width: int, input_height: int) -> tuple[int, int]:
        """Compute final output dimensions maintaining exact input aspect ratio.

        Guarantees dimensions are even numbers (required by h264/yuv420p).
        """
        if self.processing.target_resolution == ResolutionTarget.ORIGINAL:
            return input_width, input_height

        aspect_ratio = input_width / float(input_height)

        if self.processing.target_resolution == ResolutionTarget.P1080:
            target_h = 1080
            target_w = int(target_h * aspect_ratio)
        elif self.processing.target_resolution == ResolutionTarget.P4K:
            target_h = 2160
            target_w = int(target_h * aspect_ratio)
        else:
            return input_width, input_height

        # Ensure width and height are even integers for FFmpeg codecs
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        target_h = target_h if target_h % 2 == 0 else target_h + 1
        return target_w, target_h
