"""Video probing and metadata extraction using ffprobe and OpenCV fallback."""

from __future__ import annotations
import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("visionscale.io.probe")


@dataclass
class AudioStreamInfo:
    """Metadata for an audio stream present in the video container."""
    index: int
    codec_name: str
    sample_rate: int
    channels: int
    bit_rate: Optional[int] = None
    language: Optional[str] = None


@dataclass
class VideoMetadata:
    """Comprehensive video properties extracted from source file."""
    path: Path
    width: int
    height: int
    fps: float
    fps_fraction: str  # e.g. "30/1" or "24000/1001"
    total_frames: int
    duration_seconds: float
    video_codec: str
    pixel_format: str
    aspect_ratio: str  # e.g. "16:9", "4:3", or custom
    has_audio: bool
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    file_size_bytes: int = 0
    bit_rate: Optional[int] = None

    @property
    def is_hd(self) -> bool:
        return self.height >= 720

    @property
    def is_1080p(self) -> bool:
        return self.height >= 1080

    @property
    def is_4k(self) -> bool:
        return self.height >= 2160

    def summary(self) -> str:
        mb = self.file_size_bytes / (1024 * 1024)
        audio_str = f"{len(self.audio_streams)} stream(s)" if self.has_audio else "None"
        return (
            f"Resolution: {self.width}x{self.height} | FPS: {self.fps:.3f} | "
            f"Frames: {self.total_frames} | Duration: {self.duration_seconds:.2f}s | "
            f"Codec: {self.video_codec} ({self.pixel_format}) | Audio: {audio_str} | "
            f"Size: {mb:.2f} MB"
        )


def _parse_fractional_fps(fps_str: str) -> float:
    """Convert string fraction like '30000/1001' or '24/1' or decimal '29.97' to float."""
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 30.0
        return float(fps_str)
    except Exception:
        return 30.0


def _probe_via_ffprobe(video_path: Path) -> Optional[VideoMetadata]:
    """Inspect video using ffprobe CLI."""
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream:format",
        "-print_format", "json",
        str(video_path.resolve()),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
        data = json.loads(res.stdout)

        streams = data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video_stream:
            return None

        # Resolution
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # FPS
        fps_fraction = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate") or "30/1"
        fps = _parse_fractional_fps(fps_fraction)

        # Duration
        format_info = data.get("format", {})
        duration_str = video_stream.get("duration") or format_info.get("duration")
        duration = float(duration_str) if duration_str else 0.0

        # Frame count
        nb_frames_str = video_stream.get("nb_frames")
        if nb_frames_str and nb_frames_str.isdigit():
            total_frames = int(nb_frames_str)
        elif duration > 0 and fps > 0:
            total_frames = int(math.ceil(duration * fps))
        else:
            total_frames = 0

        # Audio streams
        audio_streams: list[AudioStreamInfo] = []
        for s in streams:
            if s.get("codec_type") == "audio":
                audio_streams.append(
                    AudioStreamInfo(
                        index=int(s.get("index", 0)),
                        codec_name=s.get("codec_name", "unknown"),
                        sample_rate=int(s.get("sample_rate", 44100)),
                        channels=int(s.get("channels", 2)),
                        bit_rate=int(s.get("bit_rate")) if s.get("bit_rate") else None,
                        language=s.get("tags", {}).get("language"),
                    )
                )

        codec = video_stream.get("codec_name", "unknown")
        pix_fmt = video_stream.get("pix_fmt", "unknown")

        # Aspect ratio
        dar = video_stream.get("display_aspect_ratio")
        if dar and ":" in dar:
            aspect_ratio = dar
        elif width > 0 and height > 0:
            # Simplify GCD
            gcd = math.gcd(width, height)
            aspect_ratio = f"{width // gcd}:{height // gcd}"
        else:
            aspect_ratio = "16:9"

        file_size = int(format_info.get("size", video_path.stat().st_size))
        bit_rate = int(format_info.get("bit_rate")) if format_info.get("bit_rate") else None

        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            fps_fraction=fps_fraction,
            total_frames=total_frames,
            duration_seconds=duration,
            video_codec=codec,
            pixel_format=pix_fmt,
            aspect_ratio=aspect_ratio,
            has_audio=len(audio_streams) > 0,
            audio_streams=audio_streams,
            file_size_bytes=file_size,
            bit_rate=bit_rate,
        )
    except Exception as e:
        logger.debug(f"ffprobe failed: {e}")
        return None


def _probe_via_opencv(video_path: Path) -> VideoMetadata:
    """Fallback inspection using OpenCV VideoCapture."""
    import cv2

    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file with OpenCV: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        duration = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0

        gcd = math.gcd(width, height) if width > 0 and height > 0 else 1
        aspect_ratio = f"{width // gcd}:{height // gcd}" if gcd > 1 else "16:9"
        file_size = video_path.stat().st_size if video_path.exists() else 0

        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            fps_fraction=f"{int(round(fps))}/1",
            total_frames=total_frames,
            duration_seconds=duration,
            video_codec="unknown",
            pixel_format="bgr24",
            aspect_ratio=aspect_ratio,
            has_audio=False,  # OpenCV cannot inspect audio streams
            audio_streams=[],
            file_size_bytes=file_size,
            bit_rate=None,
        )
    finally:
        cap.release()


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Analyze and extract all video container, audio, and visual properties."""
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    # Primary probe using ffprobe
    meta = _probe_via_ffprobe(path)
    if meta is not None and meta.width > 0 and meta.height > 0:
        # If total_frames was 0 or unreadable from container, verify with OpenCV
        if meta.total_frames <= 0:
            try:
                import cv2
                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frames_cv > 0:
                        meta.total_frames = frames_cv
                    cap.release()
            except Exception:
                pass
        return meta

    # Fallback to OpenCV
    logger.info("ffprobe not available or returned empty; using OpenCV fallback.")
    return _probe_via_opencv(path)
