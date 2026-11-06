"""FFmpeg-based video reconstruction preserving FPS, audio, duration, and color metadata."""

from __future__ import annotations
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np

from visionscale.config import EncodingConfig
from visionscale.io.video_probe import VideoMetadata, probe_video

logger = logging.getLogger("visionscale.io.writer")


class FFmpegVideoWriter:
    """Streams processed numpy frames directly into an active FFmpeg process via stdin pipe."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        source_audio_path: Optional[str | Path] = None,
        encoding_config: Optional[EncodingConfig] = None,
    ) -> None:
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.source_audio_path = Path(source_audio_path).resolve() if source_audio_path else None
        self.cfg = encoding_config or EncodingConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._frames_written = 0

        self._start_ffmpeg()

    def _start_ffmpeg(self) -> None:
        """Launch FFmpeg with rawvideo stdin pipe and target encoding settings."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg executable not found in system PATH.")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",  # Read frames from stdin pipe
        ]

        # Audio stream inclusion if source video provided
        has_audio = False
        if self.source_audio_path and self.source_audio_path.exists():
            try:
                meta = probe_video(self.source_audio_path)
                if meta.has_audio:
                    cmd.extend(["-i", str(self.source_audio_path)])
                    has_audio = True
            except Exception as e:
                logger.warning(f"Failed to probe audio from source: {e}")

        # Map streams
        if has_audio:
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
            # Try copy audio codec to prevent lossy re-encoding; FFmpeg will fallback if needed
            cmd.extend(["-c:a", self.cfg.audio_codec])
            # Shortest to avoid desync
            cmd.extend(["-shortest"])
        else:
            cmd.extend(["-map", "0:v:0"])

        # Video encoding settings
        cmd.extend([
            "-c:v", self.cfg.video_codec,
            "-preset", self.cfg.preset,
            "-crf", str(self.cfg.crf),
            "-pix_fmt", self.cfg.pix_fmt,
            "-movflags", "+faststart",  # Enable instant web playback
            str(self.output_path),
        ])

        logger.debug(f"Launching FFmpeg: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        """Write a single BGR frame into the FFmpeg stdin pipe."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("FFmpeg process is not running or stdin is closed.")

        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            raise ValueError(
                f"Frame size {frame_bgr.shape[1]}x{frame_bgr.shape[0]} does not match "
                f"writer configuration {self.width}x{self.height}."
            )

        try:
            self._proc.stdin.write(frame_bgr.tobytes())
            self._frames_written += 1
        except (BrokenPipeError, IOError) as e:
            stderr = self._proc.stderr.read().decode() if self._proc.stderr else ""
            raise RuntimeError(f"FFmpeg stdin pipe broken. Stderr: {stderr}") from e

    def close(self) -> None:
        """Flush stdin pipe, terminate FFmpeg process, and verify output."""
        if self._proc is not None:
            if self._proc.stdin:
                try:
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
                except Exception:
                    pass
            stdout, stderr = self._proc.communicate()
            return_code = self._proc.returncode
            self._proc = None

            if return_code != 0:
                err_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"FFmpeg encoding failed with exit code {return_code}: {err_msg}")

            logger.info(f"Successfully encoded {self._frames_written} frames to {self.output_path}")

    def __enter__(self) -> FFmpegVideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def compile_video_from_frames(
    frames_dir: str | Path,
    output_path: str | Path,
    fps: float,
    source_audio_path: Optional[str | Path] = None,
    encoding_config: Optional[EncodingConfig] = None,
    frame_pattern: str = "frame_%08d.png",
) -> Path:
    """Reconstruct video from saved image sequence on disk, preserving original audio and fps."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg executable not found in system PATH.")

    frames_path = Path(frames_dir).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = encoding_config or EncodingConfig()

    input_pattern = str(frames_path / frame_pattern)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-i", input_pattern,
    ]

    has_audio = False
    if source_audio_path:
        src = Path(source_audio_path).resolve()
        if src.exists():
            try:
                meta = probe_video(src)
                if meta.has_audio:
                    cmd.extend(["-i", str(src)])
                    has_audio = True
            except Exception:
                pass

    if has_audio:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
        cmd.extend(["-c:a", cfg.audio_codec])
        cmd.extend(["-shortest"])
    else:
        cmd.extend(["-map", "0:v:0"])

    cmd.extend([
        "-c:v", cfg.video_codec,
        "-preset", cfg.preset,
        "-crf", str(cfg.crf),
        "-pix_fmt", cfg.pix_fmt,
        "-movflags", "+faststart",
        str(out_path),
    ])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # If audio copy failed (e.g. unsupported container stream combination), retry with aac
        if has_audio and "-c:a copy" in " ".join(cmd):
            logger.warning("Audio copy failed, retrying with AAC encoding...")
            idx = cmd.index(cfg.audio_codec)
            cmd[idx] = cfg.audio_fallback_codec
            res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg compilation failed ({res.returncode}): {res.stderr}")

    return out_path
