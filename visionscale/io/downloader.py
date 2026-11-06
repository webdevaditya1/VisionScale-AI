"""Media ingestion: YouTube video downloader via yt-dlp and local upload validation."""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("visionscale.io.downloader")

YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"
)


class MediaDownloader:
    """Manages YouTube media ingestion and local uploaded file validation."""

    def __init__(self, download_dir: str | Path) -> None:
        self.download_dir = Path(download_dir).resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Validate if the string matches a supported YouTube video URL format."""
        if not url:
            return False
        return bool(YOUTUBE_REGEX.match(url.strip()))

    def download_youtube_video(
        self,
        url: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_height: int = 1080,
    ) -> Path:
        """Download video from YouTube using yt-dlp with audio preservation.

        Selects best video stream up to max_height (default 1080p source) and best audio.
        """
        import yt_dlp

        if not self.is_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")

        output_template = str(self.download_dir / "%(id)s.%(ext)s")

        def _hook(d: dict) -> None:
            if progress_callback and d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
                downloaded = d.get("downloaded_bytes", 0)
                pct = (downloaded / total) * 100.0
                speed = d.get("_speed_str", "N/A")
                progress_callback(pct, f"Downloading: {pct:.1f}% ({speed})")

        ydl_opts = {
            # Pick best video <= max_height + best audio, fallback to best single file
            "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "progress_hooks": [_hook] if progress_callback else [],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id")
            ext = info.get("ext", "mp4")
            downloaded_file = self.download_dir / f"{video_id}.{ext}"

            # If merged as mp4
            mp4_file = self.download_dir / f"{video_id}.mp4"
            if mp4_file.exists():
                return mp4_file
            if downloaded_file.exists():
                return downloaded_file

            # Search if extension differs
            matches = list(self.download_dir.glob(f"{video_id}.*"))
            if matches:
                return matches[0]

            raise FileNotFoundError(f"Failed to find downloaded YouTube file for ID {video_id}")

    def validate_local_upload(self, file_path: str | Path) -> Path:
        """Verify that an uploaded or local path exists and is a valid video file."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Local file does not exist: {p}")

        valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts"}
        if p.suffix.lower() not in valid_exts:
            raise ValueError(
                f"Unsupported file format '{p.suffix}'. Supported formats: {', '.join(sorted(valid_exts))}"
            )

        if p.stat().st_size == 0:
            raise ValueError(f"Uploaded video file is empty: {p}")

        return p
