"""Unit tests for media downloader and upload validator."""

from pathlib import Path
import pytest
from visionscale.io.downloader import MediaDownloader


def test_youtube_url_detection(tmp_path: Path):
    downloader = MediaDownloader(tmp_path)

    # Valid URLs
    assert downloader.is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert downloader.is_youtube_url("http://youtu.be/dQw4w9WgXcQ")
    assert downloader.is_youtube_url("https://youtube.com/shorts/dQw4w9WgXcQ")

    # Invalid URLs
    assert not downloader.is_youtube_url("https://vimeo.com/123456")
    assert not downloader.is_youtube_url("https://example.com/video.mp4")
    assert not downloader.is_youtube_url("")


def test_validate_local_upload(tmp_path: Path):
    downloader = MediaDownloader(tmp_path)

    # Valid file
    valid_file = tmp_path / "valid.mp4"
    valid_file.write_bytes(b"dummy video content")
    assert downloader.validate_local_upload(valid_file) == valid_file

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        downloader.validate_local_upload(tmp_path / "missing.mp4")

    # Invalid extension
    invalid_ext = tmp_path / "test.txt"
    invalid_ext.write_text("not a video")
    with pytest.raises(ValueError, match="Unsupported file format"):
        downloader.validate_local_upload(invalid_ext)

    # Empty file
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="is empty"):
        downloader.validate_local_upload(empty_file)
