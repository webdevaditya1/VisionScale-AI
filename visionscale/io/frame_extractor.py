"""Streaming and batch-based frame extraction without loading full video into memory."""

from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import Generator, Iterator, Optional
import numpy as np

logger = logging.getLogger("visionscale.io.extractor")


class FrameStream:
    """Iterates through video frames sequentially as numpy arrays (BGR format).

    Guarantees constant memory consumption regardless of video length.
    """

    def __init__(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
    ) -> None:
        self.video_path = Path(video_path).resolve()
        self.start_frame = start_frame
        self.max_frames = max_frames
        self._cap = None

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file for streaming: {self.video_path}")

        try:
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            curr_idx = self.start_frame
            yielded_count = 0

            while True:
                if self.max_frames is not None and yielded_count >= self.max_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                yield curr_idx, frame
                curr_idx += 1
                yielded_count += 1
        finally:
            cap.release()

    def batches(self, batch_size: int = 1) -> Generator[list[tuple[int, np.ndarray]], None, None]:
        """Yield frames grouped into batches."""
        batch: list[tuple[int, np.ndarray]] = []
        for idx, frame in self:
            batch.append((idx, frame))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class FrameExtractor:
    """Extracts frames from video to disk or temporary directory for checkpointing / out-of-core pipelines."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_all_to_disk(
        self,
        video_path: str | Path,
        image_format: str = "png",
        start_frame: int = 0,
        max_frames: Optional[int] = None,
    ) -> list[Path]:
        """Extract frames directly into output directory as sequential images."""
        import cv2

        extracted_paths: list[Path] = []
        stream = FrameStream(video_path, start_frame=start_frame, max_frames=max_frames)

        for idx, frame in stream:
            filename = f"frame_{idx:08d}.{image_format}"
            out_path = self.output_dir / filename
            cv2.imwrite(str(out_path), frame)
            extracted_paths.append(out_path)

        logger.info(f"Extracted {len(extracted_paths)} frames to {self.output_dir}")
        return extracted_paths

    def clear(self) -> None:
        """Clean up extracted frame files in output directory."""
        if self.output_dir.exists():
            for p in self.output_dir.glob("frame_*.*"):
                try:
                    p.unlink()
                except OSError:
                    pass
