"""Video Analyzer Agent: deterministic inspection of video container and visual frames."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np

from visionscale.agents.base_agent import AgentContext, BaseAgent
from visionscale.config import ContentCategory, VisionScaleConfig
from visionscale.io.frame_extractor import FrameStream
from visionscale.io.video_probe import VideoMetadata, probe_video

logger = logging.getLogger("visionscale.agents.analyzer")


class VideoAnalyzerAgent(BaseAgent):
    """Analyzes container metadata and visual properties to classify content."""

    def __init__(self, config: VisionScaleConfig) -> None:
        super().__init__("VideoAnalyzerAgent")
        self.config = config

    def _analyze_frame_samples(self, video_path: Path, sample_count: int = 5) -> dict:
        """Sample a few evenly spaced frames to detect anime/CGI vs real-world content and sharpness."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"is_anime_candidate": False, "avg_sharpness": 100.0}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100

        step = max(1, total_frames // (sample_count + 1))
        sample_indices = [step * (i + 1) for i in range(sample_count)]

        sharpness_list: list[float] = []
        flat_area_ratios: list[float] = []

        try:
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # 1. Laplacian variance for sharpness / texture measure
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_list.append(lap_var)

                # 2. Color flat region ratio (Anime typically has large flat solid tone areas)
                # Compute gradient magnitude
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(sobelx**2 + sobely**2)
                # Flat pixels are those with very low gradient
                flat_ratio = float(np.mean(grad_mag < 8.0))
                flat_area_ratios.append(flat_ratio)
        finally:
            cap.release()

        avg_sharpness = float(np.mean(sharpness_list)) if sharpness_list else 100.0
        avg_flat_ratio = float(np.mean(flat_area_ratios)) if flat_area_ratios else 0.2

        # Anime typically has > 45% flat uniform color regions
        is_anime_candidate = avg_flat_ratio > 0.45

        return {
            "is_anime_candidate": is_anime_candidate,
            "avg_sharpness": avg_sharpness,
            "avg_flat_ratio": avg_flat_ratio,
        }

    def run(self, context: AgentContext) -> AgentContext:
        self._current_context = context
        video_path = context.get("video_input_path")
        if not video_path:
            raise ValueError("Context missing 'video_input_path'.")

        path = Path(video_path).resolve()
        self.log(self.name, f"Probing video stream: {path.name}")

        meta: VideoMetadata = probe_video(path)
        context.set("video_metadata", meta)

        self.log(self.name, f"Source metadata: {meta.summary()}")

        # Content classification
        user_cat = self.config.processing.content_category
        if user_cat == ContentCategory.AUTO:
            analysis = self._analyze_frame_samples(path)
            detected_anime = analysis["is_anime_candidate"]
            category = ContentCategory.ANIME if detected_anime else ContentCategory.REAL_WORLD
            self.log(
                self.name,
                f"Auto-classification: detected '{category.value}' "
                f"(flat_ratio: {analysis['avg_flat_ratio']:.2f}, sharpness: {analysis['avg_sharpness']:.1f})",
            )
        else:
            category = user_cat
            self.log(self.name, f"Using explicitly requested category: '{category.value}'")

        # Compute target output dimensions
        target_w, target_h = self.config.compute_target_dimensions(meta.width, meta.height)
        context.set("content_category", category)
        context.set("target_dimensions", (target_w, target_h))
        self.log(
            self.name,
            f"Target upscaling resolution: {target_w}x{target_h} "
            f"({self.config.processing.target_resolution.value})",
        )

        return context
