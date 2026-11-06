"""Abstract base class and contract for all VisionScale enhancement models."""

from __future__ import annotations
import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing model capabilities, licensing, and memory constraints."""
    model_id: str
    name: str
    task: str  # "upscale", "anime_upscale", "face_restoration", "interpolation", "temporal_upscale"
    scale: int
    license: str
    license_url: str
    source_url: str
    weights_filename: str
    weights_download_url: str
    weights_size_mb: float
    recommended_min_vram_mb: int
    supports_tiling: bool
    supports_fp16: bool
    description: str


class BaseEnhancerModel(abc.ABC):
    """Abstract interface that all neural network enhancers must implement."""

    def __init__(
        self,
        metadata: ModelMetadata,
        device: str = "cuda:0",
        use_fp16: bool = True,
        weights_dir: Optional[Path] = None,
    ) -> None:
        self.metadata = metadata
        self.device = device
        self.use_fp16 = use_fp16
        self.weights_dir = Path(weights_dir) if weights_dir else Path("./weights")
        self.weights_path = self.weights_dir / metadata.weights_filename
        self.is_loaded = False

    def is_weights_downloaded(self) -> bool:
        """Check whether local weights file exists and is non-empty."""
        return self.weights_path.is_file() and self.weights_path.stat().st_size > 1024

    def ensure_weights(self, progress_callback=None) -> Path:
        """Download weights file if not already present."""
        if self.is_weights_downloaded():
            return self.weights_path

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request

        url = self.metadata.weights_download_url
        target = self.weights_path
        tmp_target = target.with_suffix(".download.tmp")

        try:
            # Download with basic progress hook
            def _reporthook(count, block_size, total_size):
                if progress_callback and total_size > 0:
                    pct = (count * block_size / total_size) * 100
                    progress_callback(min(pct, 100.0), f"Downloading {self.metadata.name}: {pct:.1f}%")

            urllib.request.urlretrieve(url, tmp_target, reporthook=_reporthook if progress_callback else None)
            tmp_target.replace(target)
            return target
        except Exception as e:
            if tmp_target.exists():
                tmp_target.unlink()
            raise RuntimeError(f"Failed to download weights for {self.metadata.name} from {url}: {e}")

    @abc.abstractmethod
    def load(self) -> None:
        """Load model architecture, weights onto device, and set eval mode."""
        pass

    @abc.abstractmethod
    def unload(self) -> None:
        """Unload model from GPU memory and invoke garbage collection."""
        pass

    @abc.abstractmethod
    def enhance_frame(
        self,
        frame: np.ndarray,
        tile_size: int = 0,
        tile_pad: int = 10,
    ) -> np.ndarray:
        """Enhance a single BGR image array (H x W x C).

        Returns enhanced BGR image array.
        """
        pass
