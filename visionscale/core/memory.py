"""Memory management, tiled inference parameter calculation, and OOM recovery."""

from __future__ import annotations
import gc
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("visionscale.memory")

T = TypeVar("T")


@dataclass
class ProcessingProfile:
    """Tiling and batching parameters calibrated to current VRAM and video dimensions."""
    batch_size: int
    tile_size: int  # 0 means no tiling (process full frame if VRAM permits)
    tile_pad: int = 10
    use_fp16: bool = True
    vram_profile: str = "medium"  # "low" (<8GB), "medium" (8-16GB / T4), "high" (>16GB)


@dataclass
class CheckpointState:
    """State record for resuming an interrupted video enhancement pipeline."""
    video_id: str
    input_path: str
    output_path: str
    target_resolution: str
    total_frames: int
    last_processed_frame_idx: int
    completed: bool
    model_name: str
    created_at: float
    updated_at: float


class MemoryManager:
    """Monitors VRAM, calculates safe tile and batch sizes, and orchestrates OOM recovery."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._is_cuda = device.startswith("cuda")

    def get_current_vram_mb(self) -> dict[str, int]:
        """Return allocated, reserved, and free VRAM in MB."""
        if not self._is_cuda:
            return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}

        try:
            import torch
            if not torch.cuda.is_available():
                return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}

            dev_idx = 0
            if ":" in self.device:
                dev_idx = int(self.device.split(":")[1])

            free_b, total_b = torch.cuda.mem_get_info(dev_idx)
            allocated_b = torch.cuda.memory_allocated(dev_idx)
            reserved_b = torch.cuda.memory_reserved(dev_idx)

            return {
                "allocated_mb": int(allocated_b / (1024 * 1024)),
                "reserved_mb": int(reserved_b / (1024 * 1024)),
                "free_mb": int(free_b / (1024 * 1024)),
                "total_mb": int(total_b / (1024 * 1024)),
            }
        except Exception as e:
            logger.debug(f"Could not read CUDA memory: {e}")
            return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}

    def clear_cache(self) -> None:
        """Run garbage collection and flush CUDA memory caches."""
        gc.collect()
        if self._is_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    def compute_profile(
        self,
        frame_width: int,
        frame_height: int,
        scale: int = 4,
        available_vram_mb: Optional[int] = None,
        prefer_fp16: bool = True,
    ) -> ProcessingProfile:
        """Derive safe batch size and tile dimension to strictly prevent out-of-memory errors.

        Colab/Kaggle NVIDIA T4 has ~15.1 GB usable VRAM.
        Upscaling 1080p -> 4K requires ~4x-8x peak intermediate feature maps.
        """
        if available_vram_mb is None:
            vram_info = self.get_current_vram_mb()
            # If total_mb is reported, use it; otherwise assume 15000 MB for T4 target
            vram_mb = vram_info["free_mb"] if vram_info["free_mb"] > 0 else 15000
        else:
            vram_mb = available_vram_mb

        if not self._is_cuda:
            # CPU profile: Small batches, tiling to avoid RAM blowout
            return ProcessingProfile(
                batch_size=1,
                tile_size=256,
                tile_pad=10,
                use_fp16=False,
                vram_profile="cpu",
            )

        # Classify VRAM tier
        if vram_mb >= 20000:
            # High tier (A100, RTX 3090/4090, V100 32GB)
            # Full 1080p frames can often be processed untiled or lightly tiled
            tile_size = 0 if (frame_width * frame_height <= 1920 * 1080) else 512
            batch_size = 2 if tile_size == 0 else 1
            v_tier = "high"
        elif vram_mb >= 12000:
            # Target tier: NVIDIA T4 (16GB), RTX 3080/4070 (12-16GB)
            # 720p or lower can run untiled. 1080p upscaling to 4K is safest with 400-512px tiles.
            pixels = frame_width * frame_height
            if pixels <= 854 * 480:
                tile_size = 0  # Untiled for SD
                batch_size = 2
            elif pixels <= 1280 * 720:
                tile_size = 512
                batch_size = 1
            else:
                # 1080p or higher
                tile_size = 400
                batch_size = 1
            v_tier = "medium"
        elif vram_mb >= 6000:
            # Low tier (6-8GB VRAM)
            tile_size = 256
            batch_size = 1
            v_tier = "low"
        else:
            # Ultra low (<6GB VRAM)
            tile_size = 192
            batch_size = 1
            v_tier = "ultra_low"

        return ProcessingProfile(
            batch_size=batch_size,
            tile_size=tile_size,
            tile_pad=10,
            use_fp16=prefer_fp16,
            vram_profile=v_tier,
        )

    def execute_with_oom_recovery(
        self,
        fn: Callable[..., T],
        *args: Any,
        fallback_profile: Optional[ProcessingProfile] = None,
        **kwargs: Any,
    ) -> tuple[T, ProcessingProfile]:
        """Execute a processing callable. If CUDA OOM is encountered, free cache,

        scale down tile size, and retry.
        """
        try:
            return fn(*args, **kwargs), fallback_profile or ProcessingProfile(1, 400)
        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str and "memory" in err_str:
                logger.warning("CUDA Out of Memory caught! Initiating OOM recovery protocol...")
                self.clear_cache()

                # Adjust profile
                curr_tile = fallback_profile.tile_size if fallback_profile else 400
                new_tile = max(128, int(curr_tile * 0.7)) if curr_tile > 0 else 384
                new_profile = ProcessingProfile(
                    batch_size=1,
                    tile_size=new_tile,
                    tile_pad=10,
                    use_fp16=fallback_profile.use_fp16 if fallback_profile else True,
                    vram_profile="fallback_oom",
                )
                logger.warning(f"Retrying with degraded tile size: {new_tile}")

                # Update tile_size in kwargs if present
                if "tile_size" in kwargs:
                    kwargs["tile_size"] = new_tile
                if "profile" in kwargs:
                    kwargs["profile"] = new_profile

                result = fn(*args, **kwargs)
                return result, new_profile
            raise


class CheckpointManager:
    """Manages pipeline checkpoint state persistence for resuming long video jobs."""

    def __init__(self, checkpoints_dir: str | Path) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, video_id: str) -> Path:
        safe_id = "".join(c for c in video_id if c.isalnum() or c in ("-", "_"))
        return self.checkpoints_dir / f"checkpoint_{safe_id}.json"

    def save(self, state: CheckpointState) -> Path:
        """Write checkpoint state atomically to disk."""
        target_path = self._get_path(state.video_id)
        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)
        temp_path.replace(target_path)
        return target_path

    def load(self, video_id: str) -> Optional[CheckpointState]:
        """Load checkpoint state if it exists."""
        target_path = self._get_path(video_id)
        if not target_path.exists():
            return None
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CheckpointState(**data)
        except Exception as e:
            logger.warning(f"Failed to read checkpoint for {video_id}: {e}")
            return None

    def remove(self, video_id: str) -> None:
        """Clear checkpoint upon successful final video export."""
        target_path = self._get_path(video_id)
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
