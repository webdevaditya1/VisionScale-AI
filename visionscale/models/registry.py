"""Model Registry for VisionScale AI.

Catalogues all supported enhancement models, download URLs, licenses,
VRAM requirements, and capability metadata.
"""

from __future__ import annotations
from typing import Optional
from visionscale.models.base import ModelMetadata

# Comprehensive Registry of Open-Source Models
MODEL_REGISTRY: dict[str, ModelMetadata] = {
    # 1. Real-ESRGAN General (Standard Photo / Live Action Video)
    "realesrgan-x4plus": ModelMetadata(
        model_id="realesrgan-x4plus",
        name="Real-ESRGAN x4+",
        task="upscale",
        scale=4,
        license="BSD-3-Clause",
        license_url="https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE",
        source_url="https://github.com/xinntao/Real-ESRGAN",
        weights_filename="RealESRGAN_x4plus.pth",
        weights_download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        weights_size_mb=63.9,
        recommended_min_vram_mb=4000,
        supports_tiling=True,
        supports_fp16=True,
        description="General real-world video and photo upscaler with artifact removal.",
    ),

    # 2. Real-ESRGAN Anime 6B (Optimized for Animation & 2D Art)
    "realesrgan-x4plus-anime": ModelMetadata(
        model_id="realesrgan-x4plus-anime",
        name="Real-ESRGAN Anime 6B",
        task="anime_upscale",
        scale=4,
        license="BSD-3-Clause",
        license_url="https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE",
        source_url="https://github.com/xinntao/Real-ESRGAN",
        weights_filename="RealESRGAN_x4plus_anime_6B.pth",
        weights_download_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        weights_size_mb=17.1,
        recommended_min_vram_mb=3000,
        supports_tiling=True,
        supports_fp16=True,
        description="Fast 6-block RRDB anime upscaler specialized for line-art and textures.",
    ),

    # 3. Real-CUGAN (High Performance Anime Enhancer)
    "real-cugan-2x": ModelMetadata(
        model_id="real-cugan-2x",
        name="Real-CUGAN 2x",
        task="anime_upscale",
        scale=2,
        license="MIT",
        license_url="https://github.com/bilibili/ailab/blob/main/Real-CUGAN/LICENSE",
        source_url="https://github.com/bilibili/ailab/tree/main/Real-CUGAN",
        weights_filename="pro-conservative-up2x.pth",
        weights_download_url="https://github.com/bilibili/ailab/releases/download/real-cugan-models/pro-conservative-up2x.pth",
        weights_size_mb=14.5,
        recommended_min_vram_mb=3500,
        supports_tiling=True,
        supports_fp16=True,
        description="Bilibili AI-Lab Real-CUGAN anime super-resolution model.",
    ),

    # 4. GFPGAN v1.4 (Facial Restoration)
    "gfpgan-v1.4": ModelMetadata(
        model_id="gfpgan-v1.4",
        name="GFPGAN v1.4",
        task="face_restoration",
        scale=1,  # 1x restoration applied over faces
        license="Apache-2.0",
        license_url="https://github.com/TencentARC/GFPGAN/blob/master/LICENSE",
        source_url="https://github.com/TencentARC/GFPGAN",
        weights_filename="GFPGANv1.4.pth",
        weights_download_url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        weights_size_mb=332.0,
        recommended_min_vram_mb=4000,
        supports_tiling=False,  # Evaluates cropped face bounding boxes
        supports_fp16=True,
        description="Tencent ARC practical face restoration algorithm for degraded real-world faces.",
    ),

    # 5. BasicVSR++ (Temporal Recurrent Super-Resolution)
    "basicvsr-plus-plus": ModelMetadata(
        model_id="basicvsr-plus-plus",
        name="BasicVSR++",
        task="temporal_upscale",
        scale=4,
        license="Apache-2.0",
        license_url="https://github.com/open-mmlab/mmediting/blob/master/LICENSE",
        source_url="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/basicvsr_plusplus_reds4",
        weights_filename="basicvsr_plusplus_c64n7_8x1_600k_reds4.pth",
        weights_download_url="https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210220-80223e78.pth",
        weights_size_mb=29.3,
        recommended_min_vram_mb=12000,
        supports_tiling=False,
        supports_fp16=True,
        description="State-of-the-art recurrent temporal video super-resolution with optical flow alignment.",
    ),

    # 6. RIFE v4.6 (Real-Time Video Frame Interpolation)
    "rife-v4.6": ModelMetadata(
        model_id="rife-v4.6",
        name="RIFE v4.6",
        task="interpolation",
        scale=1,  # Temporal frame count doubler
        license="MIT",
        license_url="https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE",
        source_url="https://github.com/hzwer/Practical-RIFE",
        weights_filename="flownet-v4.6.pkl",
        weights_download_url="https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/flownet.pkl",
        weights_size_mb=68.2,
        recommended_min_vram_mb=4500,
        supports_tiling=True,
        supports_fp16=True,
        description="Real-Time Intermediate Flow Estimation for video frame rate doubling.",
    ),
}


class ModelRegistry:
    """Registry query interface."""

    @staticmethod
    def list_all() -> list[ModelMetadata]:
        return list(MODEL_REGISTRY.values())

    @staticmethod
    def get(model_id: str) -> Optional[ModelMetadata]:
        return MODEL_REGISTRY.get(model_id)

    @staticmethod
    def list_by_task(task: str) -> list[ModelMetadata]:
        return [m for m in MODEL_REGISTRY.values() if m.task == task]

    @staticmethod
    def format_license_table() -> str:
        """Markdown formatted table of open-source models and licenses."""
        rows = [
            "| Model | Task | Scale | License | Size | Source |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for m in MODEL_REGISTRY.values():
            rows.append(
                f"| {m.name} | `{m.task}` | {m.scale}x | [{m.license}]({m.license_url}) | "
                f"{m.weights_size_mb:.1f} MB | [Repo]({m.source_url}) |"
            )
        return "\n".join(rows)
