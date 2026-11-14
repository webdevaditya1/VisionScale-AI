#!/usr/bin/env python3
"""VisionScale AI - Kaggle T4 GPU ready-to-run script.

This script is meant to be run inside a Kaggle notebook or as a Python file in
Kaggle. It locates an input video, runs tiled Real-ESRGAN inference, and writes
the enhanced MP4 to the VisionScale output directory.

Usage:
    python kaggle_t4_run.py --input /kaggle/working/my_video.mp4 --target-res 4k
    python kaggle_t4_run.py --input /kaggle/input/my_video.mp4 --category anime
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visionscale.config import ContentCategory, ResolutionTarget, VisionScaleConfig
from visionscale.core.hardware import detect_hardware
from visionscale.core.platform import initialize_platform
from visionscale.pipeline import enhance_video


def print_header() -> None:
    print("=" * 72)
    print("VisionScale AI - Kaggle T4 GPU Run Script")
    print("=" * 72)


def find_video_input(user_input: str | None) -> str:
    """Find a video file in common Kaggle locations."""
    candidates: list[Path] = []

    if user_input:
        p = Path(user_input).expanduser().resolve()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Input video not found: {user_input}")

    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
    ]

    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No video file found. Upload a video to Kaggle, or pass --input /path/to/video.mp4"
        )

    # Pick the first video found, but prefer working dir files.
    preferred = sorted(
        candidates,
        key=lambda p: (0 if "working" in p.parts and "kaggle" in p.parts else 1, str(p)),
    )
    return str(preferred[0])


def configure_runtime(target_resolution: str, content_category: str, force_cpu: bool) -> VisionScaleConfig:
    config = VisionScaleConfig()
    config.processing.force_cpu = force_cpu
    config.processing.use_fp16 = True
    config.processing.target_resolution = ResolutionTarget.from_str(target_resolution)

    category_map = {
        "auto": ContentCategory.AUTO,
        "real": ContentCategory.REAL_WORLD,
        "anime": ContentCategory.ANIME,
        "archival": ContentCategory.ARCHIVAL,
    }
    try:
        config.processing.content_category = category_map[content_category.lower()]
    except KeyError as exc:
        raise ValueError(
            "Unsupported category. Choose one of: auto, real, anime, archival."
        ) from exc

    return config


def run_pipeline(video_path: str, target_resolution: str, content_category: str, force_cpu: bool) -> Path:
    print(f"Input video: {video_path}")

    config = configure_runtime(target_resolution, content_category, force_cpu)
    hardware = detect_hardware()
    platform = initialize_platform(custom_base_dir=str(PROJECT_ROOT / "visionscale_workspace"))

    print("\nHardware summary:")
    print(hardware.summary())
    print(f"\nOutput directory: {platform.paths.output_dir}")
    output_path = enhance_video(
        input_path=video_path,
        config=config,
        output_dir=platform.paths.output_dir,
        hardware=hardware,
        progress_callback=lambda done, total: print(
            f"\rProcessed {done}/{total} frames", end="\n" if done == total else ""
        ) if done == total or done % 25 == 0 else None,
    )
    return output_path


def main() -> None:
    print_header()

    parser = argparse.ArgumentParser(description="Enhance a video with VisionScale AI on Kaggle")
    parser.add_argument("--input", type=str, default=None, help="Path to input video file")
    parser.add_argument("--target-res", type=str, default="original", help="1080p | 4k | original")
    parser.add_argument("--category", type=str, default="auto", help="auto | anime | real | archival")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU mode for debugging")
    args = parser.parse_args()

    print("\nChecking GPU and runtime environment...")
    if not torch.cuda.is_available() and not args.force_cpu:
        raise RuntimeError(
            "CUDA is not available. Enable GPU in Kaggle notebook settings, restart the session, "
            "or pass --force-cpu (CPU inference is much slower)."
        )
    video_path = find_video_input(args.input)

    print("\nStarting VisionScale pipeline...")
    output_path = run_pipeline(
        video_path=video_path,
        target_resolution=args.target_res,
        content_category=args.category,
        force_cpu=args.force_cpu,
    )

    print(f"\nEnhancement complete: {output_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
