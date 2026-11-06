"""Unit tests for hardware diagnostics and memory profile computation."""

from visionscale.core.hardware import detect_hardware
from visionscale.core.memory import MemoryManager, ProcessingProfile


def test_detect_hardware():
    hw = detect_hardware()
    assert hw.cpu_cores >= 1
    assert hw.system_ram_gb > 0
    assert hw.recommended_device in ("cpu", "cuda", "cuda:0")
    report = hw.summary()
    assert "VisionScale AI - Hardware Diagnostics Report" in report
    assert "Physical Cores" in report


def test_memory_manager_cpu_profile():
    mm = MemoryManager(device="cpu")
    profile = mm.compute_profile(frame_width=1920, frame_height=1080)
    assert profile.use_fp16 is False
    assert profile.batch_size == 1
    assert profile.tile_size == 256
    assert profile.vram_profile == "cpu"


def test_memory_manager_t4_vram_profile():
    # Simulate NVIDIA T4 with 15GB VRAM
    mm = MemoryManager(device="cuda:0")
    # For 1080p source -> 4K upscaling on T4 (15000 MB available)
    profile_1080p = mm.compute_profile(
        frame_width=1920,
        frame_height=1080,
        available_vram_mb=15000,
        prefer_fp16=True,
    )
    assert profile_1080p.vram_profile == "medium"
    assert profile_1080p.tile_size == 400
    assert profile_1080p.batch_size == 1
    assert profile_1080p.use_fp16 is True

    # For 480p source on T4
    profile_480p = mm.compute_profile(
        frame_width=854,
        frame_height=480,
        available_vram_mb=15000,
        prefer_fp16=True,
    )
    assert profile_480p.tile_size == 0  # Untiled for SD
    assert profile_480p.batch_size == 2


def test_memory_manager_low_vram_profile():
    mm = MemoryManager(device="cuda:0")
    profile = mm.compute_profile(
        frame_width=1920,
        frame_height=1080,
        available_vram_mb=5000,
    )
    assert profile.vram_profile == "ultra_low"
    assert profile.tile_size <= 192
