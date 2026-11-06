"""Core platform, hardware, and memory management modules for VisionScale AI."""

from visionscale.core.platform import PlatformEnvironment, detect_platform, get_workspace_paths
from visionscale.core.hardware import HardwareInfo, detect_hardware
from visionscale.core.memory import MemoryManager

__all__ = [
    "PlatformEnvironment",
    "detect_platform",
    "get_workspace_paths",
    "HardwareInfo",
    "detect_hardware",
    "MemoryManager",
]
