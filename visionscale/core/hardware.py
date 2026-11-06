"""Automatic hardware and GPU detection for VisionScale AI."""

from __future__ import annotations
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPUInfo:
    """Detailed specifications of an available GPU device."""
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    compute_capability: tuple[int, int]
    supports_fp16: bool
    supports_bf16: bool

    @property
    def total_vram_gb(self) -> float:
        return round(self.total_vram_mb / 1024.0, 2)

    @property
    def free_vram_gb(self) -> float:
        return round(self.free_vram_mb / 1024.0, 2)

    @property
    def is_t4(self) -> bool:
        """True if the GPU is an NVIDIA T4 (standard Colab/Kaggle free GPU)."""
        return "T4" in self.name.upper()


@dataclass(frozen=True)
class HardwareInfo:
    """System-wide hardware profile."""
    has_cuda: bool
    device_count: int
    devices: list[GPUInfo]
    cpu_cores: int
    system_ram_gb: float
    recommended_device: str  # "cuda:0", "cuda", "cpu"
    recommended_fp16: bool

    def summary(self) -> str:
        """Human-readable diagnostic report."""
        lines = [
            "=" * 60,
            " VisionScale AI - Hardware Diagnostics Report",
            "=" * 60,
            f"CUDA Available      : {self.has_cuda}",
            f"Device Count        : {self.device_count}",
            f"CPU Physical Cores  : {self.cpu_cores}",
            f"System RAM (est.)   : {self.system_ram_gb:.1f} GB",
            f"Recommended Device  : {self.recommended_device}",
            f"Use FP16 Precision  : {self.recommended_fp16}",
        ]
        if self.devices:
            for d in self.devices:
                lines.append(
                    f" - GPU [{d.index}]: {d.name} | VRAM: {d.total_vram_gb:.1f} GB "
                    f"(Free: {d.free_vram_gb:.1f} GB) | CC: {d.compute_capability[0]}.{d.compute_capability[1]} "
                    f"| FP16: {d.supports_fp16}"
                )
        else:
            lines.append(" - No NVIDIA CUDA devices found. Falling back to CPU processing.")
        lines.append("=" * 60)
        return "\n".join(lines)


def _get_nvidia_smi_gpus() -> list[dict]:
    """Fallback query via nvidia-smi if torch is not yet installed."""
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_vram_mb": int(parts[2]),
                    "free_vram_mb": int(parts[3]),
                })
        return gpus
    except Exception:
        return []


def _detect_torch_gpus() -> list[GPUInfo]:
    """Inspect GPUs using PyTorch if installed."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        devices: list[GPUInfo] = []
        count = torch.cuda.device_count()
        for idx in range(count):
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total_mb = int(props.total_memory / (1024 * 1024))

            # Query free memory
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                free_mb = int(free_b / (1024 * 1024))
            except Exception:
                free_mb = total_mb

            cc = (props.major, props.minor)
            # FP16 tensor core acceleration is available on Compute Capability >= 7.0 (Volta/Turing+)
            supports_fp16 = props.major >= 7
            # BF16 is generally supported on Ampere+ (CC >= 8.0)
            supports_bf16 = props.major >= 8

            devices.append(
                GPUInfo(
                    index=idx,
                    name=name,
                    total_vram_mb=total_mb,
                    free_vram_mb=free_mb,
                    compute_capability=cc,
                    supports_fp16=supports_fp16,
                    supports_bf16=supports_bf16,
                )
            )
        return devices
    except ImportError:
        # PyTorch not installed in this environment, try nvidia-smi fallback
        raw_gpus = _get_nvidia_smi_gpus()
        devices = []
        for g in raw_gpus:
            # Estimate compute capability: T4 is 7.5
            is_t4 = "T4" in g["name"].upper()
            cc = (7, 5) if is_t4 else (7, 0)
            devices.append(
                GPUInfo(
                    index=g["index"],
                    name=g["name"],
                    total_vram_mb=g["total_vram_mb"],
                    free_vram_mb=g["free_vram_mb"],
                    compute_capability=cc,
                    supports_fp16=True,
                    supports_bf16=False,
                )
            )
        return devices


def _estimate_system_ram_gb() -> float:
    """Estimate total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback estimation via os sysconf or Windows wmic/powershell
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
        return 16.0  # Safe default assumption for Colab/Kaggle standard containers


def detect_hardware(preferred_gpu_index: Optional[int] = None) -> HardwareInfo:
    """Scan and return complete hardware and GPU specifications."""
    devices = _detect_torch_gpus()
    has_cuda = len(devices) > 0
    cpu_cores = os.cpu_count() or 4
    system_ram = _estimate_system_ram_gb()

    if has_cuda:
        selected_idx = 0
        if preferred_gpu_index is not None and 0 <= preferred_gpu_index < len(devices):
            selected_idx = preferred_gpu_index
        target_dev = devices[selected_idx]
        rec_device = f"cuda:{selected_idx}"
        rec_fp16 = target_dev.supports_fp16
    else:
        rec_device = "cpu"
        rec_fp16 = False

    return HardwareInfo(
        has_cuda=has_cuda,
        device_count=len(devices),
        devices=devices,
        cpu_cores=cpu_cores,
        system_ram_gb=round(system_ram, 2),
        recommended_device=rec_device,
        recommended_fp16=rec_fp16,
    )
