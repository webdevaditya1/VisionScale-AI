"""Tiled Real-ESRGAN inference using the project's registered pretrained weights."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from visionscale.models.base import BaseEnhancerModel, ModelMetadata


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.activation(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.activation(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + x5 * 0.2


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.rdb3(self.rdb2(self.rdb1(x))) * 0.2


class RRDBNet(nn.Module):
    """RRDB architecture used by the official Real-ESRGAN x4+ checkpoints."""

    def __init__(self, num_block: int = 23, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *(ResidualInResidualDenseBlock(num_feat, num_grow_ch) for _ in range(num_block))
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.activation(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.activation(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.activation(self.conv_hr(feat)))


class RealESRGANEnhancer(BaseEnhancerModel):
    """Loads Real-ESRGAN weights and enhances BGR video frames in bounded tiles."""

    UPSCALE_FACTOR = 4

    def __init__(
        self,
        metadata: ModelMetadata,
        device: str = "cuda:0",
        use_fp16: bool = True,
        weights_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(metadata, device, use_fp16, weights_dir)
        self._model: Optional[RRDBNet] = None

    def load(self) -> None:
        if self.is_loaded:
            return

        model_id = self.metadata.model_id
        if model_id == "realesrgan-x4plus":
            num_block = 23
        elif model_id == "realesrgan-x4plus-anime":
            num_block = 6
        else:
            raise ValueError(f"Unsupported Real-ESRGAN model: {model_id}")

        self.ensure_weights()
        model = RRDBNet(num_block=num_block)
        try:
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.weights_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid model checkpoint format: {self.weights_path}")

        model.load_state_dict(state_dict, strict=True)
        model.eval().to(torch.device(self.device))
        if self.use_fp16 and self.device.startswith("cuda"):
            model.half()
        self._model = model
        self.is_loaded = True

    def _infer_tiled(self, tensor: torch.Tensor, tile_size: int, tile_pad: int) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model is not loaded.")

        _, _, height, width = tensor.shape
        scale = self.UPSCALE_FACTOR
        if tile_size <= 0 or (height <= tile_size and width <= tile_size):
            return self._model(tensor)

        output = tensor.new_empty((1, 3, height * scale, width * scale))
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                input_top = max(top - tile_pad, 0)
                input_left = max(left - tile_pad, 0)
                input_bottom = min(bottom + tile_pad, height)
                input_right = min(right + tile_pad, width)

                tile_output = self._model(
                    tensor[:, :, input_top:input_bottom, input_left:input_right]
                )
                crop_top = (top - input_top) * scale
                crop_left = (left - input_left) * scale
                crop_bottom = crop_top + (bottom - top) * scale
                crop_right = crop_left + (right - left) * scale
                output[:, :, top * scale:bottom * scale, left * scale:right * scale] = (
                    tile_output[:, :, crop_top:crop_bottom, crop_left:crop_right]
                )
        return output

    def enhance_frame(
        self,
        frame: np.ndarray,
        tile_size: int = 0,
        tile_pad: int = 10,
    ) -> np.ndarray:
        if not self.is_loaded or self._model is None:
            raise RuntimeError("Load the model before enhancing frames.")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR frame with three color channels.")

        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(device=torch.device(self.device), dtype=torch.float32)
        tensor.div_(255.0)
        if self.use_fp16 and self.device.startswith("cuda"):
            tensor = tensor.half()

        with torch.inference_mode():
            output = self._infer_tiled(tensor, tile_size, tile_pad)
            output = output.squeeze(0).float().clamp(0, 1).mul(255).round().byte()
        rgb_output = output.permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(rgb_output, cv2.COLOR_RGB2BGR)

    def unload(self) -> None:
        self._model = None
        self.is_loaded = False
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
