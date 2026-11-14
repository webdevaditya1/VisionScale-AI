from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from visionscale.agents.base_agent import AgentContext
from visionscale.agents.model_selector import ModelSelectorAgent
from visionscale.config import ContentCategory, ResolutionTarget, VisionScaleConfig
from visionscale.core.hardware import detect_hardware
from visionscale.models.realesrgan import RRDBNet, RealESRGANEnhancer
from visionscale.models.registry import MODEL_REGISTRY
from visionscale.pipeline import enhance_video


def test_rrdbnet_outputs_four_times_input_dimensions():
    model = RRDBNet(num_block=1, num_feat=4, num_grow_ch=2).eval()
    result = model(torch.rand(1, 3, 8, 10))
    assert result.shape == (1, 3, 32, 40)


def test_tiled_realesrgan_frame_output_shape_and_color():
    class NearestUpscaler(nn.Module):
        def forward(self, tensor):
            return torch.nn.functional.interpolate(tensor, scale_factor=4, mode="nearest")

    enhancer = RealESRGANEnhancer(
        metadata=MODEL_REGISTRY["realesrgan-x4plus-anime"],
        device="cpu",
        use_fp16=False,
    )
    enhancer._model = NearestUpscaler()
    enhancer.is_loaded = True
    frame = np.full((7, 9, 3), (12, 80, 190), dtype=np.uint8)

    result = enhancer.enhance_frame(frame, tile_size=3, tile_pad=1)

    assert result.shape == (28, 36, 3)
    np.testing.assert_array_equal(result, np.broadcast_to(frame[0, 0], result.shape))


def test_anime_selection_uses_an_implemented_backend():
    config = VisionScaleConfig()
    context = AgentContext()
    context.set("video_metadata", SimpleNamespace(width=3840, height=2160))
    context.set("content_category", ContentCategory.ANIME)
    context.set("target_dimensions", (1920, 1080))

    result = ModelSelectorAgent(config, detect_hardware()).run(context)

    assert result.get("primary_upscaler_id") == "realesrgan-x4plus-anime"


def test_enhance_video_streams_frames_and_exports(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (32, 24),
    )
    for index in range(3):
        writer.write(np.full((24, 32, 3), index * 50, dtype=np.uint8))
    writer.release()

    unloaded = []

    def fake_load(self):
        self.is_loaded = True

    def fake_enhance(self, frame, tile_size=0, tile_pad=10):
        return cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    def fake_unload(self):
        self.is_loaded = False
        unloaded.append(True)

    monkeypatch.setattr(RealESRGANEnhancer, "ensure_weights", lambda self, progress_callback=None: self.weights_path)
    monkeypatch.setattr(RealESRGANEnhancer, "load", fake_load)
    monkeypatch.setattr(RealESRGANEnhancer, "enhance_frame", fake_enhance)
    monkeypatch.setattr(RealESRGANEnhancer, "unload", fake_unload)

    config = VisionScaleConfig(workspace_dir=tmp_path / "workspace")
    config.processing.target_resolution = ResolutionTarget.ORIGINAL
    config.processing.content_category = ContentCategory.REAL_WORLD
    output = enhance_video(
        source,
        config=config,
        output_dir=tmp_path / "output",
        hardware=detect_hardware(),
    )

    assert output.is_file()
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
    finally:
        capture.release()
    assert unloaded


def test_enhance_video_rejects_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Input video not found"):
        enhance_video(tmp_path / "missing.mp4")
