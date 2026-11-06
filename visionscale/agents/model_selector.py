"""Model Selector Agent: deterministic decision engine matching video profile to models."""

from __future__ import annotations
import logging
from visionscale.agents.base_agent import AgentContext, BaseAgent
from visionscale.config import ContentCategory, VisionScaleConfig
from visionscale.core.hardware import HardwareInfo
from visionscale.io.video_probe import VideoMetadata
from visionscale.models.registry import MODEL_REGISTRY, ModelMetadata

logger = logging.getLogger("visionscale.agents.selector")


class ModelSelectorAgent(BaseAgent):
    """Deterministic selection of upscaler, face restoration, and interpolation models."""

    def __init__(self, config: VisionScaleConfig, hardware: HardwareInfo) -> None:
        super().__init__("ModelSelectorAgent")
        self.config = config
        self.hardware = hardware

    def run(self, context: AgentContext) -> AgentContext:
        self._current_context = context
        meta: VideoMetadata = context.get("video_metadata")
        category: ContentCategory = context.get("content_category", ContentCategory.REAL_WORLD)
        target_w, target_h = context.get("target_dimensions", (1920, 1080))

        # Calculate required scale factor
        scale_x = target_w / float(meta.width)
        scale_y = target_h / float(meta.height)
        scale_factor = max(scale_x, scale_y)

        # 1. Primary Upscaler Selection
        if category == ContentCategory.ANIME:
            if scale_factor <= 2.2 and "real-cugan-2x" in MODEL_REGISTRY:
                chosen_upscaler_id = "real-cugan-2x"
            else:
                chosen_upscaler_id = "realesrgan-x4plus-anime"
        else:
            chosen_upscaler_id = "realesrgan-x4plus"

        upscaler_meta = MODEL_REGISTRY[chosen_upscaler_id]
        selected_models: list[ModelMetadata] = [upscaler_meta]

        self.log(
            self.name,
            f"Selected primary upscaler: '{upscaler_meta.name}' ({upscaler_meta.model_id}) "
            f"[Scale: {upscaler_meta.scale}x, License: {upscaler_meta.license}]",
        )

        # 2. Face Restoration Model (Optional)
        face_restorer_id = None
        if self.config.processing.enable_face_restoration:
            face_restorer_id = "gfpgan-v1.4"
            face_meta = MODEL_REGISTRY[face_restorer_id]
            selected_models.append(face_meta)
            self.log(
                self.name,
                f"Enabled face restoration model: '{face_meta.name}' [License: {face_meta.license}]",
            )

        # 3. Interpolation Model (Optional)
        interpolator_id = None
        if self.config.processing.enable_interpolation:
            interpolator_id = "rife-v4.6"
            interp_meta = MODEL_REGISTRY[interpolator_id]
            selected_models.append(interp_meta)
            self.log(
                self.name,
                f"Enabled temporal interpolation model: '{interp_meta.name}' [License: {interp_meta.license}]",
            )

        context.set("selected_models", selected_models)
        context.set("primary_upscaler_id", chosen_upscaler_id)
        context.set("face_restoration_model_id", face_restorer_id)
        context.set("interpolation_model_id", interpolator_id)
        context.set("effective_scale_factor", scale_factor)

        return context
