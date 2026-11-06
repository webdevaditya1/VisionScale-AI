"""Quality Checker Agent: frame integrity, dimension compliance, and visual artifact validation."""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np

from visionscale.agents.base_agent import AgentContext, BaseAgent

logger = logging.getLogger("visionscale.agents.quality")


class QualityCheckerAgent(BaseAgent):
    """Verifies generated frames against corruption, black frames, and dimension mismatch."""

    def __init__(self) -> None:
        super().__init__("QualityCheckerAgent")

    def inspect_frame(
        self,
        frame: np.ndarray,
        expected_width: int,
        expected_height: int,
        frame_idx: int = 0,
    ) -> bool:
        """Validate a single reconstructed frame."""
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error(f"Frame {frame_idx} is None or not a numpy array.")
            return False

        h, w = frame.shape[:2]
        if w != expected_width or h != expected_height:
            logger.error(
                f"Frame {frame_idx} dimension mismatch: expected ({expected_width}x{expected_height}), "
                f"got ({w}x{h})."
            )
            return False

        if frame.dtype != np.uint8:
            logger.error(f"Frame {frame_idx} invalid dtype: {frame.dtype}, expected uint8.")
            return False

        # Check for NaN / Inf
        if np.isnan(frame).any() or np.isinf(frame).any():
            logger.error(f"Frame {frame_idx} contains NaN or Inf values.")
            return False

        # Check for total black frame (mean < 0.5) or blown-out frame (mean > 254.5)
        mean_val = float(np.mean(frame))
        var_val = float(np.var(frame))
        if var_val < 0.001 and (mean_val < 0.5 or mean_val > 254.5):
            logger.warning(
                f"Frame {frame_idx} appears completely blank/uniform (mean={mean_val:.2f}, var={var_val:.2f})."
            )

        return True

    def run(self, context: AgentContext) -> AgentContext:
        self._current_context = context
        self.log(self.name, "Quality checker initialized for stream verification.")
        context.set("quality_checker_active", True)
        return context
