"""Cerebellum-style sensory cancellation layer.

Implements the core idea from Perks et al. (Nature, 2026, doi:10.1038/s41586-026-10690-6)
mapped onto a camera+IMU rig:

- Corollary discharge analogue: the IMU (self-motion) signal, expanded into a
  temporal basis, acts as the granule-cell layer that *could* predict the
  sensory consequences of the rig's own motion.
- Negative image: paired fast/slow plasticity learns to predict the residual
  that remains after gyro-based rotation compensation (parallax, unmodelled
  effects, sensor artifacts). Fast weights adapt within a session; slow
  weights anchor long-term statistics and are protected from fast drift.
- Surprise: the final residual (actual - predicted - slow negative image) is
  the event trigger. Only surprising content is worth recording.
"""

from .core import CerebellumCancel, TriggerConfig
from .pipeline import PipelineResult, run_pipeline

__all__ = ["CerebellumCancel", "TriggerConfig", "PipelineResult", "run_pipeline"]
