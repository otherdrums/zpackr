"""ZPackR — LSH-attenuated delta training with per-block convergence detection.

Frozen BERT base + trainable delta with per-block attenuation.
Multi-scale LSH comparison detects convergence and prevents overfitting.

Usage:
    from zpackr import compress_model, ZPackRConfig

    model = compress_model(model, ZPackRConfig())
    # train normally — post_step() handles block-level attenuation
"""

from .zpackr_layer import ZPackRLinear, BLOCK_SIZE, ATTENUATION_SKIP_THRESHOLD, DeltaSignatureDB
from .config import ZPackRConfig
from .layer_patcher import compress_model
from .prompt_gate import should_skip_backward
from .checkpoint import save_zpackr_checkpoint, load_zpackr_checkpoint

__all__ = [
    "ZPackRLinear",
    "BLOCK_SIZE",
    "ATTENUATION_SKIP_THRESHOLD",
    "DeltaSignatureDB",
    "ZPackRConfig",
    "compress_model",
    "should_skip_backward",
    "save_zpackr_checkpoint",
    "load_zpackr_checkpoint",
]
