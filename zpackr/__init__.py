"""ZPackR — LZ4-compressed delta training with per-block attenuation.

Frozen BERT base + LZ4-compressed trainable delta.
Per-block LZ4 compressibility directly attenuates delta in forward pass —
no external LR scheduler needed.

Usage:
    from zpackr import compress_model, ZPackRConfig

    model = compress_model(model, ZPackRConfig())
    # train normally — post_step() handles block-level attenuation
"""

from .zpackr_layer import ZPackRLinear, BLOCK_SIZE, I_MAX, ATTENUATION_SKIP_THRESHOLD, DeltaAccumulator
from .config import ZPackRConfig
from .layer_patcher import compress_model
from .prompt_gate import should_skip_backward
from .checkpoint import save_zpackr_checkpoint, load_zpackr_checkpoint

__all__ = [
    "ZPackRLinear",
    "BLOCK_SIZE",
    "I_MAX",
    "ATTENUATION_SKIP_THRESHOLD",
    "DeltaAccumulator",
    "ZPackRConfig",
    "compress_model",
    "should_skip_backward",
    "save_zpackr_checkpoint",
    "load_zpackr_checkpoint",
]
