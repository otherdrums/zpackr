"""ZPackR — zstd-native compressed neural network training.

Dual-dictionary architecture: frozen BERT base + WeightDict-compressed delta.
Block-level zstd compression ratios serve as the convergence signal —
no external LR scheduler needed.

Usage:
    from zpackr import compress_model, ZPackRConfig

    model = compress_model(model, ZPackRConfig())
    # train normally — post_step() handles block-level attenuation
"""

from .zpackr_layer import ZPackRLinear
from .zstd_dict import WeightDict
from .config import ZPackRConfig
from .layer_patcher import compress_model
from .super_dict import load_super_dict, ZPackRSuperDict
from .prompt_gate import should_train
from .salience import compute_salience

__all__ = [
    "ZPackRLinear",
    "WeightDict",
    "ZPackRConfig",
    "compress_model",
    "load_super_dict",
    "ZPackRSuperDict",
    "should_train",
    "compute_salience",
]
