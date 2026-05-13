"""ZPackR configuration — extends PackRConfig with dual-dict fields."""

from dataclasses import dataclass
from typing import Literal
from packr.config import PackRConfig, SchemeType


@dataclass
class ZPackRConfig:
    """Configuration for ZPackR dual-dict training.

    Extends PackRConfig with WeightDict and SuperDict parameters.
    All PackRConfig fields are passed through to the base compressor.

    Args:
        layer_scope:            Which linear layers to replace
        gradient_checkpointing: Enable gradient checkpointing
        use_8bit_optimizer:     Use FusedQuantizedAdam (Triton 8-bit Adam)
        offload:                Enable CPU/system RAM offloading
        block_size:             Quantization block / salience block size
        zstd_max_entries:       Max entries in adaptive WeightDict
        zstd_salience_threshold: Initial ratio threshold (auto-calibrated)
        zstd_calibration_multiplier: Calibration level for auto-threshold
        zstd_regrow_noise:      Gaussian noise scale for regrown blocks
    """

    layer_scope: Literal["ffn", "attention", "all"] = "ffn"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    offload: bool = False
    block_size: int = 256

    # ZPackR-specific
    zstd_max_entries: int = 16384
    zstd_salience_threshold: float = 1.4
    zstd_calibration_multiplier: float = 0.01
    zstd_regrow_noise: float = 1e-4

    def to_packr_config(self) -> PackRConfig:
        """Convert to base PackRConfig for compressor initialization."""
        return PackRConfig(
            scheme="phr",
            layer_scope=self.layer_scope,
            gradient_checkpointing=self.gradient_checkpointing,
            use_8bit_optimizer=self.use_8bit_optimizer,
            offload=self.offload,
            block_size=self.block_size,
        )
