"""ZPackR configuration."""

from dataclasses import dataclass
from typing import Literal
from packr.config import PackRConfig, SchemeType


@dataclass
class ZPackRConfig:
    """Configuration for ZPackR LZ4-compressed delta training.

    Args:
        layer_scope:            Which linear layers to replace
        gradient_checkpointing: Enable gradient checkpointing
        use_8bit_optimizer:     Use FusedQuantizedAdam (Triton 8-bit Adam)
        offload:                Enable CPU/system RAM offloading
        block_size:             Salience block size
    """

    layer_scope: Literal["ffn", "attention", "all"] = "ffn"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    offload: bool = False
    block_size: int = 256

    def to_packr_config(self) -> PackRConfig:
        return PackRConfig(
            scheme="phr",
            layer_scope=self.layer_scope,
            gradient_checkpointing=self.gradient_checkpointing,
            use_8bit_optimizer=self.use_8bit_optimizer,
            offload=self.offload,
            block_size=self.block_size,
        )
