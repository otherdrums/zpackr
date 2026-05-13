"""Block salience — determines which weight blocks stay in VRAM (ZPackR v2.0).

Uses the WeightDict compression ratio as a signal: blocks whose byte patterns
compress well (ratio >= threshold) are marked as "learned" and pruned from
VRAM.  Blocks that don't compress well (ratio < threshold) are "novel" and
kept salient (resident in GPU memory).

Block size 256 matches FusedQuantizedAdam.block_size for aligned indexing.
"""

import math
import torch


def compute_salience(
    weight: torch.Tensor,
    weight_dict,
    block_size: int = 256,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Compute per-block salience from WeightDict compression ratios.

    Args:
        weight:      Weight matrix [in_features, out_features] in bf16.
        weight_dict: WeightDict instance with .ratio(bytes) -> float.
        block_size:  Number of elements per block (default 256).
        threshold:   Ratio threshold.  >= threshold → pruned, < → kept.

    Returns:
        Bool tensor [num_blocks] where True = keep in VRAM (salient).
    """
    in_f, out_f = weight.shape
    total_elements = in_f * out_f

    # Convert weight to contiguous bytes (bf16 → 2 bytes/element)
    weight_bytes = weight.contiguous().view(torch.uint8).view(-1)

    num_blocks = math.ceil(total_elements / block_size)
    salient = torch.zeros(num_blocks, dtype=torch.bool)

    for blk in range(num_blocks):
        start = blk * block_size
        end = min(start + block_size, total_elements)

        # Each element is 2 bytes (bf16)
        byte_start = start * 2
        byte_end = end * 2
        blk_bytes = weight_bytes[byte_start:byte_end].numpy().tobytes()

        ratio = weight_dict.ratio(blk_bytes)
        salient[blk] = ratio < threshold

    return salient


def block_mask_to_indices(block_mask: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """Convert a bool block_mask to element-level index ranges.

    Returns:
        LongTensor [num_kept, 2] with (start, end) element indices.
    """
    kept_blocks = block_mask.nonzero(as_tuple=True)[0]
    indices = torch.zeros(len(kept_blocks), 2, dtype=torch.long)
    for i, blk in enumerate(kept_blocks):
        start = int(blk) * block_size
        indices[i, 0] = start
        indices[i, 1] = start + block_size
    return indices


def num_salient_blocks(block_mask: torch.Tensor) -> int:
    """Count the number of salient (kept) blocks."""
    return int(block_mask.sum().item())
