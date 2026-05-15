"""Convergence gate — LSH-based per-row training gate.

Checks whether ALL rows across ALL ZPackRLinear layers are fully
attenuated (attenuation >= ATTENUATION_SKIP_THRESHOLD).  If so, the model
has fully converged on this prompt and backward can be skipped.
"""

from .zpackr_layer import ATTENUATION_SKIP_THRESHOLD


def should_skip_backward(zpl_layers, threshold: float = None) -> bool:
    """Return True if backward should be skipped (all rows converged).

    Args:
        zpl_layers: List of (name, ZPackRLinear) tuples.
        threshold: Override ATTENUATION_SKIP_THRESHOLD.

    Returns:
        True if every row in every layer has attenuation >= threshold.
        False if any row still has room to learn.
    """
    if not zpl_layers:
        return True

    limit = threshold if threshold is not None else ATTENUATION_SKIP_THRESHOLD

    for _, module in zpl_layers:
        # _atten_byte is a uint8 GPU tensor [in_features], 0-255
        min_attn = module._atten_byte.float().min().item()
        attn_norm = min_attn / 255.0
        if attn_norm < limit:
            return False

    return True
