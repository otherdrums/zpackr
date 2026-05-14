"""Convergence gate — LZ4-based per-block training gate.

Checks whether ALL blocks across ALL ZPackRLinear layers are fully
attenuated (attenuation >= ATTENUATION_SKIP_THRESHOLD).  If so, the model
has fully converged on this prompt and backward can be skipped.
"""

from .zpackr_layer import ATTENUATION_SKIP_THRESHOLD


def should_skip_backward(zpl_layers, threshold: float = None) -> bool:
    """Return True if backward should be skipped (all blocks converged).

    Args:
        zpl_layers: List of (name, ZPackRLinear) tuples.
        threshold: Override ATTENUATION_SKIP_THRESHOLD.

    Returns:
        True if every block in every layer has attenuation >= threshold.
        False if any block still has room to learn.
    """
    if not zpl_layers:
        return True

    limit = threshold if threshold is not None else ATTENUATION_SKIP_THRESHOLD

    for _, module in zpl_layers:
        factors = module._attenuation_factors
        if factors is None:
            return False  # No factors yet → still training
        if any(a < limit for a in factors):
            return False  # At least one block still novel

    return True
