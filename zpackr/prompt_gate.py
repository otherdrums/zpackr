"""Prompt gate — Super Dict binary training gate for ZPackR v2.0.

The Super Dict compresses the prompt text.  If the compression ratio is
high (>= threshold), the prompt is "known" and training can be skipped.
If the ratio is low (< threshold), the prompt is "novel" and the model
should train on it.

Together with Velvet's continuous LR modulation, this eliminates:
  - "How many epochs to train?"   → Super Dict gate says when to stop
  - "When to stop?"               → Super Dict ratio drops → keep training
  - LR schedule tuning             → Velvet handles continuous LR
"""


def should_train(
    prompt_bytes: bytes,
    super_zstd,
    threshold: float = 2.0,
) -> bool:
    """Compress prompt against the frozen Super Dict.

    Args:
        prompt_bytes: Raw prompt text as bytes (UTF-8 encoded).
        super_zstd:   A ZPackRSuperDict instance with .compress(text_bytes) -> ratio.
        threshold:    Ratio threshold.  >= threshold → known, < threshold → novel.

    Returns:
        True if the model should train on this prompt (ratio < threshold).
        False if the prompt is already known (ratio >= threshold).

    Raises:
        ValueError: If prompt_bytes is empty.
    """
    if not prompt_bytes:
        raise ValueError("prompt_bytes must not be empty")

    ratio = super_zstd.compress(prompt_bytes)
    return ratio < threshold
