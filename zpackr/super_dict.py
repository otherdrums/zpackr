"""Super Dict loader — frozen text codec for ZPackR v2.0.

The Super Dict is a zstd compression dictionary built from English text
(GLUE corpus + dictionary words).  It is FROZEN forever — the model learns
IN this encoding but the encoding itself never changes.

Provides one method:  compress(text_bytes) -> ratio (uncomp/comp).
Ratio >= 2.0 means the text is familiar/compressible; < 2.0 means novel.
"""

import os


def load_super_dict(path: str = None):
    """Load the frozen Super Dict from a .zdict file.

    Lazily imports zstandard — only when mode='zpackr' is used.

    Args:
        path: Path to super_dict.zdict.  Defaults to packr/super_dict.zdict.

    Returns:
        A ZPackRSuperDict instance with a .compress(text_bytes) -> ratio method.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "super_dict.zdict")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Super Dict not found at {path}.  "
            f"Run tools/build_super_dict.py to build it."
        )

    import zstandard as zstd

    with open(path, "rb") as f:
        dict_data = f.read()

    cctx = zstd.ZstdCompressor(level=3, dict_data=zstd.ZstdCompressionDict(dict_data))
    dctx = zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(dict_data))

    return ZPackRSuperDict(dict_data, cctx, dctx, path)


class ZPackRSuperDict:
    """Frozen zstd dictionary for text compression.

    Ratio convention: ratio = uncompressed / compressed.
    >= 2.0 → familiar text (known).
    <  2.0 → novel text (should train).
    """

    def __init__(self, dict_data: bytes, cctx, dctx, path: str):
        self._dict_data = dict_data
        self._cctx = cctx
        self._dctx = dctx
        self._path = path

    def compress(self, text_bytes: bytes) -> float:
        """Compress text and return the compression ratio.

        Ratio = len(uncompressed) / len(compressed).  Always >= 1.0.
        """
        compressed = self._cctx.compress(text_bytes)
        if len(compressed) == 0:
            return float("inf")
        return len(text_bytes) / len(compressed)

    def decompress(self, compressed_bytes: bytes) -> bytes:
        """Decompress bytes back to original text."""
        return self._dctx.decompress(compressed_bytes)

    def should_train(self, text_bytes: bytes, threshold: float = 2.0) -> bool:
        """Return True if this text should trigger training (ratio < threshold)."""
        return self.compress(text_bytes) < threshold

    def decompress_to_text(self, compressed_bytes: bytes) -> str:
        """Decompress zstd bytes back to text string."""
        text_bytes = self._dctx.decompress(compressed_bytes)
        return text_bytes.decode("utf-8", errors="replace")

    def prompt_roundtrip(self, text: str) -> tuple[bytes, float]:
        """Compress text → zstd bytes, return (compressed_bytes, ratio)."""
        text_bytes = text.encode("utf-8")
        compressed = self._cctx.compress(text_bytes)
        ratio = len(text_bytes) / max(len(compressed), 1)
        return compressed, ratio

    @property
    def path(self):
        return self._path

    def __repr__(self):
        return f"ZPackRSuperDict(path={self._path!r}, size={len(self._dict_data)} bytes)"
