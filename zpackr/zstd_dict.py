"""WeightDict — adaptive zstd dictionary for weight byte compression (ZPackR v2.0).

Stores a zstd compression dictionary trained on weight byte patterns.
Used to compress/decompress weight deltas and compute salience signals.

The dictionary is rebuilt at explicit reindex() call-sites using zstd's
built-in dictionary training, which finds optimal compression patterns
(no hand-rolled window scanning needed).

Ratio convention:  ratio = len(uncompressed) / len(compressed).  >= 1.0 always.
  ratio >= 2.0 → weight pattern is familiar (learned).
  ratio <  2.0 → weight pattern is novel (should keep salient).

Key properties:
  - Lossless roundtrip for bf16 weight bytes.
  - Dictionary trained via zstd.train_dictionary (production-quality).
  - Evolves at explicit reindex() call-sites — never mid-step.
  - Save/load for checkpoint-based training era rewind.
"""

import json
import os


class WeightDict:
    """Adaptive zstd dictionary built from weight byte patterns.

    Manages salience state and enables checkpoint save/load.
    """

    def __init__(self, max_entries: int = 16384):
        self.max_entries = max_entries
        self._zstd_dict = None
        self._cctx = None
        self._dctx = None
        self._dict_bytes = b""
        self._num_entries = 0
        self._base_samples = []  # cached BERT base weight chunks for accumulating reindex

    @property
    def num_entries(self) -> int:
        return self._num_entries

    @property
    def is_empty(self) -> bool:
        return self._num_entries == 0

    def compress(self, weight_bytes: bytes) -> bytes:
        self._ensure_cctx()
        return self._cctx.compress(weight_bytes)

    def decompress(self, compressed_bytes: bytes) -> bytes:
        self._ensure_dctx()
        return self._dctx.decompress(compressed_bytes)

    def ratio(self, weight_bytes: bytes) -> float:
        compressed = self.compress(weight_bytes)
        if len(compressed) == 0:
            return float("inf")
        return len(weight_bytes) / len(compressed)

    def batch_ratios(self, block_bytes_list: list) -> list:
        """Compute compression ratios for multiple blocks in one C-level call.

        Uses multi_compress_to_buffer which releases the GIL and processes
        all blocks in pure C.  ~3-5x faster than per-block Python loops.
        """
        self._ensure_cctx()
        if not block_bytes_list:
            return []
        results = self._cctx.multi_compress_to_buffer(block_bytes_list)
        ratios = []
        for i, blk in enumerate(block_bytes_list):
            clen = len(results[i].tobytes())
            if clen == 0:
                ratios.append(float("inf"))
            else:
                ratios.append(len(blk) / clen)
        return ratios

    def _ensure_cctx(self):
        if self._cctx is None:
            self._rebuild_zstd()

    def _ensure_dctx(self):
        if self._dctx is None:
            self._rebuild_zstd()

    def _rebuild_zstd(self):
        import zstandard as zstd

        if not self._dict_bytes:
            self._zstd_dict = None
            self._cctx = zstd.ZstdCompressor(level=1)
            self._dctx = zstd.ZstdDecompressor()
            return

        self._zstd_dict = zstd.ZstdCompressionDict(self._dict_bytes)
        self._cctx = zstd.ZstdCompressor(level=1, dict_data=self._zstd_dict)
        self._dctx = zstd.ZstdDecompressor(dict_data=self._zstd_dict)

    def reindex(self, weight_bytes: bytes, min_frequency: float = 0.10, stride: int = 8, min_count: int = 10,
                delta_bytes: bytes = None):
        """Train the zstd dictionary on current weight bytes.

        Accumulates base weight patterns (cached at setup) + delta patterns
        from current training state.  This keeps BERT base patterns as a
        signal floor while delta patterns evolve — critical for continuous
        learning and maintaining a wide signal gap.

        Args:
            weight_bytes: Weight bytes to train on (base at setup, delta in loop).
            delta_bytes: If provided, delta patterns are also included.
        """
        import zstandard as zstd

        if len(weight_bytes) < 256:
            return self._num_entries

        # Chunk into 8KB samples
        sample_size = 8192
        samples = []
        for i in range(0, len(weight_bytes), sample_size):
            chunk = weight_bytes[i:i + sample_size]
            if len(chunk) >= 256:
                samples.append(chunk)

        # Always include cached base samples (BERT foundation never leaves)
        if self._base_samples:
            samples.extend(self._base_samples)

        # Add delta samples if provided (current training state)
        if delta_bytes and len(delta_bytes) >= 256:
            for i in range(0, len(delta_bytes), sample_size):
                chunk = delta_bytes[i:i + sample_size]
                if len(chunk) >= 256:
                    samples.append(chunk)

        if len(samples) < 2:
            return self._num_entries

        total_bytes = sum(len(s) for s in samples)
        dict_size = max(4096, min(131072, total_bytes // 40))

        try:
            dict_obj = zstd.train_dictionary(dict_size, samples)
        except zstd.ZstdError:
            return self._num_entries

        self._dict_bytes = dict_obj.as_bytes()
        self._zstd_dict = dict_obj
        self._num_entries = len(self._dict_bytes) // 16

        self._cctx = zstd.ZstdCompressor(level=1, dict_data=self._zstd_dict)
        self._dctx = zstd.ZstdDecompressor(dict_data=self._zstd_dict)

        return self._num_entries

    def set_base_samples(self, base_weight_bytes: bytes):
        """Cache BERT base weight chunks — these persist across all reindex calls.

        Call once at setup with the combined base weights from all layers.
        """
        sample_size = 8192
        self._base_samples = []
        for i in range(0, len(base_weight_bytes), sample_size):
            chunk = base_weight_bytes[i:i + sample_size]
            if len(chunk) >= 256:
                self._base_samples.append(chunk)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        state = {
            "version": 2,
            "max_entries": self.max_entries,
            "num_entries": self._num_entries,
        }

        with open(path + ".json", "w") as f:
            json.dump(state, f)

        if self._dict_bytes:
            with open(path + ".zdict", "wb") as f:
                f.write(self._dict_bytes)
        else:
            # Empty dict — create an empty file marker
            with open(path + ".zdict", "wb") as f:
                pass

    @classmethod
    def load(cls, path: str):
        meta_path = path + ".json"
        zdict_path = path + ".zdict"

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"WeightDict metadata not found: {meta_path}")

        with open(meta_path, "r") as f:
            state = json.load(f)

        wd = cls(max_entries=state.get("max_entries", 16384))
        wd._num_entries = state.get("num_entries", 0)

        if os.path.exists(zdict_path):
            with open(zdict_path, "rb") as f:
                dict_bytes = f.read()
            if dict_bytes:
                wd._dict_bytes = dict_bytes
                wd._rebuild_zstd()

        return wd

    def __repr__(self):
        return f"WeightDict(entries={self._num_entries}/{self.max_entries})"
