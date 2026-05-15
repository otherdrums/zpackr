"""ZPackRLinear — frozen base + LZ4-attenuated trainable delta.

Drop-in replacement for nn.Linear that stores a frozen base weight plus
a trainable delta.  Per-block attenuation prevents overfitting by suppressing
delta contribution for blocks the model has already converged on.

Forward:  output = x @ (base_W + delta * (1 - attenuation))
             └─ frozen ─┘   └─ trainable ─┘

The attenuation signal comes from DeltaAccumulator: delta bytes from every
block at every post_step are accumulated. Current deltas are compressed
against this accumulated history using zstd prefix dictionary mode.
Blocks whose delta patterns match the accumulated history are "known"
→ high compressibility → high attenuation → suppressed.
Blocks with novel delta patterns are "novel" → low compressibility →
low attenuation → train fully.

AIT formula: attenuation = max(0, 1 - 1/(ratio × I_MAX))
  I_MAX = 1/1.27 ≈ 0.79 (bf16 entropy floor, measured)
"""

import math
import torch
import torch.nn as nn
import zstandard as zstd
import threading


BLOCK_SIZE = 256

# AIT-derived constant: bf16 entropy floor.
# I_MAX = 1 / ratio_for_random_bf16
# Measured: zstd compresses random bf16 to ~79% (ratio ~1.27).
# Used in: attenuation = max(0, 1 - 1/(ratio * I_MAX))
I_MAX = 1.0 / 1.27  # ≈ 0.7874

# Gate threshold: if ALL blocks across ALL layers have attenuation >= this,
# the prompt is fully converged and backward can be skipped.
ATTENUATION_SKIP_THRESHOLD = 0.9


class DeltaAccumulator:
    """Accumulates delta byte chunks and compresses against them using zstd prefix mode.

    All ZPackRLinear layers share a single DeltaAccumulator. Each post_step,
    layers append their delta bytes. Compression uses accumulated bytes as a
    zstd prefix dictionary — blocks that match accumulated patterns compress
    well (known), blocks that don't compress poorly (novel).
    """

    def __init__(self, max_bytes: int = 500_000_000):
        self._chunks: list = []
        self._total_bytes = 0
        self._max_bytes = max_bytes
        self._prefix_cache = None  # cached concatenated bytes
        self._prefix_dirty = True

    def append(self, block_bytes: bytes):
        """Append delta bytes to the accumulator."""
        if len(block_bytes) >= 256:
            self._chunks.append(block_bytes)
            self._total_bytes += len(block_bytes)
            self._prefix_dirty = True
            # Evict oldest if over cap
            while self._total_bytes > self._max_bytes and len(self._chunks) > 1:
                removed = self._chunks.pop(0)
                self._total_bytes -= len(removed)

    @property
    def prefix_bytes(self) -> bytes:
        """Cached concatenated prefix bytes for zstd prefix dict."""
        if self._prefix_dirty:
            self._prefix_cache = b"".join(self._chunks)
            self._prefix_dirty = False
        return self._prefix_cache

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def compress(self, block_bytes: bytes) -> float:
        """Compress a block against accumulated prefix. Returns ratio."""
        prefix = self.prefix_bytes
        if not prefix:
            # No history — use raw zstd (entropy floor)
            compressed = zstd.ZstdCompressor(level=1).compress(block_bytes)
            return len(block_bytes) / max(len(compressed), 1)
        dict_obj = zstd.ZstdCompressionDict(prefix,
                                             dict_type=zstd.DICT_TYPE_RAWCONTENT)
        cctx = zstd.ZstdCompressor(level=1, dict_data=dict_obj)
        compressed = cctx.compress(block_bytes)
        return len(block_bytes) / max(len(compressed), 1)


class ZPackRLinear(nn.Module):
    """Linear layer with frozen base + zstd-compressed trainable delta.

    CPU/pinned (authoritative):
        _full_delta:      torch.Tensor [in, out] bf16   full delta matrix
        _zstd_delta:      bytes                          zstd-compressed delta

    GPU/VRAM:
        base_W:           torch.Tensor [in, out] bf16    frozen pretrained weight
        delta_salient:    torch.Tensor [kept*block, out] bf16  only kept blocks
        block_mask:       torch.Tensor bool[num_blocks]  which delta blocks in VRAM
        bias:             torch.Tensor [out] bf16 (optional)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = BLOCK_SIZE
        self.num_blocks = math.ceil(in_features / self.block_size)

        # Frozen base — stored in bf16 for direct matmul compatibility
        self.base_W = nn.Parameter(
            torch.zeros(in_features, out_features, dtype=torch.bfloat16),
            requires_grad=False,
        )

        # Delta starts as all-zeros, all blocks salient
        self.block_mask = torch.ones(self.num_blocks, dtype=torch.bool)
        self.delta_salient = nn.Parameter(
            torch.zeros(in_features, out_features, dtype=torch.bfloat16),
            requires_grad=True,
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features, dtype=torch.bfloat16)
        ) if bias else None

        self._full_delta = None       # [in, out] on CPU — authoritative delta
        self._zstd_delta = None       # zstd-compressed delta bytes (checkpoint)
        self._salient_count = self.num_blocks  # cached, updated in post_step

        # Cached kept-block indices (updated when mask changes)
        self._kept_indices = torch.arange(self.num_blocks, dtype=torch.long)

        # Ratio cache for diagnostic tools — invalidated on post_step
        self._ratio_cache = None

        # Per-block attenuation factors [0,1], computed from AIT formula.
        # 0.0 = fully active (novel), 1.0 = fully suppressed (known).
        self._attenuation_factors = None
        self._block_gaps = None  # cached per-block ratios

        # Cached scatter indices for fused forward matmul
        self._scatter_indices = None  # pre-built for index_add_ in forward

        # Async compression — double-buffer attenuation
        self._attenuation_lock = threading.Lock()
        self._attenuation_pending = None  # computed by background thread
        self._attenuation_thread = None

        # Shared DeltaAccumulator (set by harness after creation)
        self._accumulator: DeltaAccumulator = None

    def _compress_async(self):
        """Compress delta blocks against accumulated prefix in background thread.

        Called after stage_delta_async() snapshots the CPU delta.
        Uses DeltaAccumulator prefix compression for temporal pattern matching.
        Sets self._attenuation_pending and self._block_gaps when done.
        """
        if self._full_delta is None:
            with self._attenuation_lock:
                self._attenuation_pending = None
                self._block_gaps = None
            return

        import numpy as np
        delta_np = self._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
        block_el_bytes = self.block_size * self.out_features * 2
        ratios = []

        acc = self._accumulator
        if acc is None:
            # Fallback: raw zstd (no accumulator attached yet)
            cctx = zstd.ZstdCompressor(level=1)
            for blk in range(self.num_blocks):
                byte_start = blk * block_el_bytes
                byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
                if byte_end <= byte_start:
                    ratios.append(1.0)
                    continue
                blk_arr = delta_np[byte_start:byte_end]
                if not np.any(blk_arr):
                    ratios.append(1.0)
                    continue
                compressed = cctx.compress(blk_arr.tobytes())
                ratios.append(len(blk_arr) / max(len(compressed), 1))
        else:
            # Prefix compression against accumulated history
            for blk in range(self.num_blocks):
                byte_start = blk * block_el_bytes
                byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
                if byte_end <= byte_start:
                    ratios.append(1.0)
                    continue
                blk_arr = delta_np[byte_start:byte_end]

                # Cold-start protection: near-zero delta → no attenuation
                l2 = float(np.sqrt(np.sum(blk_arr.astype(np.float32) ** 2)))
                if l2 < 1e-4:
                    ratios.append(1.0)
                    continue

                if not np.any(blk_arr):
                    ratios.append(1.0)
                    continue

                # Prefix compression: compress against accumulated history
                blk_bytes = blk_arr.tobytes()
                ratios.append(acc.compress(blk_bytes))

                # Append delta bytes to accumulator (after compressing)
                acc.append(blk_bytes)

        # AIT-derived attenuation: 1 - 1/(ratio * I_MAX)
        factors = [max(0.0, 1.0 - 1.0 / (r * I_MAX)) for r in ratios]

        with self._attenuation_lock:
            self._attenuation_pending = factors
            self._block_gaps = ratios

    def swap_attenuation(self):
        """Swap in attenuation factors computed by background thread.

        Called by harness before next forward.
        If pending factors are ready, they become active and pruning
        is applied based on the current ratios.
        """
        with self._attenuation_lock:
            if self._attenuation_pending is not None:
                self._attenuation_factors = self._attenuation_pending
                self._attenuation_pending = None

                # Pruning: blocks at/above RATIO_CEILING * 0.75 are fully known
                if self._block_gaps is not None:
                    use_threshold = RATIO_CEILING * 0.75
                    new_mask = torch.zeros(self.num_blocks, dtype=torch.bool)
                    for blk in range(self.num_blocks):
                        new_mask[blk] = self._block_gaps[blk] < use_threshold

                    old_kept = self._salient_count
                    self.block_mask.copy_(new_mask)
                    self._salient_count = int(new_mask.sum().item())
                    if self._salient_count != old_kept:
                        self._rebuild_delta_salient()

                self._ratio_cache = None

    # ── Forward ──

    def forward(self, x):
        """Forward: x @ W_combined  (single cuBLAS matmul).

        Builds a combined weight matrix:  base_W + scatter(delta_salient).
        Novelty attenuation is applied to delta before scattering.
        Only allocates one [in, out] temp (W_combined, ~4.7MB per layer).
        """
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        dev = x.device

        if x.dtype == torch.bfloat16:
            x_bf16 = x
        else:
            x_bf16 = x.to(torch.bfloat16)
            del x

        # Build combined weight matrix: base_W + novelty-scaled delta
        kept = self._salient_count
        if kept == 0:
            out = (x_bf16 @ self.base_W.to(dev)).float()
        elif kept == self.num_blocks:
            delta = self.delta_salient
            if delta.device != dev:
                delta = delta.to(device=dev, non_blocking=True)
            if self._attenuation_factors is not None:
                nv = torch.tensor(self._attenuation_factors, device=dev,
                                  dtype=torch.bfloat16)
                nv = nv.repeat_interleave(self.block_size)[:self.in_features].unsqueeze(1)
                W = self.base_W + delta * (1.0 - nv)
            else:
                W = self.base_W + delta
            out = (x_bf16 @ W).float()
        else:
            # Partial salience: scatter compacted delta into full weight matrix
            delta = self.delta_salient
            if delta.device != dev:
                delta = delta.to(device=dev, non_blocking=True)
            W = self.base_W.clone().to(dev)
            self._scatter_delta(W, delta, dev)
            out = (x_bf16 @ W).float()

        if self.bias is not None:
            out = out + self.bias.float()

        if len(orig_shape) == 3:
            out = out.reshape(orig_shape[0], orig_shape[1], -1)
        return out

    def _scatter_delta(self, W, delta_salient, dev):
        """Scatter compacted delta_salient into W using cached indices.

        Novelty-scaled: known blocks contribute less, novel blocks contribute
        fully.  Uses index_add_ (single GPU kernel) instead of per-block loops.
        """
        kept_idx = self._kept_indices
        K = kept_idx.numel()
        BS = self.block_size
        N = self.out_features

        # Build scatter indices lazily, cached until mask changes
        if self._scatter_indices is None or len(self._scatter_indices) != K * BS:
            idx_list = []
            for blk in kept_idx.tolist():
                fs = int(blk) * BS
                fe = min(fs + BS, self.in_features)
                idx_list.extend(range(fs, fe))
            self._scatter_indices = torch.tensor(idx_list, device=dev, dtype=torch.long)

        total_rows = min(K * BS, self.in_features)
        d_rows = delta_salient[:total_rows]

        # Apply attenuation to delta before scattering
        if self._attenuation_factors is not None:
            nv_flat = torch.tensor(
                [self._attenuation_factors[blk] for blk in kept_idx.tolist()],
                device=dev, dtype=torch.bfloat16
            )
            nv_broadcast = nv_flat.repeat_interleave(BS)[:total_rows].unsqueeze(1)
            d_rows = d_rows * (1.0 - nv_broadcast)

        W.index_add_(0, self._scatter_indices, d_rows)

    # ── post_step — delta salience update with deterministic attenuation ──

    @torch.no_grad()
    def post_step(self, threshold: float = None, calibration_multiplier: float = 0.01):
        """Update delta salience and attenuation factors.

        Compresses every block's full delta bytes against accumulated prefix,
        derives attenuation via AIT formula, and updates block mask for pruning.
        Also appends delta bytes to accumulator for future prefix compression.
        """
        self._sync_full_delta()

        delta_np = self._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
        block_el_bytes = self.block_size * self.out_features * 2
        ratios = []
        acc = self._accumulator

        for blk in range(self.num_blocks):
            byte_start = blk * block_el_bytes
            byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
            if byte_end <= byte_start:
                ratios.append(1.0)
                continue

            blk_bytes = delta_np[byte_start:byte_end].tobytes()

            # Zero-delta / cold-start: no attenuation
            if blk_bytes == b'\x00' * len(blk_bytes):
                ratios.append(1.0)
                continue

            # Prefix compression against accumulated history
            if acc is not None:
                ratios.append(acc.compress(blk_bytes))
                acc.append(blk_bytes)
            else:
                compressed = zstd.compress(blk_bytes)
                ratios.append(len(blk_bytes) / max(len(compressed), 1))

        # AIT-derived attenuation: 1 - 1/(ratio * I_MAX)
        self._attenuation_factors = [
            max(0.0, 1.0 - 1.0 / (r * I_MAX))
            for r in ratios
        ]

        # Pruning: blocks at/above RATIO_CEILING are fully known → prune
        use_threshold = threshold if threshold is not None else 8.0 * 0.75

        new_mask = torch.zeros(self.num_blocks, dtype=torch.bool)
        for blk in range(self.num_blocks):
            new_mask[blk] = ratios[blk] < use_threshold

        old_kept = self._salient_count
        self.block_mask.copy_(new_mask)
        self._salient_count = int(new_mask.sum().item())
        if self._salient_count != old_kept:
            self._rebuild_delta_salient()

        self._zstd_delta = None
        self._ratio_cache = None

    @property
    def salient_count(self) -> int:
        return self._salient_count

    @torch.no_grad()
    def stage_delta_async(self, stream=None):
        """Snapshot delta_salient to CPU for background compression.

        Stores a CPU copy in _staged_cpu for use by apply_staged_delta().
        No GPU staging buffer needed — direct .cpu() call.
        """
        self._staged_cpu = self.delta_salient.cpu()

    def apply_staged_delta(self):
        """Consume staged CPU delta data, merge into _full_delta.

        Avoids the GPU→CPU copy that _sync_full_delta normally does.
        """
        if not hasattr(self, '_staged_cpu') or self._staged_cpu is None:
            self._sync_full_delta()
            return
        if self._full_delta is None:
            self._full_delta = torch.zeros(self.in_features, self.out_features, dtype=torch.bfloat16)
        if self._salient_count == self.num_blocks:
            self._full_delta.copy_(self._staged_cpu)
        else:
            kept = self._kept_indices
            for view_idx, blk_idx in enumerate(kept.tolist()):
                vs = view_idx * self.block_size
                ve = vs + self.block_size
                fs = int(blk_idx) * self.block_size
                fe = min(fs + self.block_size, self.in_features)
                rows = fe - fs
                self._full_delta[fs:fs + rows] = self._staged_cpu[vs:vs + rows]
        self._staged_cpu = None

    def get_block_ratios(self):
        """Return per-block compression ratios, delta L2 norms, and attenuation.

        Called by diagnostic tools during post_step.  Cached until the next
        post_step (which invalidates via _ratio_cache = None).
        """
        if self._ratio_cache is not None:
            return self._ratio_cache

        # Fast path: use cached post_step data, no GPU sync
        if self._block_gaps is not None:
            gaps = self._block_gaps
            attenuations = self._attenuation_factors
            if attenuations is None:
                attenuations = [
                    max(0.0, 1.0 - 1.0 / (g * I_MAX))
                    for g in gaps
                ]

            delta_l2 = [0.0] * self.num_blocks
            if self._full_delta is not None:
                for blk in range(self.num_blocks):
                    fs = blk * self.block_size
                    fe = min(fs + self.block_size, self.in_features)
                    if fe > fs:
                        delta_l2[blk] = self._full_delta[fs:fe, :].float().norm().item()

            self._ratio_cache = {
                'ratios': list(gaps),
                'delta_l2': delta_l2,
                'block_gaps': list(gaps),
                'attenuation_scores': attenuations,
                'salient_count': self._salient_count,
                'num_blocks': self.num_blocks,
            }
            return self._ratio_cache

        # Slow path: no post_step data yet, recompute from fresh delta
        self._sync_full_delta()

        delta_np = self._full_delta.contiguous().view(torch.uint8).view(-1).numpy()
        block_el_bytes = self.block_size * self.out_features * 2
        ratios = [1.0] * self.num_blocks
        delta_l2 = [0.0] * self.num_blocks

        for blk in range(self.num_blocks):
            byte_start = blk * block_el_bytes
            byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
            if byte_end <= byte_start:
                continue
            blk_bytes = delta_np[byte_start:byte_end].tobytes()
            acc = self._accumulator
            if acc is not None:
                ratios[blk] = acc.compress(blk_bytes)
            else:
                compressed = zstd.compress(blk_bytes)
                ratios[blk] = len(blk_bytes) / max(len(compressed), 1)

            fs = blk * self.block_size
            fe = min(fs + self.block_size, self.in_features)
            if fe > fs:
                delta_l2[blk] = self._full_delta[fs:fe, :].float().norm().item()

        attenuations = [
            max(0.0, 1.0 - 1.0 / (r * I_MAX))
            for r in ratios
        ]

        self._ratio_cache = {
            'ratios': ratios,
            'delta_l2': delta_l2,
            'block_gaps': ratios,
            'attenuation_scores': attenuations,
            'salient_count': self._salient_count,
            'num_blocks': self.num_blocks,
        }
        return self._ratio_cache

    def _sync_full_delta(self):
        """Merge current delta_salient into _full_delta on CPU.

        Uses staged CPU data when available (from stage_delta_async),
        avoiding a redundant GPU→CPU copy.
        """
        if hasattr(self, '_staged_cpu') and self._staged_cpu is not None:
            delta_cpu = self._staged_cpu
            self._staged_cpu = None
        else:
            delta_cpu = self.delta_salient.cpu()

        if self._full_delta is None:
            self._full_delta = torch.zeros(self.in_features, self.out_features, dtype=torch.bfloat16)

        if self._salient_count == self.num_blocks:
            self._full_delta.copy_(delta_cpu)
        else:
            kept = self._kept_indices
            for view_idx, blk_idx in enumerate(kept.tolist()):
                vs = view_idx * self.block_size
                ve = vs + self.block_size
                fs = int(blk_idx) * self.block_size
                fe = min(fs + self.block_size, self.in_features)
                rows = fe - fs
                self._full_delta[fs:fs + rows] = delta_cpu[vs:vs + rows]

    def _rebuild_delta_salient(self):
        if self._full_delta is None:
            return

        new_kept = self._salient_count
        device = self.delta_salient.device

        self._kept_indices = self.block_mask.nonzero(as_tuple=True)[0].to(dtype=torch.long)
        self._scatter_indices = None   # invalidate — mask changed

        if new_kept == self.num_blocks:
            self.delta_salient = nn.Parameter(
                self._full_delta.clone().to(device=device),
                requires_grad=True,
            )
            return

        kept_indices = self._kept_indices
        new_size = new_kept * self.block_size
        new_view = torch.zeros(new_size, self.out_features, dtype=torch.bfloat16)

        for view_idx, blk_idx in enumerate(kept_indices.tolist()):
            vs = view_idx * self.block_size
            fs = int(blk_idx) * self.block_size
            fe = min(fs + self.block_size, self.in_features)
            rows = fe - fs
            new_view[vs:vs + rows] = self._full_delta[fs:fe]

        self.delta_salient = nn.Parameter(new_view.to(device=device), requires_grad=True)

    # ── Export ──

    @torch.no_grad()
    def export_merged(self) -> torch.Tensor:
        """Return merged weights: base_W + delta (as fp16), suitable for nn.Linear."""
        self._sync_full_delta()
        return (self.base_W + self._full_delta.to(torch.bfloat16)).t().contiguous()

    # ── Checkpoint ──

    def save_checkpoint(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._sync_full_delta()

        if self._zstd_delta is None and self._full_delta is not None:
            raw = self._full_delta.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
            self._zstd_delta = zstd.compress(raw)

        torch.save({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "has_bias": self.bias is not None,
        }, path + ".meta")

        torch.save(self.base_W.data, path + ".base_W")

        if self._zstd_delta is not None:
            with open(path + ".zstd", "wb") as f:
                f.write(self._zstd_delta)

        torch.save(self.block_mask, path + ".mask")

    @classmethod
    def load_checkpoint(cls, path: str):
        import os

        meta = torch.load(path + ".meta", weights_only=True)
        inst = cls.__new__(cls)
        nn.Module.__init__(inst)

        inst.in_features = meta["in_features"]
        inst.out_features = meta["out_features"]
        inst.block_size = meta["block_size"]
        inst.num_blocks = meta["num_blocks"]
        inst.block_mask = torch.ones(inst.num_blocks, dtype=torch.bool)
        inst._salient_count = inst.num_blocks

        # Restore delta from zstd-compressed bytes
        zstd_path = path + ".zstd"
        lz4_path = path + ".lz4"  # legacy support
        if os.path.exists(zstd_path):
            with open(zstd_path, "rb") as f:
                compressed = f.read()
            wb = zstd.decompress(compressed)
            inst._full_delta = torch.frombuffer(
                bytearray(wb), dtype=torch.uint8
            ).view(torch.bfloat16).view(inst.in_features, inst.out_features)
        elif os.path.exists(lz4_path):
            import lz4.block
            with open(lz4_path, "rb") as f:
                compressed = f.read()
            raw_size = inst.in_features * inst.out_features * 2
            wb = lz4.block.decompress(compressed, uncompressed_size=raw_size)
            inst._full_delta = torch.frombuffer(
                bytearray(wb), dtype=torch.uint8
            ).view(torch.bfloat16).view(inst.in_features, inst.out_features)
        else:
            inst._full_delta = torch.zeros(inst.in_features, inst.out_features, dtype=torch.bfloat16)

        mask_path = path + ".mask"
        if os.path.exists(mask_path):
            inst.block_mask = torch.load(mask_path, weights_only=True)
            inst._salient_count = int(inst.block_mask.sum().item())

        inst.base_W = nn.Parameter(torch.zeros(inst.in_features, inst.out_features, dtype=torch.bfloat16), requires_grad=False)
        base_path = path + ".base_W"
        if os.path.exists(base_path):
            saved = torch.load(base_path, weights_only=True)
            inst.base_W.data.copy_(saved.to(torch.bfloat16))
        inst.delta_salient = nn.Parameter(torch.zeros(inst.in_features, inst.out_features, dtype=torch.bfloat16), requires_grad=True)
        inst._rebuild_delta_salient()
        inst.bias = nn.Parameter(torch.zeros(inst.out_features, dtype=torch.bfloat16)) \
            if meta.get("has_bias", True) else None
        inst._scatter_indices = None
        inst._attenuation_factors = None
        inst._zstd_delta = None
        inst._ratio_cache = None
        inst._accumulator = None
        return inst

    # ── Conversion from nn.Linear ──

    @classmethod
    def from_linear(cls, module: nn.Linear):
        """Convert nn.Linear → frozen base + zero delta."""
        inst = cls(module.in_features, module.out_features,
                   bias=module.bias is not None)

        w = module.weight.detach().t().contiguous()
        inst.base_W.data.copy_(w.to(torch.bfloat16))
        inst._full_delta = torch.zeros(inst.in_features, inst.out_features, dtype=torch.bfloat16)

        if module.bias is not None:
            inst.bias.data.copy_(module.bias.detach().to(torch.bfloat16))

        return inst

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"delta_salient={self._salient_count}/{self.num_blocks} blocks, "
                f"bias={self.bias is not None}")
