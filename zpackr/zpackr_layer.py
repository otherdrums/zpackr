"""ZPackRLinear — frozen base + LZ4-compressed trainable delta.

Drop-in replacement for nn.Linear that stores a frozen base weight plus
an LZ4-compressed trainable delta.  Only salient blocks reside in VRAM.

Forward:  output = x @ (base_W + delta * (1 - attenuation))
             └─ frozen ─┘   └─ trainable, LZ4-compressed ─┘

Per-block LZ4 compression ratios drive delta attenuation — blocks whose
delta bytes compress well are attenuated, preventing overfitting at the
block level.  No dictionaries, no reindex, no calibration.
"""

import math
import torch
import torch.nn as nn
import lz4.block


BLOCK_SIZE = 256

# Fixed deterministic attenuation mapping constants.
# RATIO_FLOOR: below this, block is fully novel (attenuation = 0).
# RATIO_CEILING: at/above this, block is fully known (attenuation = 1).
RATIO_FLOOR = 1.0
RATIO_CEILING = 8.0

# Gate threshold: if ALL blocks across ALL layers have attenuation >= this,
# the prompt is fully converged and backward can be skipped.
ATTENUATION_SKIP_THRESHOLD = 0.9


class ZPackRLinear(nn.Module):
    """Linear layer with frozen base + LZ4-compressed trainable delta.

    CPU/pinned (authoritative):
        _full_delta:      torch.Tensor [in, out] bf16   full delta matrix
        _lz4_delta:       bytes                          LZ4-compressed delta

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
        self._lz4_delta = None       # LZ4-compressed delta bytes
        self._salient_count = self.num_blocks  # cached, updated in post_step
        self._salience_threshold = None  # pruning threshold (kept simple)

        # Cached kept-block indices (updated when mask changes)
        self._kept_indices = torch.arange(self.num_blocks, dtype=torch.long)

        # Ratio cache for diagnostic tools — invalidated on post_step
        self._ratio_cache = None

        # Per-block attenuation factors [0,1], computed from fixed constants.
        # 0.0 = fully active (novel), 1.0 = fully suppressed (known).
        self._attenuation_factors = None
        self._block_gaps = None  # cached per-block ratios

        # Delta tracking for incremental compression (skip unchanged blocks)
        self._prev_delta_l2 = [0.0] * self.num_blocks  # L2 norms from last post_step

        # Async GPU→CPU delta staging (populated by harness)
        self._delta_staging = None   # GPU buffer for D2D copy (same shape as delta_salient)

        # Cached scatter indices for fused forward matmul
        self._scatter_indices = None  # pre-built for index_add_ in forward

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

        Compresses each block's full delta bytes with LZ4, derives
        attenuation from fixed constants (RATIO_FLOOR, RATIO_CEILING),
        and updates the block mask for pruning.
        """
        self._sync_full_delta()

        delta_np = self._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
        block_el_bytes = self.block_size * self.out_features * 2
        ratios = []
        cur_l2 = []

        for blk in range(self.num_blocks):
            byte_start = blk * block_el_bytes
            byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
            if byte_end <= byte_start:
                ratios.append(1.0)
                cur_l2.append(0.0)
                continue

            fs = blk * self.block_size
            fe = min(fs + self.block_size, self.in_features)
            l2 = self._full_delta[fs:fe, :].float().norm().item() if fe > fs else 0.0
            cur_l2.append(l2)

            # Delta variance gating: skip if block delta unchanged by >15%
            prev = self._prev_delta_l2[blk]
            if prev > 0 and abs(l2 - prev) / max(prev, 1e-8) < 0.15:
                ratios.append(self._block_gaps[blk] if self._block_gaps else 1.0)
                continue

            blk_bytes = delta_np[byte_start:byte_end].tobytes()
            compressed = lz4.block.compress(blk_bytes, store_size=False)
            ratio = len(blk_bytes) / max(len(compressed), 1)
            ratios.append(ratio)

        self._prev_delta_l2 = cur_l2

        # Store ratios for future variance gating
        self._block_gaps = list(ratios)

        # Compute deterministic attenuation from fixed constants
        span = RATIO_CEILING - RATIO_FLOOR
        self._attenuation_factors = [
            max(0.0, min(1.0, (r - RATIO_FLOOR) / span))
            for r in ratios
        ]

        # Pruning: blocks at/above RATIO_CEILING are fully known → prune
        use_threshold = threshold if threshold is not None else RATIO_CEILING * 0.75

        new_mask = torch.zeros(self.num_blocks, dtype=torch.bool)
        for blk in range(self.num_blocks):
            new_mask[blk] = ratios[blk] < use_threshold

        old_kept = self._salient_count
        self.block_mask.copy_(new_mask)
        self._salient_count = int(new_mask.sum().item())
        if self._salient_count != old_kept:
            self._rebuild_delta_salient()

        self._lz4_delta = None
        self._ratio_cache = None

    @property
    def salience_threshold(self):
        return self._salience_threshold

    @property
    def salient_count(self) -> int:
        return self._salient_count

    @torch.no_grad()
    def stage_delta_async(self, stream):
        """D2D snapshot of delta_salient + launch async CPU copy on stream.

        Called by harness after optimizer.step(), before next forward.
        The CPU copy overlaps with the upcoming forward+backward on the GPU.
        """
        if self._delta_staging is None or self._delta_staging.shape != self.delta_salient.shape:
            self._delta_staging = torch.empty_like(self.delta_salient)
        self._delta_staging.copy_(self.delta_salient)  # D2D, default stream
        self._staged_cpu = self._delta_staging.to("cpu", non_blocking=False)  # sync D2H on default
        # TODO: make D2H async on dedicated stream

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
                span = RATIO_CEILING - RATIO_FLOOR
                attenuations = [
                    max(0.0, min(1.0, (g - RATIO_FLOOR) / span))
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
            compressed = lz4.block.compress(blk_bytes, store_size=False)
            ratio = len(blk_bytes) / max(len(compressed), 1)
            ratios[blk] = ratio

            fs = blk * self.block_size
            fe = min(fs + self.block_size, self.in_features)
            if fe > fs:
                delta_l2[blk] = self._full_delta[fs:fe, :].float().norm().item()

        span = RATIO_CEILING - RATIO_FLOOR
        attenuations = [
            max(0.0, min(1.0, (r - RATIO_FLOOR) / span))
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

        if self._lz4_delta is None and self._full_delta is not None:
            raw = self._full_delta.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
            self._lz4_delta = lz4.block.compress(raw, store_size=False)

        torch.save({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "has_bias": self.bias is not None,
        }, path + ".meta")

        torch.save(self.base_W.data, path + ".base_W")

        if self._lz4_delta is not None:
            with open(path + ".lz4", "wb") as f:
                f.write(self._lz4_delta)

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

        # Restore delta from LZ4-compressed bytes
        lz4_path = path + ".lz4"
        zstd_path = path + ".zstd"  # legacy support
        if os.path.exists(lz4_path):
            with open(lz4_path, "rb") as f:
                compressed = f.read()
            raw_size = inst.in_features * inst.out_features * 2
            wb = lz4.block.decompress(compressed, uncompressed_size=raw_size)
            inst._full_delta = torch.frombuffer(
                bytearray(wb), dtype=torch.uint8
            ).view(torch.bfloat16).view(inst.in_features, inst.out_features)
        elif os.path.exists(zstd_path):
            # Legacy zstd checkpoint support
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with open(zstd_path, "rb") as f:
                wb = dctx.decompress(f.read())
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
        inst._block_gaps = None
        inst._lz4_delta = None
        inst._prev_delta_l2 = [0.0] * inst.num_blocks
        inst._delta_staging = None
        inst._ratio_cache = None
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
