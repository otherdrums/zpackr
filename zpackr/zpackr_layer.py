"""ZPackRLinear — dual-dict linear layer (ZPackR v2.0).

Drop-in replacement for nn.Linear that stores a frozen base weight plus
a zstd-compressed trainable delta.  Only salient blocks of the delta
reside in VRAM.

Forward:  output = x @ W_combined  (single cuBLAS matmul)
             └─ frozen ─┘        └─ trainable, zstd-compressed ─┘

post_step() uses a self-calibrating per-layer threshold: the threshold
ratchets up as the WeightDict learns to compress delta patterns better.
Blocks that become more compressible than the current baseline are pruned.
"""

import math
import torch
import torch.nn as nn


BLOCK_SIZE = 256


class ZPackRLinear(nn.Module):
    """Linear layer with frozen base + zstd-compressed trainable delta.

    CPU/pinned (authoritative):
        _full_delta:      torch.Tensor [in, out] bf16   full delta matrix
        zstd_delta:       bytes                          compressed delta

    GPU/VRAM:
        base_W:           torch.Tensor [in, out] fp16    frozen pretrained weight
        delta_salient:    torch.Tensor [kept*block, out] bf16  only kept blocks
        block_mask:       torch.Tensor bool[num_blocks]  which delta blocks in VRAM
        bias:             torch.Tensor [out] bf16 (optional)

    Salience state:
        _salient_count:   int  cached count of kept blocks (no GPU sync)
        _salience_threshold: float | None  per-layer auto-calibrated threshold
    """

    def __init__(self, in_features, out_features, weight_dict, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = BLOCK_SIZE
        self.num_blocks = math.ceil(in_features / self.block_size)
        self.weight_dict = weight_dict

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
        self._zstd_delta = None       # compressed delta bytes
        self._salient_count = self.num_blocks  # cached, updated in post_step
        self._salience_threshold = None  # auto-calibrated on first post_step
        self._calibration_max = None     # recorded at calibration time

        # Cached kept-block indices (updated when mask changes)
        self._kept_indices = torch.arange(self.num_blocks, dtype=torch.long)

        # Ratio cache for diagnostic tools — invalidated on post_step
        self._ratio_cache = None

        # Per-block novelty system (ratio → novelty → attenuation + decay)
        self._block_gaps = None                       # set on first post_step
        self._novelty_scores = None                   # derived from gaps, clamped [0,1]
        self._gap_hist_max = 1.0                      # historical max ratio (zero deltas)
        self._gap_hist_min = 1.0                      # historical min ratio (trained)
        self._gap_enabled = True                      # toggle per layer
        self._gap_decay_rate = 0.01                   # 1% per step for fully known blocks

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
            if self._novelty_scores is not None and self._block_gaps is not None:
                nv = torch.tensor(self._novelty_scores, device=dev,
                                  dtype=torch.bfloat16)
                nv = nv.repeat_interleave(self.block_size)[:self.in_features].unsqueeze(1)
                W = self.base_W + delta * nv
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

        # Apply novelty attenuation to delta before scattering
        if (self._novelty_scores is not None and self._block_gaps is not None):
            nv_flat = torch.tensor(
                [self._novelty_scores[blk] for blk in kept_idx.tolist()],
                device=dev, dtype=torch.bfloat16
            )
            nv_broadcast = nv_flat.repeat_interleave(BS)[:total_rows].unsqueeze(1)
            d_rows = d_rows * nv_broadcast

        W.index_add_(0, self._scatter_indices, d_rows)

    # ── post_step — delta salience update with auto-calibrating threshold ──

    @torch.no_grad()
    def post_step(self, threshold: float = None, calibration_multiplier: float = 0.01):
        """Update delta salience.  Uses self-calibrating per-layer threshold.

        On first call (or first after reindex): auto-calibrates threshold
        at calibration_multiplier % of the maximum observed ratio.
        Blocks must exceed this baseline to be pruned.

        Skips calibration when all ratios are uniform (e.g., zero-delta).
        """
        self._sync_full_delta()

        delta_np = self._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
        block_el_bytes = self.block_size * self.out_features * 2
        # Sub-sample: only compress first 256KB of each block (ratio is homogenous)
        sample_bytes = min(block_el_bytes, 256 * 1024)
        ratios = []
        cur_l2 = []
        batch_bytes = []       # for bulk zstd
        batch_indices = []     # which blocks need fresh compression

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

            blk_bytes = delta_np[byte_start:byte_start + sample_bytes].tobytes()
            batch_bytes.append(blk_bytes)
            batch_indices.append(blk)
            ratios.append(0.0)  # placeholder, filled below

        # Bulk zstd compression: all blocks in one C-level call (GIL released)
        if batch_bytes and not self.weight_dict.is_empty:
            batch_ratios = self.weight_dict.batch_ratios(batch_bytes)
            for idx, ratio in zip(batch_indices, batch_ratios):
                ratios[idx] = ratio
        # Fill any remaining placeholders (unchanged blocks already have values)
        for i in range(len(ratios)):
            if ratios[i] == 0.0:
                ratios[i] = 1.0

        self._prev_delta_l2 = cur_l2

        # Per-block gaps = weight_ratio (higher = more compressible = known)
        self._block_gaps = list(ratios)
        self._compute_novelty()
        # Auto-calibrate threshold on first post_step (or first after reindex).
        # Strategy: find if there are two clusters (zero-delta vs trained-delta).
        # When the max ratio ≥ 2x the min ratio, zero-delta blocks co-exist with
        # trained blocks — set threshold in the gap.  Otherwise keep everything.
        if self._salience_threshold is None and ratios:
            if not self.weight_dict.is_empty:
                cal_max = max(ratios)
                cal_min = min(ratios)
                gap_ratio = cal_max / max(cal_min, 1e-8)
                if gap_ratio >= 2.0:  # Bimodal: zero-delta blocks present
                    self._calibration_max = cal_max
                    # Put threshold 30% up from trained baseline toward zero-delta peak
                    self._salience_threshold = cal_min + (cal_max - cal_min) * 0.3
            else:
                self._salience_threshold = 0.0  # No dict → keep all
            if threshold is not None:
                self._salience_threshold = threshold
            self._zstd_delta = None
            self._ratio_cache = None
            return  # No pruning on calibration pass (threshold set or deferred)

        use_threshold = self._salience_threshold if self._salience_threshold is not None else (threshold or 1.4)

        new_mask = torch.zeros(self.num_blocks, dtype=torch.bool)
        for blk in range(self.num_blocks):
            # ratio < threshold → novel (low compressibility) → KEEP
            new_mask[blk] = ratios[blk] < use_threshold

        old_kept = self._salient_count
        self.block_mask.copy_(new_mask)
        self._salient_count = int(new_mask.sum().item())
        if self._salient_count != old_kept:
            self._rebuild_delta_salient()

        # Don't ratchet — threshold stays at calibration baseline until next reindex.
        # The calibration baseline captures the "learned" compressibility floor.
        # Any block falling BELOW (less compressible = novel) is kept.

        self._zstd_delta = None
        self._ratio_cache = None

    @property
    def salience_threshold(self):
        return self._salience_threshold

    @property
    def salient_count(self) -> int:
        return self._salient_count

    def _compute_novelty(self):
        """Map _block_gaps → per-block novelty scores [0,1].

        Uses historical max/min ratio range for stable normalization.
        When zero-delta blocks exist (high ratio ≈ 6), novelties are
        well-separated.  When all blocks are uniform, novelty is ≥ 0.5
        for all (conservative — don't decay active blocks).

        novelty = (gap_max_hist - ratio) / span, clamped [0,1].
        Low ratio (compresses poorly = novel) → high novelty.
        High ratio (compresses well = known) → low novelty.
        """
        if self._block_gaps is None:
            return
        gaps = self._block_gaps
        gap_max = max(gaps)
        gap_min = min(gaps)
        # Track historical range for stable normalization
        self._gap_hist_max = max(self._gap_hist_max, gap_max)
        self._gap_hist_min = min(self._gap_hist_min, gap_min)
        span = max(self._gap_hist_max - self._gap_hist_min, 0.1)
        self._novelty_scores = [
            max(0.0, min(1.0, (self._gap_hist_max - g) / span))
            for g in gaps
        ]

    @torch.no_grad()
    def shrink_known_delta(self):
        """Decay known blocks toward zero, driven by novelty scores.

        Called after optimizer.step() + zero_grad(), before next forward.
        Known (novelty=0): shrinks by _gap_decay_rate each step.
        Novel (novelty=1): no decay.
        In-between: proportional decay.
        """
        if (not self._gap_enabled or self._block_gaps is None
                or self._novelty_scores is None):
            return
        BS = self.block_size
        kept_idx = self._kept_indices
        for view_idx, blk in enumerate(kept_idx.tolist()):
            novelty = self._novelty_scores[blk]
            decay = (1.0 - novelty) * self._gap_decay_rate
            if decay > 0:
                vs = view_idx * BS
                ve = vs + BS
                rows = min(BS, self.in_features - int(blk) * BS)
                self.delta_salient[vs:vs + rows] *= (1.0 - decay)

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
        """Return per-block compression ratios, delta L2 norms, and calibration state.

        Called by diagnostic tools during post_step.  Cached until the next
        post_step (which invalidates via _ratio_cache = None).

        When _block_gaps is available (post_step has run), returns cached
        instance data without GPU→CPU sync.  Only recomputes from scratch
        before the first post_step.

        Returns:
            dict with keys:
                ratios:              list[float]  compression ratio per block
                delta_l2:            list[float]  L2 norm of delta per block
                block_gaps:          list[float]  same as ratios (cached)
                novelty_scores:      list[float]  [0,1] per block
                calibration_max:     float | None
                calibrated_threshold: float | None
                salient_count:       int
                num_blocks:          int
        """
        if self._ratio_cache is not None:
            return self._ratio_cache

        # Fast path: use cached post_step data, no GPU sync
        if self._block_gaps is not None:
            gaps = self._block_gaps
            novelties = self._novelty_scores
            if novelties is None:
                gap_max = max(gaps)
                gap_min = min(gaps)
                span = max(gap_max - gap_min, 0.1)
                novelties = [max(0.0, min(1.0, (gap_max - g) / span)) for g in gaps]

            # delta_l2 from last _full_delta (may be stale between post_steps)
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
                'novelty_scores': novelties,
                'calibration_max': self._calibration_max,
                'calibrated_threshold': self._salience_threshold,
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
            ratio = self.weight_dict.ratio(blk_bytes) if not self.weight_dict.is_empty else 1.0
            ratios[blk] = ratio

            fs = blk * self.block_size
            fe = min(fs + self.block_size, self.in_features)
            if fe > fs:
                delta_l2[blk] = self._full_delta[fs:fe, :].float().norm().item()

        self._ratio_cache = {
            'ratios': ratios,
            'delta_l2': delta_l2,
            'block_gaps': ratios,
            'novelty_scores': [1.0] * self.num_blocks,
            'calibration_max': self._calibration_max,
            'calibrated_threshold': self._salience_threshold,
            'salient_count': self._salient_count,
            'num_blocks': self.num_blocks,
        }
        return self._ratio_cache
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

    # ── Reindex ──

    @torch.no_grad()
    def reindex(self, min_frequency: float = 0.01, min_count: int = 10):
        """Evolve WeightDict from delta patterns.

        Resets the self-calibrating threshold — next post_step will
        recalibrate against the updated dictionary.
        """
        if self._full_delta is None:
            self._full_delta = self.delta_salient.cpu()
        delta_bytes = self._full_delta.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        result = self.weight_dict.reindex(delta_bytes, min_frequency=min_frequency, min_count=min_count)
        self._salience_threshold = None  # Recalibrate on next post_step
        self._calibration_max = None
        self._block_gaps = None           # Recompute on next post_step
        self._novelty_scores = None
        self._gap_hist_max = 1.0          # Reset historical range
        self._gap_hist_min = 1.0
        self._prev_delta_l2 = [0.0] * self.num_blocks
        self._ratio_cache = None
        return result

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
            self._zstd_delta = self.weight_dict.compress(
                self._full_delta.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
            )

        torch.save({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "has_bias": self.bias is not None,
            "salience_threshold": self._salience_threshold,
        }, path + ".meta")

        torch.save(self.base_W.data, path + ".base_W")

        if self._zstd_delta is not None:
            with open(path + ".zstd", "wb") as f:
                f.write(self._zstd_delta)

        torch.save(self.block_mask, path + ".mask")
        self.weight_dict.save(path + ".wd")

    @classmethod
    def load_checkpoint(cls, path: str, weight_dict):
        import os

        meta = torch.load(path + ".meta", weights_only=True)
        inst = cls.__new__(cls)
        nn.Module.__init__(inst)

        inst.in_features = meta["in_features"]
        inst.out_features = meta["out_features"]
        inst.block_size = meta["block_size"]
        inst.num_blocks = meta["num_blocks"]
        inst.weight_dict = weight_dict
        inst.block_mask = torch.ones(inst.num_blocks, dtype=torch.bool)
        inst._salience_threshold = meta.get("salience_threshold")
        inst._salient_count = inst.num_blocks

        zstd_path = path + ".zstd"
        if os.path.exists(zstd_path):
            with open(zstd_path, "rb") as f:
                inst._zstd_delta = f.read()
            wb = weight_dict.decompress(inst._zstd_delta)
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
        inst._novelty_scores = None
        inst._block_gaps = None
        inst._gap_hist_max = 1.0
        inst._gap_hist_min = 1.0
        inst._gap_enabled = True
        inst._gap_decay_rate = 0.01
        inst._prev_delta_l2 = [0.0] * inst.num_blocks
        inst._delta_staging = None
        inst._ratio_cache = None
        inst._calibration_max = None
        inst._ratio_thread = None
        return inst

    # ── Conversion from nn.Linear ──

    @classmethod
    def from_linear(cls, module: nn.Linear, weight_dict):
        """Convert nn.Linear → frozen base + zero delta."""
        inst = cls(module.in_features, module.out_features, weight_dict,
                   bias=module.bias is not None)

        w = module.weight.detach().t().contiguous()
        inst.base_W.data.copy_(w.to(torch.bfloat16))
        inst._full_delta = torch.zeros(inst.in_features, inst.out_features, dtype=torch.bfloat16)

        if module.bias is not None:
            inst.bias.data.copy_(module.bias.detach().to(torch.bfloat16))

        return inst

    def extra_repr(self):
        thresh = f", threshold={self._salience_threshold:.3f}" if self._salience_threshold else ""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"delta_salient={self._salient_count}/{self.num_blocks} blocks{thresh}, "
                f"bias={self.bias is not None}")
