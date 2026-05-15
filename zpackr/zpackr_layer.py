"""ZPackRLinear — frozen base + LSH-attenuated trainable delta.

Drop-in replacement for nn.Linear that stores a frozen base weight plus
a trainable delta.  Per-block attenuation prevents overfitting by suppressing
delta contribution for blocks the model has already converged on.

Forward:  output = x @ (base_W + delta * (1 - attenuation))
             └─ frozen ─┘   └─ trainable ─┘

The attenuation signal comes from DeltaSignatureDB: a sliding window of
LSH hashes of each block's delta.  Multi-scale comparison (vs 1, 5, 10,
25, 50 steps ago) produces two signals per block:
  - mean_sim: average cosine similarity across scales (convergence level)
  - flatness: std of cosine similarity across scales (stability)
  - attenuation = mean_sim * (1 - flatness)

Pure deterministic computation.  No thresholds.  No zstd in hot path.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd
from collections import deque


BLOCK_SIZE = 256

# Gate threshold: if ALL blocks across ALL layers have attenuation >= this,
# the prompt is fully converged and backward can be skipped.
ATTENUATION_SKIP_THRESHOLD = 1.0

# Multi-scale comparison offsets (in steps)
LSH_OFFSETS = (1, 5, 10, 25, 50)


class DeltaSignatureDB:
    """Sliding window of LSH hashes for per-block convergence detection.

    Each post_step, all block deltas are hashed via LSH (sign of random
    projections) and appended to a ring buffer.  Multi-scale comparison
    against past hashes produces mean_sim and flatness signals.

    Attenuation = mean_sim * (1 - flatness).
      - Converged + stable → ~1.0 (fully attenuated)
      - Learning + varying → ~0.8 (partially attenuated)
      - Novel (no history) → 0.0 (fully active)
    """

    _projection_cache: dict = {}  # class-level: (K, block_elements) → bf16 GPU tensor

    def __init__(self, block_elements: int, num_blocks: int,
                 K: int = 64, window_size: int = 60, seed: int = 42):
        """
        Args:
            block_elements: number of bf16 elements per block (block_size * out_features)
            num_blocks: number of blocks in this layer
            K: number of LSH projection bits (more = higher resolution)
            window_size: number of hash snapshots to keep
            seed: deterministic seed for projection vectors
        """
        self.K = K
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.block_elements = block_elements

        # Sliding window of hash snapshots: deque of [num_blocks, K] uint8 tensors
        self._window: deque = deque(maxlen=window_size)

    @classmethod
    def get_gpu_projections(cls, block_elements: int, K: int = 64, seed: int = 42) -> torch.Tensor:
        """Get or create shared GPU projection matrix for a given block size.

        Returns a [K, block_elements] bf16 tensor cached on GPU, shared across
        all layers with the same block_elements.
        """
        key = (K, block_elements)
        if key not in cls._projection_cache:
            gen = torch.Generator().manual_seed(seed)
            proj = torch.randn(K, block_elements, generator=gen)
            proj = proj / proj.norm(dim=1, keepdim=True)
            cls._projection_cache[key] = proj.to(torch.float32).cuda()
        return cls._projection_cache[key]

    def push(self, hashes: torch.Tensor):
        """Append hash snapshot to the window."""
        self._window.append(hashes.clone())

    def hash_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """LSH hash all blocks.  Device-agnostic.

        Uses shared GPU projections when blocks are on CUDA,
        falls back to CPU copy when blocks are on CPU.

        Args:
            blocks: [num_blocks, block_elements] bf16 tensor

        Returns:
            [num_blocks, K] uint8 tensor (sign bits)
        """
        proj = self.get_gpu_projections(blocks.shape[1], self.K)
        if blocks.device.type == 'cpu':
            proj = proj.cpu()
        result = blocks.float() @ proj.t()
        return (result > 0).to(torch.uint8)

    def compute_attenuation(self, current_hashes: torch.Tensor) -> torch.Tensor:
        """Compute per-block attenuation from multi-scale comparison.

        Args:
            current_hashes: [num_blocks, K] uint8 tensor

        Returns:
            [num_blocks] float32 tensor, values in [0, 1]
            0.0 = fully active (novel), 1.0 = fully suppressed (converged)
        """
        n_offsets = 0
        dev = current_hashes.device
        sim_sum = torch.zeros(self.num_blocks, device=dev)
        sim_sqs = torch.zeros(self.num_blocks, device=dev)

        window_len = len(self._window)
        for off in LSH_OFFSETS:
            if off > window_len:
                break
            stored = self._window[-off].to(device=dev)
            matching = (current_hashes == stored).float().mean(dim=1)
            cos_sim = 2 * matching - 1
            sim_sum += cos_sim
            sim_sqs += cos_sim * cos_sim
            n_offsets += 1

        if n_offsets == 0:
            return torch.zeros(self.num_blocks, device=dev)

        mean_sim = sim_sum / n_offsets
        variance = sim_sqs / n_offsets - mean_sim * mean_sim
        flatness = torch.sqrt(torch.clamp(variance, min=0.0))

        attenuation = mean_sim * (1.0 - flatness)
        return torch.clamp(attenuation, 0.0, 1.0)


class ZPackRLinear(nn.Module):
    """Linear layer with frozen base + LSH-attenuated trainable delta.

    CPU/pinned (authoritative):
        _full_delta:      torch.Tensor [in, out] bf16   full delta matrix
        _zstd_delta:      bytes                          zstd-compressed delta (checkpoint only)

    GPU/VRAM:
        base_W:           torch.Tensor [in, out] bf16    frozen pretrained weight
        delta_salient:    torch.Tensor [kept*block, out] bf16  only kept blocks
        block_mask:       torch.Tensor bool[num_blocks]  which delta blocks in VRAM
        bias:             torch.Tensor [out] bf16 (optional)
    """

    def __init__(self, in_features, out_features, bias=True, lsh_K=64, lsh_window=60):
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

        self._full_delta = None       # [in, out] on CPU — populated only during checkpoint save/load
        self._zstd_delta = None       # zstd-compressed delta bytes (checkpoint only)
        self._salient_count = self.num_blocks  # cached

        # Cached kept-block indices (updated when mask changes)
        self._kept_indices = torch.arange(self.num_blocks, dtype=torch.long)

        # Ratio cache for diagnostic tools — invalidated on each hash
        self._ratio_cache = None

        # Per-block attenuation factors [0,1].
        # 0.0 = fully active (novel), 1.0 = fully suppressed (known).
        self._attenuation_factors = None
        self._block_gaps = None  # cached per-block scores (for diagnostics)

        # Cached scatter indices for fused forward matmul
        self._scatter_indices = None  # pre-built for index_add_ in forward

        # Per-layer LSH signature database (sliding window + multi-scale comparison)
        block_elements = self.block_size * out_features
        self._sig_db = DeltaSignatureDB(
            block_elements=block_elements,
            num_blocks=self.num_blocks,
            K=lsh_K,
            window_size=lsh_window,
        )

    @torch.no_grad()
    def compute_hash_gpu(self):
        """Compute LSH hash on GPU directly from delta_salient, update attenuation.

        Called after optimizer.step().  Uses shared GPU projection matrices,
        avoiding any GPU→CPU copy of the delta.  Only the tiny hash result
        (48 bytes per layer) is copied to CPU for the sliding window.
        """
        n_blocks = self.num_blocks
        n_pad = n_blocks * self.block_size - self.in_features
        padded = F.pad(self.delta_salient, (0, 0, 0, n_pad), mode='constant', value=0)
        blocks = padded.reshape(n_blocks, self.block_size * self.out_features)

        current_hashes = self._sig_db.hash_blocks(blocks)

        # Compute attenuation BEFORE pushing current hash to window
        attenuation = self._sig_db.compute_attenuation(current_hashes)

        # Push current hash into window after computing attenuation
        self._sig_db.push(current_hashes)

        self._attenuation_factors = attenuation.tolist()
        self._block_gaps = self._attenuation_factors
        self._zstd_delta = None
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

    # ── post_step — compute hash and update attenuation ──

    @torch.no_grad()
    def post_step(self, threshold: float = None, calibration_multiplier: float = 0.01):
        """Update attenuation factors from GPU hash."""
        self.compute_hash_gpu()

    @property
    def salient_count(self) -> int:
        return self._salient_count

    def get_block_ratios(self):
        """Return per-block attenuation scores, delta L2 norms, and diagnostics.

        Called by diagnostic tools during post_step.  Cached until the next
        post_step (which invalidates via _ratio_cache = None).
        """
        if self._ratio_cache is not None:
            return self._ratio_cache

        attenuations = self._attenuation_factors
        if attenuations is None:
            attenuations = [0.0] * self.num_blocks

        delta_l2 = [0.0] * self.num_blocks
        delta_cpu = self.delta_salient.cpu()
        for blk in range(self.num_blocks):
            fs = blk * self.block_size
            fe = min(fs + self.block_size, self.in_features)
            if fe > fs:
                delta_l2[blk] = delta_cpu[fs:fe, :].float().norm().item()

        self._ratio_cache = {
            'ratios': list(attenuations),
            'delta_l2': delta_l2,
            'block_gaps': list(attenuations),
            'attenuation_scores': list(attenuations),
            'salient_count': self._salient_count,
            'num_blocks': self.num_blocks,
        }
        return self._ratio_cache

    def _sync_full_delta(self):
        """Copy delta_salient from GPU to CPU _full_delta.

        Only called during checkpoint save/load, not every step.
        """
        if self._full_delta is None:
            self._full_delta = torch.zeros(self.in_features, self.out_features, dtype=torch.bfloat16)
        self._full_delta.copy_(self.delta_salient.cpu())

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
        return (self.base_W + self.delta_salient.to(torch.bfloat16)).t().contiguous()

    # ── Checkpoint ──

    def save_checkpoint(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Copy delta to CPU once (only at checkpoint time)
        delta_cpu = self.delta_salient.cpu()

        raw = delta_cpu.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        zstd_compressed = zstd.compress(raw)

        torch.save({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "has_bias": self.bias is not None,
        }, path + ".meta")

        torch.save(self.base_W.data, path + ".base_W")

        with open(path + ".zstd", "wb") as f:
            f.write(zstd_compressed)

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
        # Initialize LSH signature database
        block_elements = inst.block_size * inst.out_features
        inst._sig_db = DeltaSignatureDB(
            block_elements=block_elements,
            num_blocks=inst.num_blocks,
        )
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
