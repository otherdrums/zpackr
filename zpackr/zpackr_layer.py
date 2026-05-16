"""ZPackRLinear — frozen base + row-level dual-signal LSH-attenuated delta.

Uses TWO LSH signals per row:
  - delta hash:  direction stability of the delta weight (position convergence)
  - gradient hash: direction stability of the gradient (learning signal SNR)

Attenuation is a geometric mix of both:
  atten = delta_sim ** (1 - mix) * (1 - grad_sim) ** mix

Delta alone can't distinguish "converged" from "stuck" (both have stable
positions).  Gradient alone can't distinguish "learning" from "converged"
(converged gradient is noise = unstable).  Together they form a complete
signal: attenuation rises only when BOTH agree the row is done.

Forward:  output = x @ (base_W + delta * (1 - attenuation))
             └─ frozen ─┘   └─ trainable ─┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd

import triton
import triton.language as tl


BLOCK_SIZE = 256

# Gate threshold: if ALL rows across ALL layers have attenuation >= this,
# the prompt is fully converged and backward can be skipped.
ATTENUATION_SKIP_THRESHOLD = 1.0

# Multi-scale comparison offsets (in steps) — logarithmic, 3x spacing
LSH_OFFSETS = (1, 3, 10, 30, 100, 300, 1000)

# Exponential weights for far-offset dominance.
# Near offsets (1, 3) are always ~1.0 since consecutive hashes barely change;
# they dominate the mean and mask the far-offset signal.  These weights
# suppress near offsets and amplify far ones so attenuation reflects
# true long-term convergence, not short-term direction consistency.
# Weights grow as 2^{idx}, aligned with log-spaced offsets.
LSH_WEIGHTS = (1, 1, 2, 4, 8, 16, 32)


@triton.jit
def _lsh_hash_fused_kernel(
    delta_ptr,     # [in_features, out_features] bf16 row-major
    proj_ptr,      # [K, out_features] bf16 row-major
    hash_ptr,      # [in_features, K] uint8 output
    in_features,
    out_features,
    K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    """Fused Triton kernel: all K LSH hashes for one delta row.

    Grid: (in_features,) — one block per row processes ALL K projections.
    Delta[row] is loaded ONCE per chunk instead of K times, cutting
    delta memory traffic by 16× vs the old 2D (in_features, K) grid.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_OUT)

    # K per-projection accumulators (fits in float32 registers for K=16)
    acc = tl.zeros([K], dtype=tl.float32)

    for start in range(0, out_features, BLOCK_OUT):
        o = start + offs
        mask = o < out_features
        d = tl.load(delta_ptr + row * out_features + o, mask=mask).to(tl.float32)

        for k in range(K):
            p = tl.load(proj_ptr + k * out_features + o, mask=mask).to(tl.float32)
            dot = tl.sum(d * p)
            acc = tl.where(tl.arange(0, K) == k, acc + dot, acc)

    # Store all K hash bits contiguously (avoids per-element indexing)
    tl.store(hash_ptr + row * K + tl.arange(0, K), (acc > 0).to(tl.uint8))


class DeltaSignatureDB:
    """Sliding window of LSH hashes for per-row convergence detection.

    Each step, all delta (or gradient) rows are LSH-hashed via Triton kernel and
    appended to a ring buffer stored in pinned CPU memory.  Multi-scale comparison
    against past hashes produces the weighted mean of cosine similarities.

    Used TWICE per ZPackRLinear: one instance for delta hashes (position stability)
    and one for gradient hashes (learning signal SNR).  The two signals are mixed
    geometrically into the final attenuation.

    The window lives on CPU (pinned for async GPU↔CPU transfers) to save
    ~369MB of GPU VRAM.  Only the needed offsets (~644KB) are transferred
    to GPU during compute_attenuation.

    Attenuation = exponentially weighted mean of cosine similarities across
    all valid window offsets.  Far offsets (100, 300, 1000) dominate over
    near offsets (1, 3) via LSH_WEIGHTS = (1, 1, 2, 4, 8, 16, 32).
      - Converged + stable → ~255 (fully attenuated)
      - Learning + varying → ~64-128 (partially attenuated)
      - Novel (no history) → 0 (fully active)
    """

    _projection_cache: dict = {}  # class-level: (K, out_features) → float32 GPU tensor

    def __init__(self, num_rows: int, K: int = 16, window_size: int = 4200, seed: int = 42):
        self.K = K
        self.bytes_per_hash = K // 8  # 2 bytes for K=16
        self.num_rows = num_rows
        self._window_size = window_size
        # Pinned CPU ring buffer — saves 369MB GPU VRAM vs GPU window
        self._window_cpu = torch.zeros(window_size, num_rows, self.bytes_per_hash,
                                       dtype=torch.uint8, pin_memory=True)
        self._cursor = 0   # next write position
        self._count = 0    # entries written (capped at window_size)

    @classmethod
    def get_gpu_projections(cls, out_features: int, K: int = 64, seed: int = 42) -> torch.Tensor:
        """Get or create shared GPU projection matrix for a given out_features.

        Returns a [K, out_features] float32 tensor cached on GPU, shared across
        all layers with the same out_features.
        """
        key = (K, out_features)
        if key not in cls._projection_cache:
            gen = torch.Generator().manual_seed(seed)
            proj = torch.randn(K, out_features, generator=gen)
            proj = proj / proj.norm(dim=1, keepdim=True)
            cls._projection_cache[key] = proj.to(torch.float32).cuda()
        return cls._projection_cache[key]

    def hash_rows(self, delta: torch.Tensor) -> torch.Tensor:
        """LSH hash all rows and pack bits into bytes.

        Returns packed uint8 tensor of shape [in_features, K//8].
        Each byte packs 8 LSH bits (bit 0 = projection k, bit 7 = k+7).
        """
        in_f, out_f = delta.shape
        proj = self.get_gpu_projections(out_f, self.K)

        if delta.device.type == 'cpu':
            proj = proj.cpu()
            result = delta.float() @ proj.t()
            bits = (result > 0).to(torch.uint8).cuda()
        else:
            bits = torch.empty(in_f, self.K, dtype=torch.uint8, device='cuda')
            grid = (in_f,)
            _lsh_hash_fused_kernel[grid](
                delta, proj, bits,
                in_f, out_f,
                K=self.K,
                BLOCK_OUT=BLOCK_SIZE,
            )

        # Pack 8 bits per byte
        bits_view = bits.view(in_f, self.bytes_per_hash, 8)
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device='cuda', dtype=torch.uint8)
        return (bits_view * weights).sum(dim=2).to(torch.uint8)

    def push(self, hashes: torch.Tensor):
        """Append hash snapshot to the pinned CPU ring buffer."""
        self._window_cpu[self._cursor].copy_(hashes, non_blocking=True)
        self._cursor = (self._cursor + 1) % self._window_size
        self._count = min(self._count + 1, self._window_size)

    def compute_attenuation(self, current_hashes: torch.Tensor) -> torch.Tensor:
        """Compute per-row attenuation from multi-scale comparison.

        Hash history lives in pinned CPU memory to save GPU VRAM.
        Needed offsets are transferred to GPU in a single batched
        operation (~644KB, ~40μs on PCIe 3.0).

        Uses continuous byte comparison: 1 - |hash - stored| / 255 per byte.
        Each byte gives 256 levels of similarity — far finer than unpacked
        bit-by-bit matching (which gives K+1 levels).  With K=16 packed into
        2 bytes, effective resolution is ~512 levels per offset.

        Attenuation is the exponentially weighted mean of cosine similarities
        across valid window offsets — far offsets (100, 300, 1000) dominate
        over near offsets (1, 3) so the signal reflects true long-term
        convergence rather than short-term direction consistency.

        Returns:
            [in_features] float32 tensor, values in [0, 1]
        """
        count = self._count

        # Collect valid offset indices and their exponential weights
        indices = []
        weights_list = []
        for i, off in enumerate(LSH_OFFSETS):
            if off > count:
                break
            indices.append((self._cursor - off) % self._window_size)
            weights_list.append(LSH_WEIGHTS[i])

        n_offsets = len(indices)
        if n_offsets == 0:
            return torch.zeros(self.num_rows, device='cuda')

        # Batched transfer CPU pinned → GPU (non_blocking avoids Python sync)
        stored_slices = [self._window_cpu[i].cuda(non_blocking=True) for i in indices]
        stored = torch.stack(stored_slices).float()  # [n_off, num_rows, bytes] float32

        # Batched continuous byte comparison across all offsets
        current = current_hashes.unsqueeze(0).float()  # [1, num_rows, bytes]
        diff = (current - stored).abs()
        byte_sim = 1.0 - diff / 255.0
        matching = byte_sim.mean(dim=2)  # [n_off, num_rows]
        cos_sim = 2 * matching - 1       # map [0,1] → [-1,1]

        # Exponentially weighted mean: far offsets dominate so the signal
        # reflects long-term convergence, not short-term direction stability.
        weights_t = torch.tensor(weights_list, device='cuda', dtype=torch.float32)
        attenuation = (cos_sim * weights_t.unsqueeze(1)).sum(dim=0) / weights_t.sum()

        return torch.clamp(attenuation, 0.0, 1.0)


class ZPackRLinear(nn.Module):
    """Linear layer with frozen base + dual-signal LSH-attenuated trainable delta.

    Uses TWO DeltaSignatureDB instances:
      - _sig_db:         delta hashes (weight position stability)
      - _grad_sig_db:    gradient hashes (learning signal SNR)

    Attenuation = delta_sim ** (1 - gradient_mix) * (1 - grad_sim) ** gradient_mix
    Both must agree the row is done for full attenuation (byte=255).

    GPU/VRAM:
        base_W:           torch.Tensor [in, out] bf16   frozen pretrained weight
        delta_salient:    torch.Tensor [in, out] bf16    trainable delta
        _atten_byte:      torch.Tensor [in] uint8        per-row attenuation (0-255)
        _grad_sim:        torch.Tensor [in] float32      cached gradient similarity
    """

    def __init__(self, in_features, out_features, bias=True, lsh_K=16,
                 lsh_window=4200, hash_interval=1, gradient_mix=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._hash_interval = hash_interval
        self._hash_counter = 0
        self._gradient_mix = gradient_mix

        # Frozen base — stored in bf16 for direct matmul compatibility
        self.base_W = nn.Parameter(
            torch.zeros(in_features, out_features, dtype=torch.bfloat16),
            requires_grad=False,
        )

        # Delta starts as all-zeros, all rows active
        self.delta_salient = nn.Parameter(
            torch.zeros(in_features, out_features, dtype=torch.bfloat16),
            requires_grad=True,
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features, dtype=torch.bfloat16)
        ) if bias else None

        # Per-row attenuation as uint8 GPU tensor [in_features] — 256 levels
        self.register_buffer('_atten_byte',
            torch.zeros(in_features, dtype=torch.uint8))

        # Cached gradient similarity for mixing (updated by compute_grad_hash)
        self.register_buffer('_grad_sim',
            torch.zeros(in_features, dtype=torch.float32))

        # Ratio cache for diagnostic tools
        self._ratio_cache = None

        # Delta signature database (position stability)
        self._sig_db = DeltaSignatureDB(
            num_rows=in_features,
            K=lsh_K,
            window_size=lsh_window,
        )

        # Gradient signature database (learning signal SNR)
        self._grad_sig_db = DeltaSignatureDB(
            num_rows=in_features,
            K=lsh_K,
            window_size=lsh_window,
        )

    @torch.no_grad()
    def compute_grad_hash(self):
        """Hash the gradient, update _grad_sim.

        Must be called after backward() and BEFORE optimizer.step()/zero_grad().
        Hashes delta_salient.grad via the Triton kernel into a gradient
        signature window, computes exponentially-weighted similarity, and
        caches it in _grad_sim for the next compute_hash_gpu call.

        Safe to call when grad is None (first step, or before first backward):
        _grad_sim is left at zero (no gradient contribution to mixing).
        """
        grad = self.delta_salient.grad
        if grad is None:
            self._grad_sim.zero_()
            return

        current_hashes = self._grad_sig_db.hash_rows(grad)
        grad_sim = self._grad_sig_db.compute_attenuation(current_hashes)
        self._grad_sim.copy_(grad_sim)
        self._grad_sig_db.push(current_hashes)

    @torch.no_grad()
    def compute_hash_gpu(self):
        """Compute delta LSH hash, mix with cached gradient signal, update atten.

        Called after optimizer.step().  Delta hashing runs at hash_interval
        (gradient hashing runs every step via compute_grad_hash).

        The final attenuation is a geometric mix:
          atten = delta_sim ** (1 - mix) * (1 - grad_sim) ** mix

        Where:
          - delta_sim = weighted mean of delta cosine similarities (position)
          - grad_sim  = weighted mean of gradient cosine similarities (SNR)
          - mix       = gradient_mix knob (0 = pure delta, 1 = pure gradient)

        When delta hash is skipped (hash_interval > 1), the mixed attenuation
        from the last interval step is reused — the cached grad_sim still
        evolves every step via compute_grad_hash but the delta component
        only updates at interval boundaries.
        """
        self._hash_counter += 1
        if self._hash_counter < self._hash_interval:
            return
        self._hash_counter = 0

        # Delta hash — position stability signal
        current_hashes = self._sig_db.hash_rows(self.delta_salient)
        delta_sim = self._sig_db.compute_attenuation(current_hashes)
        self._sig_db.push(current_hashes)

        # Geometric mix: both signals must agree the row is done
        mix = self._gradient_mix
        attenuation = delta_sim.pow(1.0 - mix) * (1.0 - self._grad_sim).pow(mix)

        self._atten_byte.copy_((attenuation * 255).to(dtype=torch.uint8))
        self._ratio_cache = None

    # ── Forward ──

    def forward(self, x):
        """Forward: x @ W_combined  (single cuBLAS matmul).

        Builds a combined weight matrix:  base_W + delta * (1 - nv).
        Per-row attenuation applied via _atten_byte (uint8 → float).
        """
        orig_shape = x.shape
        orig_dtype = x.dtype
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        if x.dtype == torch.bfloat16:
            x_bf16 = x
        else:
            x_bf16 = x.to(torch.bfloat16)
            del x

        # Per-row attenuation: [in_features] uint8 → [in_features, 1] bf16
        dev = self.delta_salient.device
        nv = (self._atten_byte.float() / 255.0).to(torch.bfloat16).unsqueeze(1)
        W = self.base_W.to(dev) + self.delta_salient * (1.0 - nv)

        out = x_bf16 @ W
        if self.bias is not None:
            out = out + self.bias

        # Preserve input dtype — ZPackRLinear is dtype-agnostic
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        if len(orig_shape) == 3:
            out = out.reshape(orig_shape[0], orig_shape[1], -1)
        return out

    # ── post_step — compute delta hash and update attenuation ──

    @torch.no_grad()
    def post_step(self, threshold: float = None, calibration_multiplier: float = 0.01):
        """Update attenuation factors from delta hash + cached gradient signal."""
        self.compute_hash_gpu()

    def get_block_ratios(self):
        """Return per-row attenuation scores and delta L2 norms.

        Called by diagnostic tools.  Cached until next hash.
        """
        if self._ratio_cache is not None:
            return self._ratio_cache

        attenuations = self._atten_byte.float().tolist()
        if not attenuations:
            attenuations = [0.0] * self.in_features

        delta_l2 = self.delta_salient.float().norm(dim=1).tolist()

        self._ratio_cache = {
            'ratios': list(attenuations),
            'delta_l2': delta_l2,
            'block_gaps': list(attenuations),
            'attenuation_scores': [a / 255.0 for a in attenuations],
            'salient_count': self.in_features,
            'num_blocks': self.in_features,
        }
        return self._ratio_cache

    # ── Checkpoint ──

    def save_checkpoint(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        delta_cpu = self.delta_salient.cpu()
        raw_bytes = delta_cpu.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        zstd_compressed = zstd.compress(raw_bytes)

        torch.save({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.bias is not None,
            "gradient_mix": self._gradient_mix,
        }, path + ".meta")

        torch.save(self.base_W.data, path + ".base_W")

        with open(path + ".zstd", "wb") as f:
            f.write(zstd_compressed)

    @classmethod
    def load_checkpoint(cls, path: str):
        import os

        meta = torch.load(path + ".meta", weights_only=True)
        inst = cls.__new__(cls)
        nn.Module.__init__(inst)

        inst.in_features = meta["in_features"]
        inst.out_features = meta["out_features"]

        # Restore delta from zstd-compressed bytes
        zstd_path = path + ".zstd"
        if os.path.exists(zstd_path):
            with open(zstd_path, "rb") as f:
                compressed = f.read()
            wb = zstd.decompress(compressed)
            inst.delta_salient = nn.Parameter(
                torch.frombuffer(bytearray(wb), dtype=torch.uint8)
                .view(torch.bfloat16)
                .view(inst.in_features, inst.out_features),
                requires_grad=True,
            )
        else:
            inst.delta_salient = nn.Parameter(
                torch.zeros(inst.in_features, inst.out_features, dtype=torch.bfloat16),
                requires_grad=True,
            )

        inst.base_W = nn.Parameter(
            torch.zeros(inst.in_features, inst.out_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        base_path = path + ".base_W"
        if os.path.exists(base_path):
            saved = torch.load(base_path, weights_only=True)
            inst.base_W.data.copy_(saved.to(torch.bfloat16))

        inst.bias = nn.Parameter(torch.zeros(inst.out_features, dtype=torch.bfloat16)) \
            if meta.get("has_bias", True) else None

        inst.register_buffer('_atten_byte',
            torch.zeros(inst.in_features, dtype=torch.uint8,
                        device=inst.delta_salient.device))
        inst.register_buffer('_grad_sim',
            torch.zeros(inst.in_features, dtype=torch.float32,
                        device=inst.delta_salient.device))
        inst._ratio_cache = None
        inst._gradient_mix = meta.get("gradient_mix", 0.5)
        inst._sig_db = DeltaSignatureDB(
            num_rows=inst.in_features,
        )
        inst._grad_sig_db = DeltaSignatureDB(
            num_rows=inst.in_features,
        )
        return inst

    # ── Conversion from nn.Linear ──

    @classmethod
    def from_linear(cls, module: nn.Linear, hash_interval: int = 1, gradient_mix: float = 0.5):
        """Convert nn.Linear → frozen base + zero delta."""
        inst = cls(module.in_features, module.out_features,
                   bias=module.bias is not None,
                   hash_interval=hash_interval,
                   gradient_mix=gradient_mix)

        w = module.weight.detach().t().contiguous()
        inst.base_W.data.copy_(w.to(torch.bfloat16))

        if module.bias is not None:
            inst.bias.data.copy_(module.bias.detach().to(torch.bfloat16))

        return inst

    @torch.no_grad()
    def export_merged(self) -> torch.Tensor:
        """Return merged weights: base_W + delta, suitable for nn.Linear."""
        return (self.base_W + self.delta_salient.to(torch.bfloat16)).t().contiguous()

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}")
