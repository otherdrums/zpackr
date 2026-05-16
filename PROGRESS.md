# ZPackR Progress Notes — May 2026

## Reference
- **System flowchart**: [FLOWCHART.md](FLOWCHART.md) — per-block lifecycle, signal chain
- **Implementation guide**: [ROADMAP.md](ROADMAP.md)

## Architecture Evolution

| Version | Signal | Computation | State | Status |
|---------|--------|-------------|-------|--------|
| v1 (initial) | zstd + WeightDict | CPU threaded | Dictionary + reindex | **Superseded** |
| v2 (LZ4, May 13) | LZ4 per-block | CPU | None | **Superseded** — ratios < 1.0 |
| v3 (zstd, May 14) | zstd per-block | CPU background thread | None | **Superseded** — entropy floor |
| v4 (LSH, May 14-15) | LSH multi-scale sliding window | GPU synchronous (~1ms) | Per-block ring buffer (60 steps) | **Superseded** — block-level caps accuracy |
| v5 (Row LSH, May 15) | Per-row LSH + Triton kernel + uint8 + log-spaced offsets | GPU Triton kernel (K=32) | Per-row ring buffer (1100 steps) | **Superseded** |
| v6 (CPU Window, May 15) | CPU-pinned window + batched GPU compute + optional bf16 | GPU Triton kernel (K=16) | Per-row CPU ring buffer (4200 steps) | **Superseded** |
| v7 (Fused Hash, May 15) | Fused 1D hash kernel + hash interval + dtype-agnostic forward | GPU Triton (1D grid, K=16 fused) | Per-row CPU ring buffer (4200 steps) | **Superseded** |
| v8 (CUDA Optimizer, May 15) | Dtype-agnostic CUDA 8-bit AdamW + bf16-native kernel | GPU CUDA + GPU Triton hash | Per-row CPU ring buffer (4200 steps) | **Superseded** |
| v9 (Dual-Signal, May 15) | Delta + gradient LSH mixing + exponential weights | GPU Triton hash × 2 + GPU CUDA | Dual CPU ring buffers (4200 steps) | **Current** |

## Key Findings

### 1. zstd on bf16 bytes cannot distinguish known from novel (May 14)
- All non-zero bf16 deltas compress to ~1.27x regardless of training state
- Raw compression, prefix dict, and trained dict all fail (ratio stuck at entropy floor)
- The byte-level representation of bf16 delta has no compressible patterns due to gradient noise + Adam smoothing + quantization

### 2. Cosine similarity on delta DIRECTIONS works (May 15)
- LSH (sign of random projections) preserves cosine similarity in compact bit hashes
- Same-prompt deltas: cosine similarity ~0.88 (880x discrimination vs random)
- Multi-scale comparison (offsets 1, 5, 10, 25, 50) captures convergence at multiple time scales
- Attenuation = mean_sim * (1 - flatness) — pure function, no thresholds

### 3. Convergence gate was too strict (May 15)
- With threshold 0.9, gate skipped 88-95% of backward passes after step ~1000-1500
- Accuracy capped at 90.3% (SST-2) vs 92-93% baseline
- Fix: raise threshold to 0.99, compute hash every step (even when gate fires)
- Initial fix results: 91.87% at step 1500, 0% gate skip rate, still climbing

### 4. GPU hash speed is unknown (need profiling)
- Early estimate of "~1ms for 24 layers" was wrong
- Back-of-napkin: 24 Triton kernel launches with 737K total blocks should take 10-80ms
- Actual measured step time is ~1.29s vs estimated full-finetune baseline ~0.9-1.0s
- ZPackR overhead is ~300-400ms — need per-component timers to find the real bottleneck

### 5. Continuous byte comparison fixes dead signal
- v5 with K=32 + binary byte match + squared formula produced attenation stuck at 0.0 for 3500 steps
- K=32 packed into 4 bytes + binary match → 5 similarity levels/offset
- K=16 packed into 2 bytes + continuous byte comparison `1 - |diff|/255` → ~512 levels/offset
- Fix: switch to continuous byte comparison — signal alive from step 1

### 6. GPU window is the largest VRAM consumer
- GPU window (4200 × 46080 × 2 uint8) = 369MB — biggest single chunk
- CPU-pinned window saves 369MB GPU VRAM at cost of ~644KB/step transfer (~40μs)
- Async `copy_(non_blocking=True)` in push — data arrives before next compute_attenuation

### 7. Model loaded in fp32 wastes ~100MB
- HuggingFace default loads in fp32; attention/embedding/pooler layers stay fp32
- With `--bf16` flag, entire model converts to bfloat16 — saves ~100MB, no quality impact
- Config option `PackRConfig(bf16=True)` for compatibility with other tooling

### 8. ZPackRLinear is now dtype-agnostic
- Forward no longer hardcodes `.float()` cast — output dtype matches input dtype
- bf16 input → bf16 output, fp32 input → fp32 output
- Enables clean bf16 mode without dtype cascading issues
- LayerNorm reverted to fp32 + forward patch handles bf16 input internally

### 9. Per-block variation is real
- Different layers converge at different rates (deeper layers faster)
- Within a layer, all rows vary independently

### 10. Dual-signal mixing: delta + gradient hash (May 15)
- Signal analysis revealed root cause of death spiral: delta-hash alone
  can't distinguish "converged" from "stuck" — both have stable position.
- Gradient hash measures learning signal SNR (stable gradient = learning,
  noisy gradient = converged or stuck).
- Together they form a complete signal via geometric product:
  ```
  atten = delta_sim^(1-mix) * (1-grad_sim)^mix
  ```
  Attenuation is only high when BOTH agree the row is done.
- Add `_grad_sig_db` (second `DeltaSignatureDB`), `compute_grad_hash()`
  (called after backward, before optimizer), and `--gradient-mix` CLI flag.
- Default mix=0.5 gives ~40-60% effective gradient during learning vs
  ~3-8% with delta-only, while still braking to ~24% at convergence.
- Attenuation ranges from ~0.0 (still learning) to 1.0 (fully converged) across rows
- Layer 8-10 output converge fastest, layer 2-3 intermediate slowest

## Test Data (SST-2, no velvet)

| Run | Steps | Peak Acc | Notes |
|-----|-------|---------|-------|
| `no_gate_calibration` | 2000 | 92.5% | zstd+WeightDict, v1 |
| `epoch2_full` | 8000 | 94.06% | v1, gate on |
| `zstd_attenuation_sst2` | 8000 | 90.3% | v3 zstd, entropy floor capped |
| `lsh_sst2` (v4, 0.9 gate) | 8000 | 90.31% | v4 LSH, gate killed training early |
| `lsh_sst2` (v4, 0.99 gate) | 8000 | 91.87% | v4 LSH, gate threshold 0.99 |
| `lsh_sst2` (v5, K=32, binary byte match, window=1100) | 3740 | 92.81% | Signal dead until step ~200, then slow climb |
| `lsh_sst2` (v6, K=16, continuous byte, window=4200) | 2496 | 92.19% | Real signal from step 1, still climbing |

## KNOWN BUGS / ISSUES

- Triton kernel compilation on first call adds ~2s latency — acceptable
- CPU fallback for hash_rows is slower (cuBLAS matmul instead of Triton kernel)
- No pruning (all rows stay active) — may use more VRAM than necessary

## NOT TO RETRY

- LZ4 for the ratio signal (ratios < 1.0, useless)
- WeightDict + reindex (complexity without signal improvement)
- Any compression-based signal on bf16 bytes (entropy floor kills it)
- Relative creep tracking (EMA, prev_ratio)
- Super Dict as gate (word-length dependent)
- Variance gating (stale ratios)
- Per-prompt lookup table (2.5GB storage, unnecessary)
- Block-level grouping (caps accuracy at 92%)

## VRAM Breakdown (BERT-base, batch=16, seq=128)

| Component | Before (v5) | After (v6) | Delta |
|-----------|-------------|------------|-------|
| Model weights | 417MB (FFN 2×bf16 + attention fp32) | 316MB (all bf16 with `--bf16`) | -101MB |
| LSH window | 369MB (GPU uint8 ring buffer) | 0MB (CPU pinned) | -369MB |
| Optimizer (int8) | 213MB | 213MB | 0MB |
| **Persistent total** | **~999MB** | **~529MB** | **-470MB** |
| Gradients (peak) | 220MB | 220MB | 0MB |
| Activations + overhead | ~500-800MB | ~500-800MB | 0MB |
| **nvidia-smi** | **~2.2GB** | **~1.5GB** | **-700MB** |

## Removed Complexity (May 15-16)

| Component | Reason |
|-----------|--------|
| Block-level grouping | Replaced by per-row — each row converges independently |
| `num_blocks`, `block_mask`, `_kept_indices`, `_scatter_indices` | No more block-level machinery |
| `F.pad` + reshape in hash | delta is already [in_features, out_features] |
| `repeat_interleave` in forward | Attenuation is already [in_features] — direct unsqueeze |
| `torch.tensor(list, device=cuda)` | Attenuation stored as uint8 GPU tensor via register_buffer |
| cuBLAS matmul for hash | Replaced by custom Triton kernel ([in_features, K] 2D grid) |
| float32 intermediate in hash | Triton kernel computes dot products directly from bf16 |
| GPU ring buffer | Moved to CPU pinned memory — saves 369MB GPU VRAM |
| `_sim_sum`, `_sim_sqs` pre-allocation | Replaced by single batched compute_attenuation |
| `self._atten_byte = ...` reassignment | Changed to in-place `.copy_()` — no buffer churn |
| 2D hash kernel `(in_features, K)` grid | Replaced by 1D fused kernel `(in_features,)` — 16× fewer launches, 16× less delta traffic |
| `.float()` cast in ZPackRLinear forward | Output dtype now matches input dtype — dtype-agnostic |
| Full-step hash (every step, 415ms) | Configurable `hash_interval=N` — amortized cost drops N× |

## Determinism Status (May 16)

Every component is a pure function of current state:
- LSH hash: Triton kernel `sign(delta_row · proj_row)` — fixed seed, deterministic
- Multi-scale comparison: `cos_sim = 2 * (1 - |diff|/255).mean() - 1` — continuous byte
- Attenuation: `(mean_sim * (1 - flatness))²` → uint8 `(attn * 255).to(torch.uint8)`
- Gate: `all(attenuation >= threshold)` — pure function
- Checkpoint: zstd compress/decompress — lossless roundtrip
- DataLoader: `torch.manual_seed(seed)` — fixed shuffle order
- Async CPU push: `non_blocking=True` copy has full step (~1.3s) to complete before read

## Per-Step Timing (metrics.jsonl, v8 CUDA optimizer)

Per-component timers in every step record (fields prefixed with `t_`, ms):

| Field | What it measures | v8 (no-hash step) | v8 (hash step) |
|-------|-----------------|-------------------|----------------|
| `t_forward` | Python forward launch overhead | 25ms | 25ms |
| `t_gate` | CUDA sync — captures **forward** GPU time | 290ms | 290ms |
| `t_backward` | backward GPU execution | 250ms | 250ms |
| `t_optimizer` | optimizer.step + zero_grad | **129ms** | **129ms** |
| `t_hash` | fused hash kernel + attenuation × 24 | 0ms | 619ms |
| `t_ratio_log` | _log_ratios overhead | 3ms | 3ms |
| `step_ms` | total step wall time | **~1236ms** | **~1326ms** |

To view:
```bash
python3 -c "import json;f=open('PATH/metrics.jsonl');[print(f\"{d['step']:5d} fwd={d.get('t_forward',0):5.1f} bwd={d.get('t_backward',0):5.1f} opt={d.get('t_optimizer',0):5.1f} hash={d.get('t_hash',0):5.1f} tot={d['step_ms']:.0f}ms\") for l in f if (d:=json.loads(l)).get('type')=='step']"
```

## Speed Optimization

**Current**: ~1.29s/step (measured with profiling timers).
**Target**: Match full finetune speed (~0.9-1.0s/step).

**Initial profiling data** (v6 before fused kernel):

| Component | GPU time | % of step |
|-----------|----------|-----------|
| Forward (incl. weight assembly) | 289ms | 22% |
| Backward | 311ms | 24% |
| Optimizer step | 235ms | 18% |
| **Hash + Attenuation** | **415ms** | **32%** |
| Overhead | 38ms | 3% |

### Optimization 1: Fused 1D hash kernel ✓ (implemented)

Old kernel: 2D grid `(in_features, K)` — 16 blocks per row, each reloading delta[row] independently.
New kernel: 1D grid `(in_features,)` — one block per row processes ALL K projections. Delta loaded ONCE.

- Delta memory traffic reduced **16×** (was K copies per row, now 1)
- Kernel launches reduced **16×** (was in_features × K, now in_features)
- `tl.store(hash_ptr + ... + tl.arange(0, K), result)` — K bits stored in one call
- `non_blocking=True` in compute_attenuation's `.cuda()` — avoids Python blocking on CPU→GPU transfers

### Optimization 2: Hash interval ✓ (implemented)

`--hash-interval N`: compute LSH every N steps, reuse attenuation in between.
The convergence signal is averaged across offsets spanning 1-1000 steps —
lagging by N steps is negligible.

| N | Amortized hash | Est. step time | vs target |
|---|----------------|----------------|-----------|
| 1 (default) | 396ms | ~1288ms | -288ms |
| 4 | 99ms | ~991ms | ≈ matches |
| 8 | 50ms | ~942ms | ✅ faster |

Counter-based skip in `compute_hash_gpu()` — returns early when
`_hash_counter < _hash_interval`. Window pushes only on hash steps
(coarser offsets, same log-spacing).

### Optimization 3: Parallel hash streams (if needed)

Launch hash kernels for all 24 layers concurrently on separate CUDA streams. Wall time drops from sequential sum to max-per-layer. Most useful if hash remains the bottleneck after interval.

## Optimizer Performance

| Optimizer | Memory | Step time (110M params) | Dtype support | Notes |
|-----------|--------|------------------------|---------------|-------|
| Standard AdamW (fp32 states) | 880MB | 80ms | fp32/bf16 | cuDNN-optimized |
| Triton 8-bit (per-param) | 220MB | 300ms | fp32/bf16 | 74 separate launches |
| **CUDA 8-bit (per-param)** | **220MB** | **~38ms** | **fp32 + bf16** ✅ | **Hand-tuned CUDA** |
| Full train w/ CUDA 8-bit + LSH | 565MB alloc, 861MB peak | **~1240ms** | fp32 + bf16 | ✅ matches finetune |

The CUDA 8-bit AdamW kernel (`--optimizer cuda8`) is compiled inline by nvcc on first import (cached at `~/.cache/torch_extensions/`). It uses:
- **Dtype-agnostic**: kernel accepts `void*` pointers, branches on `is_bf16` flag.
  - bf16: inline bit-shift (`(uint)p_u << 16`) for register-level conversion
  - fp32: direct float load/store
- Warp-level `__shfl_xor_sync` reductions: no shared memory stalls for absmax
- Block-level int8 quantization with per-block scales
- Per-param launch (76 launches × ~0.5ms = ~38ms total)

## Optimizer is Dtype-Agnostic

The CUDA 8-bit AdamW kernel (`packr/packr/cuda_adam.py`) handles both fp32 and bf16
params/grads without any conversion.  The `--bf16` flag is purely for VRAM savings on
model weights — the optimizer works identically either way.

```python
# fp32 model (no --bf16): kernel loads float values directly
p = torch.randn(N, dtype=torch.float32, device='cuda')  # works

# bf16 model (--bf16): kernel interprets raw bytes as bf16 via shift
p = torch.randn(N, dtype=torch.bfloat16, device='cuda')  # works
```

## New Config Options

| Flag | Config | Default | Effect |
|------|--------|---------|--------|
| `--bf16` | `PackRConfig(bf16=True)` | False | Convert model to bfloat16 before training (saves ~60MB VRAM) |
| `--hash-interval` | `PackRConfig(hash_interval=N)` | 1 | Compute LSH hash every N steps (amortized cost drops N×) |
| `--optimizer` | `PackRConfig(optimizer_type=...)` | `cuda8` | Optimizer: `cuda8` (fast CUDA, any dtype), `triton8` (Triton 8-bit), `adamw` (standard fp32) |
