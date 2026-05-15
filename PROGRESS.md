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
| v5 (Row LSH, May 15) | Per-row LSH + Triton kernel + uint8 + log-spaced offsets | GPU Triton kernel (K=32) | Per-row ring buffer (1100 steps) | **Current** |

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

### 4. GPU hash is fast (~1ms for 24 layers)
- Two shared projection matrices on GPU (120MB VRAM)
- No GPU→CPU copy of delta (108MB/step eliminated)
- No CPU matmul (7.2 GFLOPS/step eliminated)
- No background thread, no threading complexity

### 5. Per-block variation is real
- Different layers converge at different rates (deeper layers faster)
- Within a layer, all blocks converge similarly (same gradient direction)
- Attenuation varies from 0.68 (still learning) to 1.0 (fully converged) across blocks at step 250

## Test Data (SST-2, no velvet)

| Run | Steps | Peak Acc | Notes |
|-----|-------|---------|-------|
| `no_gate_calibration` | 2000 | 92.5% | zstd+WeightDict, v1 |
| `epoch2_full` | 8000 | 94.06% | v1, gate on |
| `zstd_attenuation_sst2` | 8000 | 90.3% | v3 zstd, entropy floor capped |
| `lsh_sst2` (v4, 0.9 gate) | 8000 | 90.31% | v4 LSH, gate killed training early |
| `lsh_sst2` (v4, 0.99 gate) | 8000 | 91.87% | v4 LSH, gate threshold 0.99 |
| `lsh_sst2` (v5, row-level) | running | TBD | v5 per-row LSH, Triton kernel, uint8 |

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

## Removed Complexity (May 15)

| Component | Reason |
|-----------|--------|
| Block-level grouping | Replaced by per-row — each row converges independently |
| `num_blocks`, `block_mask`, `_kept_indices`, `_scatter_indices` | No more block-level machinery |
| `F.pad` + reshape in hash | delta is already [in_features, out_features] |
| `repeat_interleave` in forward | Attenuation is already [in_features] — direct unsqueeze |
| `torch.tensor(list, device=cuda)` | Attenuation stored as uint8 GPU tensor via register_buffer |
| cuBLAS matmul for hash | Replaced by custom Triton kernel ([in_features, K] 2D grid) |
| float32 intermediate in hash | Triton kernel computes dot products directly from bf16 |

## Determinism Status (May 15)

Every component is a pure function of current state:
- LSH hash: Triton kernel `sign(delta_row · proj_row)` — fixed seed, deterministic
- Multi-scale comparison: `cos_sim = 2 * (hash == stored).mean() - 1`
- Attenuation: `mean_sim * (1 - flatness)` → uint8 `(attn * 255).to(torch.uint8)`
- Gate: `all(attenuation >= threshold)` — pure function
- Checkpoint: zstd compress/decompress — lossless roundtrip
- DataLoader: `torch.manual_seed(seed)` — fixed shuffle order
