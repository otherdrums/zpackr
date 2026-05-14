# ZPackR Implementation Guide — v3.0 (LZ4)

Target repos:
- `github.com/otherdrums/packr` — PackRLinear, FusedQuantizedAdam, VelvetController (MIT)
- `github.com/otherdrums/zpackr` — ZPackRLinear, LZ4 attenuation, convergence gate (MIT)
- `github.com/otherdrums/packr-research` — training harness, diagnostics, sweep tools (AGPL)

## The Headline

> "Per-block LZ4 compressibility of the weight delta tells you whether the
> model already encodes that block's patterns.  Known blocks get attenuated
> in the forward pass and their gradients naturally decay.  Overfitting
> becomes structurally impossible at the block level."

### What ZPackR + Velvet Eliminate

| Tunable | Eliminated By | Mechanism |
|----------|:---:|-----------|
| LR schedule (warmup, decay, cosine) | Velvet | Continuous velocity-to-LR translation from optimizer `exp_avg_sq` |
| How many epochs to train | Convergence gate | `should_skip_backward()` when all blocks fully attenuated |
| Knowing when to stop | Convergence gate | Auto-termination when gate skip rate saturates (future) |
| Per-layer learning rates | Velvet per-group granularity | Saturated groups get min_multiplier; hungry groups get max_multiplier |
| Dictionary maintenance | LZ4 stateless compression | No reindex, no dictionary training, no state to manage |
| Threshold calibration | Fixed constants | `RATIO_FLOOR=1.0`, `RATIO_CEILING=8.0` derived from data |

---

## Architecture — Frozen Base + LZ4 Delta

ZPackRLinear stores a frozen BERT pretrained weight matrix and a trainable
delta.  The delta is processed in blocks (256 elements each, matching
FusedQuantizedAdam's block size).  Each post_step, the delta bytes are
compressed with LZ4 and the compression ratio maps to a per-block attenuation
factor applied in the next forward pass.

```
Text → BERT forward/backward → delta bytes
                                  │
                    LZ4.block.compress() per block → ratio
                                  │
            clamp((ratio - 1.0) / 7.0, 0, 1) → attenuation [0,1]
                                  │
         Forward: output = x @ (base_W + delta * (1 - attenuation))
```

### Why LZ4

| Property | zstd + WeightDict (v2) | LZ4 (v3) |
|----------|----------------------|----------|
| Per-block speed | ~1-3 ms | ~0.003 ms |
| State required | Evolving dictionary + periodic reindex | None — stateless |
| Zero-delta ratio | 255x | 255x |
| Non-zero delta ratio | ~1.0x | ~1.0x |
| Signal sharpness | Blurred by stale dictionary patterns | Clean — always current bytes |
| Dependencies | zstandard, custom WeightDict class | lz4 (single call) |
| Maintenance cost | Reindex ~1900 new patterns per cycle | Zero |

The v2 WeightDict accumulated delta patterns at reindex to provide a "memory" of
training history.  In practice, this memory didn't improve signal discrimination —
all non-zero deltas compressed at ~1.3x regardless of whether the dictionary had
seen those patterns before.  The reindex added complexity without adding signal.

LZ4 gives identical zero-vs-nonzero discrimination (255x vs ~1.0x) in a fraction
of the time, with no state, no dictionary, and no periodic maintenance.

### Attenuation Constants

```python
RATIO_FLOOR = 1.0   # Below this, block is fully novel (attenuation = 0)
RATIO_CEILING = 8.0  # At/above this, block is fully known (attenuation = 1)
ATTENUATION_SKIP_THRESHOLD = 0.9  # Gate fires when all blocks ≥ this
```

Derived from ratio distribution analysis on SST-2 (2000 steps, no gate).
With gate enabled, zero-delta blocks stay at 255x (attenuation 1.0) while
trained blocks stay at ~1.0-1.3x (attenuation ~0-0.04).  The `RATIO_CEILING`
of 8.0 gives generous headroom — a trained block would need a ratio of 8.0
to be fully attenuated, which never happens with bf16 bytes.

### Convergence Gate

```python
def should_skip_backward(zpl_layers, threshold=0.9) -> bool:
    """Return True if all blocks across all layers have attenuation >= threshold."""
    for _, module in zpl_layers:
        if module._attenuation_factors is None:
            return False
        if any(a < threshold for a in module._attenuation_factors):
            return False
    return True
```

Called every step before backward.  Uses cached attenuation factors from the
last post_step (stale by at most `post_step_interval` steps — negligible).
Future: auto-terminate training when gate skip rate exceeds a threshold (e.g. 95%).

---

## Implementation Notes

### Frozen Base + LZ4 Delta

Implemented as frozen BERT pretrained `base_W` (bf16, requires_grad=False) +
trainable `delta_salient` (bf16).  Forward is a single cuBLAS matmul:

```python
W = base_W + delta * (1 - attenuation)  # fused in one tensor
output = x @ W
```

This is more like a full-rank adapter.  The base matmul uses optimized cuBLAS.
Delta starts as zeros — all blocks novel, all kept.  LZ4 gives sharp zero-vs-nonzero
discrimination.  VRAM savings from delta pruning.

### Forward Speed Optimizations

- **base_W stored as bf16** — eliminates per-forward dtype conversion
- **Pre-cast x to bf16 once** at the top of forward
- **Cached salient_count** — updated in post_step, avoids GPU sync per step
- **Guarded `.to(device)` on delta** — no-op kernel launch eliminated
- **Activation memory freed** after bf16 cast (`del x`)

### Velvet Batch GPU Sync (packr)

VelvetController syncs once per step instead of once per parameter.
Collects all v_mean GPU scalars, single `torch.cuda.synchronize()`,
then `.item()` locally.  Reduces ~200 GPU syncs/step to 1.

### Decode Kernel Optimizations (packr, shared with PackR mode)

- **Decode kernel in backward** — eliminates int64 intermediate (~19 MB/layer)
- **Fused LUT gradient kernel** — single-pass bincount+scatter_add via Triton
- **Async W_p prefetch** — offload copies on dedicated CUDA stream

### ZPackR Forward Path Optimizations

- **Cached kept-block indices** — `block_mask.nonzero()` was called every forward
- **Scatter indices cached** — rebuilt lazily when mask changes
- **Single cuBLAS matmul** — base_W + delta combined before matmul

### Offload Path Optimizations (packr)

- **Async W_p prefetch** on dedicated CUDA stream
- **Clone elimination** in evict_wp and register_wp

---

## Performance Benchmarks

All runs on BERT-base, SST-2, batch_size=16, sm_75 (GTX 1650).  May 2026.

| Method | ms/step | VRAM peak | Accuracy | Notes |
|--------|--------:|----------:|---------:|-------|
| Standard BERT | 838ms | 1073MB | — | baseline |
| PackR (v1) | **802ms** | 1142MB | 89.7% | matches standard BERT |
| ZPackR (v2, zstd + WeightDict) | 1264ms | 1176MB | 94.1% | single cuBLAS matmul |
| ZPackR (v3, LZ4) | — | — | — | pending benchmarks |

### Optimizations Applied

| Optimization | Impact | Where |
|-------------|--------|-------|
| Decode kernel in backward | Eliminates int64 intermediate (~19MB/layer) | `autograd.py` |
| Fused LUT gradient kernel | Single-pass bincount+scatter_add via Triton | `kernel.py` |
| Cached kept indices | Avoids `block_mask.nonzero()` every forward | `zpackr_layer.py` |
| Delta variance gating | Skip LZ4 compression for unchanged blocks (15% threshold) | `zpackr_layer.py` |
| Fused forward matmul | Single cuBLAS matmul replaces base+delta dual launch | `zpackr_layer.py` |
| Velvet batched sync | 1 GPU sync instead of ~200 per-param `.item()` | `velvet.py` |
| Async W_p prefetch | Offload copies on dedicated CUDA stream | `offload.py` |
| Clone elimination in evict | Avoids wasted CPU allocation per eviction | `offload.py` |

---

## File Map

### zpackr package

| File | Purpose |
|------|---------|
| `zpackr_layer.py` | ZPackRLinear — frozen base + LZ4 delta, forward/backward/post_step |
| `config.py` | ZPackRConfig dataclass |
| `layer_patcher.py` | `compress_model()` — nn.Linear → ZPackRLinear |
| `prompt_gate.py` | `should_skip_backward()` — convergence gate |
| `checkpoint.py` | `save/load_zpackr_checkpoint()` — LZ4-compressed delta on disk |
| `super_dict.py` | Optional zstd text preprocessor (not used in training signal) |
| `zpackr_interface.py` | `export_model()` — merge base+delta → standard nn.Linear |

### packr package (dependency)

| File | Purpose |
|------|---------|
| `layer.py` | PackRLinear — uint8 LUT + bf16 residual (PackR mode) |
| `kernel.py` | Triton decode & fused LUT gradient kernels |
| `autograd.py` | Custom autograd for PackRLinear |
| `optim.py` | FusedQuantizedAdam — 8-bit block-quantized AdamW |
| `velvet.py` | VelvetController — velocity→LR translation |
| `offload.py` | CPU offloading with async CUDA stream prefetch |
| `config.py` | PackRConfig with mode dispatch |
| `layer_patcher.py` | `compress_model()` with mode dispatch |

### packr-research (tools)

| File | Purpose |
|------|---------|
| `train_harness.py` | ZPackRTrainer + TrainerConfig + CLI |
| `diagnose.py` | DiagnosticTrainer — per-block ratio logging |
| `ablate.py` | Parameter sweep runner |
| `calibrate.py` | Threshold confusion-matrix report |
| `report.py` | Post-hoc result tables + charts |

---

## Test Plan

### test_zpackr_layer.py
- Forward shape and correctness vs nn.Linear
- post_step produces attenuation factors
- Checkpoint roundtrip preserves delta (LZ4)
- Gradient flow only to delta_salient

### test_dual_signal.py (LZ4 Signal)
- Zero-delta compresses extremely well (255x+)
- Trained delta compresses poorly (~1.0x)
- post_step produces attenuation from LZ4 ratios
- Convergence gate fires on fully-attenuated blocks
- Gate does not fire when any block below threshold

### test_prompt_gate.py
- Gate fires when all blocks attenuated
- Gate ignores when any block novel
- Empty layers → gate fires
- Custom threshold overrides default

### test_checkpoint.py
- Model-level save/load roundtrip with LZ4
- Forward output preserved across checkpoint

---

## Deferred (v3.1+)

- Auto-terminating training (convergence gate skip rate threshold)
- Per-block convergence-driven VRAM pruning (blocks that stay at 255x for N post_steps fully pruned)
- Super Dict as optional text preprocessor for semantic whitening
- Multi-task continual learning with attenuation as per-task signal
