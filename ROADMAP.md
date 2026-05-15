# ZPackR Implementation Guide — v4 (LSH)

Target repos:
- `github.com/otherdrums/packr` — PackRLinear, FusedQuantizedAdam, VelvetController (MIT)
- `github.com/otherdrums/zpackr` — ZPackRLinear, LSH attenuation, convergence gate (MIT)
- `github.com/otherdrums/packr-research` — training harness, diagnostics, sweep tools (AGPL)

## The Headline

> "Per-block LSH cosine similarity of the delta across a sliding window tells you
> whether the model has converged. Known blocks get attenuated in the forward
> pass and their gradients naturally decay. Local minima are escaped via
> continuous, reversible attenuation."

### What ZPackR + Velvet Eliminate

| Tunable | Eliminated By | Mechanism |
|----------|:---:|-----------|
| LR schedule (warmup, decay, cosine) | Velvet | Continuous velocity-to-LR translation from optimizer `exp_avg_sq` |
| How many epochs to train | Convergence gate | `should_skip_backward()` when all blocks fully attenuated |
| Knowing when to stop | Convergence gate | Auto-termination when gate skip rate saturates (future) |
| Per-layer learning rates | Velvet per-group granularity | Saturated groups get min_multiplier; hungry groups get max_multiplier |
| Compression parameter tuning | LSH sliding window | No dictionaries, no levels, no calibration — fixed random projections |

---

## Architecture — Frozen Base + LSH Attenuation

ZPackRLinear stores a frozen BERT pretrained weight matrix and a trainable
delta. The delta is processed in blocks (256 elements each). Each step, all
block deltas are LSH-hashed on GPU and compared against a sliding window
of past hashes using multi-scale offsets (1, 5, 10, 25, 50 steps ago).

```
Text → BERT forward/backward → delta on GPU
                                  │
                    LSH hash (sign of random projections)
                                  │
         multi-scale cos_sim vs window: mean_sim * (1 - flatness)
                                  │
        Forward: output = x @ (base_W + delta * (1 - attenuation))
```

### Why LSH over Compression

| Property | zstd (v3) | LSH (v4) |
|----------|-----------|----------|
| Signal location | Byte-level (noisy bf16) | Value-level (vector directions) |
| Discrimination | ~1x (all deltas look random) | 880x (converged vs novel) |
| Computation | CPU matmul (~400ms/step) | GPU matmul (~1ms/step) |
| State | None | 60-step ring buffer of 64-bit hashes |
| Dependencies | zstandard | torch only |
| Memory | 108MB GPU→CPU copy/step | 0 copy (stays on GPU) |

### The zstd problem (why it couldn't work)

Raw bf16 bytes from Adam-optimized deltas compress at ~1.27x regardless of
content. This is the bf16 entropy floor — random bf16 data always compresses
to this ratio. No compression technique (raw, prefix dict, trained dict) can
distinguish "known" from "novel" patterns because gradient noise + Adam
smoothing + bf16 quantization produces bytes that look random to any
compressor.

### The LSH solution

LSH (sign of random projections on the delta VALUES, not bytes) preserves
cosine similarity in compact bit hashes. Two deltas with similar direction
have similar hashes. This works because the delta's direction stabilizes as
the block converges, even though the bytes remain noisy.

### Attenuation Formula

```python
cos_sim = 2 * matching_bits / K - 1       # per offset in (1, 5, 10, 25, 50)
mean_sim = mean(cos_sim across offsets)    # convergence level
flatness = std(cos_sim across offsets)     # stability across time scales
attenuation = mean_sim * (1 - flatness)    # pure function, no thresholds
```

- Converged + stable → ~1.0 (fully attenuated)
- Learning + varying → ~0.8 (partially attenuated)
- Novel (no history) → 0.0 (fully active)

### Convergence Gate

```python
def should_skip_backward(zpl_layers, threshold=0.99):
    """Return True if all blocks across all layers have attenuation >= threshold."""
    for _, module in zpl_layers:
        if module._attenuation_factors is None:
            return False
        if any(a < threshold for a in module._attenuation_factors):
            return False
    return True
```

Hash is computed every step (even when gate fires) so the window keeps
evolving. If any block dips below threshold, the gate opens and training
resumes.

---

## Implementation Notes

### Frozen Base + LSH Delta

Implemented as frozen BERT pretrained `base_W` (bf16, requires_grad=False) +
trainable `delta_salient` (bf16). Forward is a single cuBLAS matmul:

```python
W = base_W + delta * (1 - attenuation)  # fused in one tensor
output = x @ W
```

### GPU Hash Computation

After `optimizer.step()`, `compute_hash_gpu()` is called synchronously:

```python
@torch.no_grad()
def compute_hash_gpu(self):
    padded = F.pad(delta_salient, (0, 0, 0, pad_rows))
    blocks = padded.reshape(n_blocks, block_size * out_features)
    proj = DeltaSignatureDB.get_gpu_projections(block_elements)
    hash = (blocks.float() @ proj.t() > 0).to(torch.uint8)
    self._sig_db.push(hash)
    attenuation = self._sig_db.compute_attenuation(hash)
    self._attenuation_factors = attenuation.tolist()
```

Two shared projection matrices on GPU (120MB total):
- `[K, 256*3072]` for intermediate layers (96MB)
- `[K, 256*768]` for output layers (24MB)

### Forward Speed Optimizations

- **base_W stored as bf16** — eliminates per-forward dtype conversion
- **Pre-cast x to bf16 once** at the top of forward
- **Cached salient_count** — no GPU sync per step
- **No GPU→CPU delta copy** — hash computed directly from GPU delta

---

## Performance Benchmarks

All runs on BERT-base, SST-2, batch_size=16, sm_75 (GTX 1650). May 2026.

| Method | ms/step | VRAM peak | Accuracy | Notes |
|--------|--------:|----------:|---------:|-------|
| Standard BERT | 838ms | 1073MB | — | baseline |
| PackR (v1) | **802ms** | 1142MB | 89.7% | matches standard BERT |
| ZPackR (v2, zstd + WeightDict) | 1264ms | 1176MB | 94.1% | single cuBLAS matmul |
| ZPackR (v4, LSH K=64) | ~1400ms | 1439MB | 91.9%+ (running) | GPU hash, no thread |

Note: the extra ~560ms vs standard BERT is from the forward pass weight
construction (`base_W + delta * (1 - nv)`) per layer, not from the hash
computation (which is ~1ms on GPU).

---

## File Map

### zpackr package

| File | Purpose |
|------|---------|
| `zpackr_layer.py` | ZPackRLinear, DeltaSignatureDB, `compute_hash_gpu()` |
| `config.py` | ZPackRConfig dataclass |
| `layer_patcher.py` | `compress_model()` — nn.Linear → ZPackRLinear |
| `prompt_gate.py` | `should_skip_backward()` — convergence gate |
| `checkpoint.py` | `save/load_zpackr_checkpoint()` — zstd-compressed delta on disk |
| `super_dict.py` | Optional zstd text preprocessor (archived) |
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
- post_step produces attenuation factors (via compute_hash_gpu)
- Checkpoint roundtrip preserves delta (zstd)
- Gradient flow only to delta_salient

### test_dual_signal.py (LSH Signal)
- LSH hash is deterministic (same seed → same hash)
- LSH hash has correct shape [n_blocks, K]
- Empty window gives zero attenuation
- Attenuation increases as delta stabilizes across encounters
- Convergence gate fires on fully-attenuated blocks
- Gate does not fire when any block below threshold
- Checkpoint roundtrip preserves delta state
- ZPackRLinear has DeltaSignatureDB

### test_prompt_gate.py
- Gate fires when all blocks attenuated
- Gate ignores when any block novel
- Empty layers → gate fires
- Custom threshold overrides default

### test_checkpoint.py
- Model-level save/load roundtrip with zstd
- Forward output preserved across checkpoint

---

## Deferred (v4.1+)

- Auto-terminating training (convergence gate skip rate threshold)
- Per-block VRAM pruning (blocks that stay at 1.0 for N steps)
- Multi-scale flatness as local minimum detector (add perturbation)
- Higher K values (128, 256) for finer resolution
- Cosine similarity directly (not LSH hash) for finer-grained attenuation
