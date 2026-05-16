# ZPackR Implementation Guide — v8 (CUDA Optimizer + Fused Hash)

Target repos:
- `github.com/otherdrums/packr` — CUDA8BitAdam, FusedQuantizedAdam, VelvetController (MIT)
- `github.com/otherdrums/zpackr` — ZPackRLinear, LSH attenuation, convergence gate (MIT)
- `github.com/otherdrums/packr-research` — training harness, diagnostics (AGPL)

## The Headline

> "Per-row LSH cosine similarity of the delta across a sliding window tells you
> whether the model has converged. Converged rows get attenuated in the forward
> pass and their gradients naturally decay."

### What ZPackR Eliminates

| Tunable | Eliminated By | Mechanism |
|----------|:---:|-----------|
| How many epochs to train | Convergence gate | `should_skip_backward()` when all rows byte=255 |
| Compression parameter tuning | LSH sliding window | Fixed random projections, no calibration |
| Per-layer convergence differences | Per-row attenuation | Each of ~46K rows converges independently |

---

## Architecture — Frozen Base + Per-Row LSH Attenuation

ZPackRLinear stores a frozen pretrained weight (`base_W`, bf16) and a trainable
`delta_salient` (bf16). Each step, all delta rows are LSH-hashed via a fused
Triton kernel (1D grid, K=16) and compared against a 4200-step CPU-pinned
window using log-spaced multi-scale offsets.

```
Text → BERT forward/backward → delta on GPU
                                  │
          Fused Triton kernel (1D grid, K=16)
          Each block: load delta[row] ONCE, compute 16 dot products
                                  │
         continuous byte comparison vs 4200-step CPU window
         offsets: (1, 3, 10, 30, 100, 300, 1000)
                                  │
        Forward: output = x @ (base_W + delta * (1 - attenuation))
```

### Why LSH over Compression

| Property | zstd (v3) | LSH (v8) |
|----------|-----------|----------|
| Signal location | Byte-level (noisy bf16) | Direction-level (cosine sim via random projections) |
| Discrimination | ~1× entropy floor | 880× (converged vs novel) |
| Computation | CPU matmul (~400ms/step) | GPU Triton kernel (~265ms/step) |
| State | None | 4200-step CPU ring buffer, 2 bytes/row |
| GPU→CPU traffic | 108MB delta copy/step | 92KB hash push/step |

### The zstd problem (why it couldn't work)

Raw bf16 bytes from Adam-optimized deltas compress at ~1.27x regardless of
content (the bf16 entropy floor). No technique — raw, prefix dict, or trained
dict — can distinguish "known" from "novel" bf16 deltas.

### Attenuation Formula

```python
diff = (current_hash.float() - past_hash.float()).abs()
byte_sim = 1.0 - diff / 255.0              # continuous byte comparison
matching = byte_sim.mean(dim=1)            # per-row, across packed bytes
cos_sim = 2 * matching - 1                 # map [0,1] → [-1,1]

mean_sim = cos_sim.mean(dim=0)             # [rows] — convergence level
variance = cos_sim.var(dim=0)              # [rows] — stability across time
flatness = sqrt(clamp(variance, 0.0))      # std deviation
attenuation = (mean_sim * (1 - flatness))² # squared → delays byte=255
```

- Converged + stable → ~1.0 (byte=255, fully attenuated)
- Learning + varying → ~0.5-0.8 (partially attenuated)
- Novel (no history) → 0.0 (fully active)

### Convergence Gate

```python
def should_skip_backward(zpl_layers, threshold=1.0):
    """Return True if ALL rows across ALL layers have attenuation >= threshold."""
    for _, module in zpl_layers:
        if module._atten_byte.float().min().item() / 255.0 < threshold:
            return False
    return True
```

Gate uses `min()` semantics — every single row must be fully converged.

---

## Implementation Notes

### Frozen Base + LSH Delta

```python
W = base_W + delta_salient * (1.0 - _atten_byte.float() / 255.0)
output = (x_bf16 @ W).to(x.dtype)   # dtype-agnostic
```

### GPU Hash Computation (fused Triton kernel)

```python
@torch.no_grad()
def compute_hash_gpu(self):
    self._hash_counter += 1
    if self._hash_counter < self._hash_interval:
        return                     # skip — use cached attenuation
    self._hash_counter = 0

    # 1. Fused Triton kernel: 1D grid (in_features,), K=16
    current_hashes = self._sig_db.hash_rows(self.delta_salient)

    # 2. Continuous byte comparison vs CPU-pinned window
    attenuation = self._sig_db.compute_attenuation(current_hashes)

    # 3. Async push to CPU pinned ring buffer
    self._sig_db.push(current_hashes)

    # 4. In-place uint8 update
    self._atten_byte.copy_((attenuation * 255).to(torch.uint8))
```

### Optimizer Dispatch

```
--optimizer cuda8 (default):
  CUDA8BitAdam: 76 per-param launches, ~38ms for 110M params
  dtype-agnostic (bf16/fp32 via register shift)
  warp-level __shfl_xor_sync reductions, no shared memory stalls

--optimizer triton8:
  FusedQuantizedAdam: per-param Triton kernel launches
  int8 m/v, per-block float32 scales

--optimizer adamw:
  Standard torch.optim.AdamW (fp32 states, cuDNN-optimized)
```

### Prebuilt .so

The CUDA kernel is compiled at build time via `setup.py` (CUDAExtension).
Shipped in the wheel as `_adam_8bit_cuda*.so`. Falls back to `load_inline`
JIT if the prebuilt .so is unavailable.

---

## Performance (BERT-base, SST-2, batch=16, seq=128, T1000 sm_75)

| Method | ms/step | VRAM peak | Notes |
|--------|--------:|----------:|-------|
| Standard AdamW (full finetune) | ~940ms | 1530MB | no hash, no attenuation |
| ZPackR v8 (no-hash step) | **~1239ms** | **861MB** | forward+backward+optimizer |
| ZPackR v8 (hash step) | ~1326ms | 861MB | +~87ms net hash cost |
| ZPackR v8 amortized (h=8) | ~1249ms | 861MB | 7× cheap + 1× expensive |

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
