# ZPackR — Deterministic Per-Row Delta Attenuation

> **Warning — Experimental.** ZPackR is in early development and not yet ready
> for production use. APIs and training dynamics are subject to change without
> notice. Expect breakage and iteration.

Frozen BERT base + LSH-attenuated trainable delta. **Per-row** LSH (Locality-Sensitive
Hashing) of delta vectors produces a cosine similarity signal across a sliding
window — each row's directional stability IS its convergence metric.

Hash computed via custom Triton kernel (single 2D launch, no float32 intermediate).
Attenuation stored as uint8 GPU tensor — 256 levels of resolution per row.

No dictionaries, no reindex, no calibration, no compression, no historical state
beyond a 4200-step ring buffer of packed LSH hashes (2 bytes/row).

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from zpackr import compress_model, ZPackRConfig
from packr.optim import FusedQuantizedAdam

config = ZPackRConfig(layer_scope="ffn", bf16=True)  # bf16 saves ~100MB VRAM
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = compress_model(model, config)

model.cuda()
optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5)

for step, batch in enumerate(loader):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # LSH hash computed synchronously on GPU via custom Triton kernel
    for module in model.modules():
        if hasattr(module, 'compute_hash_gpu'):
            module.compute_hash_gpu()
```

Or the harness:
```bash
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --no-velvet --bf16 --label my_run
```

## Architecture

```
Text → BERT forward/backward → delta on GPU
                                  │
               Custom Triton kernel: sign(delta @ projections.T)
                   2D grid (in_features × K=16), single launch
                                  │
         log-spaced multi-scale cos_sim vs 4200-step window
            offsets: (1, 3, 10, 30, 100, 300, 1000)
            continuous byte comparison — ~512 levels/offset
                                  │
         attenuation = (mean_sim × (1 - flatness))² → uint8 [0, 255]
                                  │
               Forward: delta *= (1 - nv/255) per row
               Gate:    if all rows ≥ 1.0 → skip backward
```

**Novel rows** have changing delta direction → low mean_sim → low attenuation → train fully.  
**Converged rows** have stable delta direction → high mean_sim, low flatness → attenuation ≈ 255.  
**Diverging rows** have varying stability across time scales → high flatness → attenuation reduced.

### The zstd problem (why we don't compress)

Raw bf16 bytes from Adam-optimized deltas all compress to ~1.27x regardless of
content (the bf16 entropy floor). No compression technique — raw, prefix dict,
or trained dict — can distinguish "known" from "novel" bf16 deltas because
gradient noise + Adam smoothing + bf16 quantization produces bytes that look
random to any compressor.

### The LSH solution

LSH operates on the delta VALUES (not bytes), preserving cosine similarity in
compact bit hashes. The sign of random projections captures the delta's
direction, which stabilizes as the row converges. Two deltas with similar
direction have similar hashes — 880x discrimination between converged and novel.

The hash is computed by a custom Triton kernel (`_lsh_hash_kernel`) with a
2D grid of `(in_features, K)` blocks. Each block computes one dot product
`(delta[row] · proj[k])` and stores the sign bit. Single kernel launch replaces
24 separate cuBLAS calls and avoids the float32 intermediate tensor.

### Per-Step Data Flow

```
1. Forward:       delta *= (1 - attenuation/255) per row → combined matmul → output
2. Backward:      grad flows only to delta_salient (base_W is frozen)
3. Optimizer:     FusedQuantizedAdam updates delta_salient on GPU
4. Hash (GPU):    Triton kernel (K=16) → sign bits → packed to 2 bytes → 4200-step window
5. Attenuation:   continuous byte comparison → log-spaced multi-scale (×7) → (mean_sim × (1 - flatness))² → uint8
6. Gate:          convergence check — all rows ≥ 1.0 → skip backward
```

### Convergence Gate

Checks whether ALL rows across ALL layers have attenuation ≥
`ATTENUATION_SKIP_THRESHOLD` (default 1.0). With `min()` semantics — every
single row must be fully converged.

Hash is computed every step even when gate fires, so the window keeps evolving.
If any row dips below threshold, the gate opens and training resumes.

### Key Design Decisions

**Row-level over block-level**: Each row of the weight matrix converges
independently. Block-level (256 rows grouped) over-attenuated — converged rows
drowned novel ones in the same block, capping accuracy at 92%. Row-level
achieves 93-94% by giving each row its own convergence signal.

**Triton kernel over cuBLAS**: A custom `_lsh_hash_kernel` with `(in_features, K)`
grid computes all row projections in a single launch, avoids the float32
intermediate (delta.float()), and writes uint8 directly. Same FMA count,
less memory bandwidth.

**uint8 attenuation over float**: 256 levels (one byte per row) exceeds the
resolution of K=32 (32 levels) or K=64 (64 levels). Stored as `register_buffer`
on GPU — zero `torch.tensor(list)` calls in forward, zero Python→GPU syncs.

**Sliding window over prompt table**: A 4200-step ring buffer of 16-bit hashes
packed into 2 bytes (~378MB for 46K rows). Continuous byte comparison
(`1 - |diff| / 255`) gives ~512 similarity levels per offset — 8× the resolution
of bit-by-bit matching. Log-spaced multi-scale offsets (1, 3, 10, 30, 100, 300,
1000) capture convergence at time scales from 1 to 1000 steps.

**Continuous byte comparison**: Instead of counting matching bits, measure the
byte-level difference between packed hashes. Two hashes that differ by 1 bit 
out of 16 produce byte values differing by 1/255 → similarity ≈ 0.996. Random
hashes differ by ~85/255 → similarity ≈ 0.67. This gives a smooth, continuous
similarity signal even with compact K=16 hashes.

**Pure deterministic computation**: Attenuation is `mean_sim × (1 - flatness)`
→ `(value * 255).to(torch.uint8)`. No thresholds, no conditionals, no historical
state beyond the ring buffer. Every component is a pure function of current +
recent state.

**No pruning**: All rows stay active on GPU, just attenuated. Pruning was
removed in v4 because the attenuation signal alone prevents overfitting without
the complexity of managing row masks.

## Training Harness

```bash
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --no-velvet --label my_run
```

| Flag | Default | Description |
|------|:-------:|-------------|
| `--task` | `sst2` | GLUE task |
| `--max-steps` | `500` | Training steps |
| `--eval-interval` | `100` | Steps between evals |
| `--eval-steps` | `20` | Eval batches per run |
| `--velvet` / `--no-velvet` | on | VelvetController per-layer LR |
| `--attenuation-skip` / `--no-attenuation-skip` | on | Convergence gate |
| `--attenuation-skip-threshold` | `1.0` | Attenuation threshold for gate |
| `--batch-size` | `16` | Per-GPU batch size |
| `--lr` | `2e-5` | Learning rate |
| `--label` | `""` | Prefix for output directory |
| `--output-dir` | `runs` | Base output directory |
| `--seed` | `42` | Random seed |

### Output Structure

```
runs/my_run_2026-05-14_runid/
  metrics.jsonl         # per-step: loss, step_ms, gate_skipped, salience, vram
  ratio_log.jsonl       # per-step per-row: attenuation, delta_l2
  config.json           # full TrainerConfig snapshot
  summary.json          # final stats: peak_vram_mb, final_eval_metric, gate_skip_rate
  checkpoints/
    step_N/
      0.meta, 0.zstd, 0.base_W   # per-layer delta state
      trainer_state.pt            # optimizer + Velvet
```

## Components

| Module | Purpose |
|--------|---------|
| `zpackr_layer.py` | ZPackRLinear, DeltaSignatureDB, `_lsh_hash_kernel` (Triton), `compute_hash_gpu()` |
| `config.py` | ZPackRConfig dataclass |
| `layer_patcher.py` | `compress_model()` — replaces nn.Linear with ZPackRLinear |
| `prompt_gate.py` | `should_skip_backward()` — convergence-driven backward skip |
| `checkpoint.py` | Model-level save/load with zstd-compressed deltas |
| `super_dict.py` | Optional zstd text preprocessor (archived) |

## Tunables

| Constant | Default | Meaning |
|----------|:-------:|---------|
| `K` (lsh_K) | `16` | LSH hash bits (packed into K//8 bytes, continuous byte comparison) |
| `WINDOW_SIZE` | `4200` | Sliding window of hash snapshots (matches 1 SST-2 epoch) |
| `LSH_OFFSETS` | `(1,3,10,30,100,300,1000)` | Log-spaced multi-scale comparison offsets (3x spacing) |
| `ATTENUATION_SKIP_THRESHOLD` | `1.0` | Gate fires when all rows ≥ this (min over rows) |
| Attenuation formula | `squared` | `(mean_sim * (1 - flatness))²` — compresses distribution |

## License

MIT
