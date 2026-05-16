# ZPackR — Deterministic Per-Row Delta Attenuation

> **Warning — Experimental.** ZPackR is in early development and not yet ready
> for production use. APIs and training dynamics are subject to change without
> notice. Expect breakage and iteration.

Frozen BERT base + LSH-attenuated trainable delta. **Per-row** LSH (Locality-Sensitive
Hashing) of delta vectors produces a cosine similarity signal across a sliding
window — each row's directional stability IS its convergence metric.

Hash computed via fused Triton kernel (1D grid, one block per row
processes all K=16 projections — 16× less delta memory traffic than 2D grid).
Attenuation stored as uint8 GPU tensor — 256 levels of resolution per row.

No dictionaries, no reindex, no calibration, no compression, no historical state
beyond a 4200-step ring buffer of packed LSH hashes (2 bytes/row).

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from zpackr import compress_model, ZPackRConfig
from packr.cuda_adam import CUDA8BitAdam

config = ZPackRConfig(layer_scope="ffn", bf16=True, hash_interval=4)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = compress_model(model, config)
model.cuda()

# Dtype-agnostic optimizer: handles bf16 or fp32 automatically
optimizer = CUDA8BitAdam(model.parameters(), lr=2e-5)

for step, batch in enumerate(loader):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    for module in model.modules():
        if hasattr(module, 'compute_hash_gpu'):
            module.compute_hash_gpu()
```

Or the harness:
```bash
# Full quality with VRAM savings (bf16 model + CUDA 8-bit optimizer)
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --no-velvet --bf16 --hash-interval 4 --optimizer cuda8 --label my_run

# FP32 model (no VRAM savings, same optimizer speed)
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --no-velvet --hash-interval 4 --optimizer cuda8 --label my_run
```

## Architecture

```
Text → BERT forward/backward → delta on GPU
                                  │
               Fused Triton kernel (1D grid, K=16)
              row[i] loaded ONCE → 16 dot products → 16 bits
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

The hash is computed by a fused Triton kernel (`_lsh_hash_fused_kernel`) with a
1D grid of `(in_features,)` blocks. Each block loads `delta[row]` ONCE and
computes all K=16 dot products, storing K hash bits in a single `tl.store`.
This replaces the old 2D grid (16× more blocks, 16× more delta memory traffic)
and the even older 24 separate cuBLAS calls.

### Per-Step Data Flow

```
1. Forward:       delta *= (1 - attenuation/255) per row → combined matmul → output
2. Backward:      grad flows only to delta_salient (base_W is frozen)
3. Optimizer:     CUDA8BitAdam | FusedQuantizedAdam | AdamW (dtype-agnostic)
4. Hash (GPU):    fused Triton kernel (1D grid, K=16) → 2 bytes/row → CPU pinned window
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
drowned novel ones in the same block, capping accuracy at ~92% on BERT-base
SST-2. Row-level achieves 93-94% by giving each row its own convergence signal.

**Fused Triton kernel over cuBLAS**: The fused `_lsh_hash_fused_kernel` with a
1D `(in_features,)` grid loads each delta row ONCE and computes all K=16 dot
products in one block. Compared to the old 2D grid (one block per (row, k)):
16× fewer blocks, 16× less delta memory traffic, K bits stored in one `tl.store`.
Compared to the original cuBLAS approach: 24 separate matmul launches eliminated.

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
| `--task` | `sst2` | GLUE task (tested on BERT-base SST-2) |
| `--max-steps` | `500` | Training steps |
| `--eval-interval` | `100` | Steps between evals |
| `--eval-steps` | `20` | Eval batches per run |
| `--bf16` | off | Convert model to bfloat16 (saves ~60MB VRAM) |
| `--optimizer` | `cuda8` | Optimizer: `cuda8` (CUDA 8-bit), `triton8`, `adamw` |
| `--hash-interval` | `1` | Hash every N steps (4-8 saves speed, same quality) |
| `--velvet` / `--no-velvet` | on | VelvetController per-layer LR |
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
| `zpackr_layer.py` | ZPackRLinear, DeltaSignatureDB, `_lsh_hash_fused_kernel` (fused Triton 1D), `compute_hash_gpu()` |
| `config.py` | ZPackRConfig dataclass |
| `layer_patcher.py` | `compress_model()` — replaces nn.Linear with ZPackRLinear |
| `prompt_gate.py` | `should_skip_backward()` — convergence-driven backward skip |
| `checkpoint.py` | Model-level save/load with zstd-compressed deltas |
| `super_dict.py` | Optional zstd text preprocessor (archived) |

## Tunables

| Constant | Default | Meaning |
|----------|:-------:|---------|
| Tunable | Default | Meaning (BERT-base, SST-2) |
|---------|:-------:|--------------------------|
| `K` (lsh_K) | `16` | LSH hash bits (packed into K//8 bytes, continuous byte comparison) |
| `WINDOW_SIZE` | `4200` | Sliding window of hash snapshots (matches 1 SST-2 epoch = 67K train / 16 batch) |
| `LSH_OFFSETS` | `(1,3,10,30,100,300,1000)` | Log-spaced multi-scale comparison offsets (3× spacing, 1000× range) |
| `ATTENUATION_SKIP_THRESHOLD` | `1.0` | Gate fires when ALL rows across ALL layers have attenuation ≥ this |
| Attenuation formula | `squared` | `(mean_sim * (1 - flatness))²` — delays convergence to byte=255 |
| `hash_interval` | `1` | Hash every N steps; 4-8 gives same convergence, ~87ms net |

## License

MIT
