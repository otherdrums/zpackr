# ZPackR — Deterministic Per-Block Delta Attenuation

> **Warning — Experimental.** ZPackR is in early development and not yet ready
> for production use. APIs and training dynamics are subject to change without
> notice. Expect breakage and iteration.

Frozen BERT base + LSH-attenuated trainable delta. Per-block LSH (Locality-Sensitive
Hashing) of delta vectors produces a cosine similarity signal across a sliding
window — the delta's directional stability IS the convergence metric.

No dictionaries, no reindex, no calibration, no compression, no historical state
beyond a 60-step ring buffer of 64-bit hash signatures.

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from zpackr import compress_model, ZPackRConfig
from packr.optim import FusedQuantizedAdam

config = ZPackRConfig(layer_scope="ffn")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = compress_model(model, config)

model.cuda()
optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5)

for step, batch in enumerate(loader):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # LSH hash computed synchronously on GPU (~1ms total)
    for module in model.modules():
        if hasattr(module, 'compute_hash_gpu'):
            module.compute_hash_gpu()
```

Or the harness:
```bash
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --no-velvet --label my_run
```

## Architecture

```
Text → BERT forward/backward → delta on GPU
                                  │
                    LSH hash: sign(delta @ projections.T)
                                  │
         multi-scale cos_sim vs 60-step sliding window
                                  │
            attenuation = mean_sim × (1 - flatness)
                                  │
             Forward: delta *= (1 - attenuation) per block
             Gate:    if all blocks ≥ 0.99 → skip backward
```

**Novel blocks** have changing delta direction → low mean_sim → low attenuation → train fully.
**Converged blocks** have stable delta direction → high mean_sim, low flatness → attenuation ≈ 1.0.
**Diverging blocks** have varying stability across time scales → high flatness → attenuation reduced.

### The zstd problem

Raw bf16 bytes from Adam-optimized deltas all compress to ~1.27x regardless of
content (the bf16 entropy floor). No compression technique — raw, prefix dict,
or trained dict — can distinguish "known" from "novel" bf16 deltas because
gradient noise + Adam smoothing + bf16 quantization produces bytes that look
random to any compressor.

### The LSH solution

LSH operates on the delta VALUES (not bytes), preserving cosine similarity in
compact bit hashes. The sign of random projections captures the delta's
direction, which stabilizes as the block converges. Two deltas with similar
direction have similar hashes — 880x discrimination between converged and novel.

### Per-Step Data Flow

```
1. Forward:       delta *= (1 - attenuation) per block → combined matmul → output
2. Backward:      grad flows only to delta_salient (base_W is frozen)
3. Optimizer:     FusedQuantizedAdam updates delta_salient on GPU
4. Hash (GPU):    LSH hash → push to sliding window → multi-scale comparison
5. Attenuation:   mean_sim × (1 - flatness) — pure function, no thresholds
6. Gate:          convergence check — all blocks ≥ 0.99 → skip backward
```

### Convergence Gate

Checks whether ALL blocks across ALL layers have attenuation ≥
`ATTENUATION_SKIP_THRESHOLD` (default 0.99). If so, backward is skipped.

Hash is computed every step even when gate fires, so the window keeps evolving.
If any block dips below threshold, the gate opens and training resumes.

### Key Design Decisions

**LSH over zstd/LZ4**: Compression on bf16 bytes can't distinguish known from
novel (entropy floor). LSH on delta values gives 880x discrimination, ~1ms on
GPU vs ~400ms CPU, with no thread complexity.

**Sliding window over prompt table**: A 60-step ring buffer of 64-bit hashes
(~2MB) replaces the 67K-entry per-prompt table (~2.5GB). Multi-scale comparison
(offsets 1, 5, 10, 25, 50) captures convergence at multiple time scales.

**Pure deterministic computation**: Attenuation is `mean_sim × (1 - flatness)`.
No thresholds, no conditionals, no historical state beyond the ring buffer.
Every component is a pure function of current + recent state.

**No pruning**: All blocks stay active on GPU, just attenuated. Pruning was
removed in v4 because the attenuation signal alone prevents overfitting without
the complexity of managing block masks.

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
| `--attenuation-skip-threshold` | `0.99` | Attenuation threshold for gate |
| `--batch-size` | `16` | Per-GPU batch size |
| `--lr` | `2e-5` | Learning rate |
| `--label` | `""` | Prefix for output directory |
| `--output-dir` | `runs` | Base output directory |
| `--seed` | `42` | Random seed |

### Output Structure

```
runs/my_run_2026-05-14_runid/
  metrics.jsonl         # per-step: loss, step_ms, gate_skipped, salience, vram
  ratio_log.jsonl       # per-step per-block: attenuation, delta_l2
  config.json           # full TrainerConfig snapshot
  summary.json          # final stats: peak_vram_mb, final_eval_metric, gate_skip_rate
  checkpoints/
    step_N/
      0.meta, 0.zstd, 0.mask, 0.base_W   # per-layer delta state
      trainer_state.pt                    # optimizer + Velvet
```

## Components

| Module | Purpose |
|--------|---------|
| `zpackr_layer.py` | ZPackRLinear, DeltaSignatureDB, `compute_hash_gpu()` |
| `config.py` | ZPackRConfig dataclass |
| `layer_patcher.py` | `compress_model()` — replaces nn.Linear with ZPackRLinear |
| `prompt_gate.py` | `should_skip_backward()` — convergence-driven backward skip |
| `checkpoint.py` | Model-level save/load with zstd-compressed deltas |
| `super_dict.py` | Optional zstd text preprocessor (archived) |

## Tunables

| Constant | Default | Meaning |
|----------|:-------:|---------|
| `K` (lsh_K) | `128` | LSH hash bits (higher = finer resolution, more memory) |
| `WINDOW_SIZE` | `60` | Sliding window of hash snapshots |
| `LSH_OFFSETS` | `(1,5,10,25,50)` | Multi-scale comparison offsets |
| `ATTENUATION_SKIP_THRESHOLD` | `0.99` | Gate fires when all blocks ≥ this |

## License

MIT
