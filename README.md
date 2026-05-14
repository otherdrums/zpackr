# ZPackR — Deterministic Per-Block Delta Attenuation

> **Warning — Experimental.**  ZPackR is in early development and not yet ready
> for production use.  APIs and training dynamics are subject to change without
> notice.  Expect breakage and iteration.

Frozen BERT base + zstd-compressed trainable delta.  Per-block zstd
compressibility directly attenuates delta contribution in the forward
pass — the delta's current compressibility IS the knowledge metric.
No dictionaries, no reindex, no calibration, no historical state.

Early testing hit 94.1% on bert-base-uncased SST-2 in 8000 steps, matching
or exceeding full fine-tune.

Depends on [PackR](https://github.com/otherdrums/packr) (MIT) for
FusedQuantizedAdam and the kernel loader.

```bash
pip install zpackr
```

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

    if step % 4 == 0:
        for module in model.modules():
            if hasattr(module, 'post_step'):
                module.post_step()
```

Or the harness:
```bash
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --post-step-interval 4 --no-velvet
```

## Architecture

```
Text → BERT forward/backward → delta bytes
                                  │
                    zstd.compress() per block → ratio
                                  │
            max(0, 1 - 1/(ratio × I_MAX)) → attenuation [0,1]
                                  │
            Forward: delta *= (1 - attenuation) per block
            Gate:    if all blocks ≥ 0.9 attenuation → skip backward
```

**Zero-delta blocks** compress at 13,000x+ → attenuation ≈ 1.0 → fully suppressed.  
**Non-zero delta blocks** compress at ~1.27x (bf16 entropy floor) → attenuation ~1% → train fully.  
**Converging delta blocks** — as zstd finds more structure → ratio climbs → attenuation rises proportionally.

The delta's current compressibility IS the knowledge metric.  The formula comes
from Algorithmic Information Theory (AIT):

```python
# I_MAX = bf16 entropy floor (measured: zstd compresses random bf16 to ~79%)
I_MAX = 1.0 / 1.27  # ≈ 0.79
attenuation = max(0.0, 1.0 - 1.0 / (ratio * I_MAX))
```

- ratio = 1.0 (no compression, max entropy) → attenuation = 0.0 (fully active)
- ratio = 1.27 (bf16 entropy floor) → attenuation = 0.003 (nearly fully active)
- ratio = 5.0 (5x compression) → attenuation = 0.747 (mostly suppressed)
- ratio = 13,000 (zero-delta) → attenuation ≈ 1.0 (fully suppressed)

The only constant is `I_MAX` — empirically measurable, theoretically grounded.
No RATIO_FLOOR, no RATIO_CEILING, no historical tracking.

### Layer Architecture

Each ZPackRLinear layer stores:

| Component | Location | Type | Trainable | Purpose |
|-----------|----------|------|:---------:|---------|
| `base_W` | GPU | bf16 | No | Frozen BERT pretrained weight |
| `delta_salient` | GPU | bf16 | **Yes** | Only kept (salient) delta blocks |
| `_full_delta` | CPU | bf16 | — | Authoritative full delta matrix |
| `_zstd_delta` | CPU | bytes | — | zstd-compressed delta for checkpoint |
| `block_mask` | — | bool[N] | — | Which delta blocks are in VRAM |

**Forward**: `output = x @ (base_W + delta * (1 - attenuation))` — single cuBLAS matmul.

### Per-Step Data Flow

```
1. Forward:       delta *= (1 - attenuation) per block → combined matmul → output
2. Backward:      grad flows only to delta_salient (base_W is frozen)
3. Optimizer.step() + optionally Velvet for per-layer LR adaptation
4. Async compress: zstd compress per block in background thread → ratios
5. Attenuation:   max(0, 1 - 1/(ratio × I_MAX)) — AIT-derived, single constant
6. Pruning:       blocks at/above RATIO_CEILING * 0.75 evicted from delta_salient
7. Gate:          convergence check — all blocks ≥ 0.9 → skip backward
```

### Convergence Gate

Checks whether ALL blocks across ALL layers have attenuation ≥
`ATTENUATION_SKIP_THRESHOLD` (default 0.9).  If so, backward is skipped.

Future: auto-terminate training when gate skip rate saturates.

### Key Design Decisions

**zstd over LZ4**:  LZ4 produced ratios < 1.0 on non-zero bf16 (data inflation),
making the AIT formula impossible.  zstd gives ratios > 1.0 (~1.27 for active
deltas, 13,000+ for zeros).  This range makes the AIT formula work.

**No dictionaries, no reindex, no calibration**:  The WeightDict was eliminated.
Per-block zstd compressibility provides a clean signal with zero maintenance.

**AIT-derived formula**:  `attenuation = max(0, 1 - 1/(ratio × I_MAX))`.
From Algorithmic Information Theory: the fraction of a block's bytes that are
compressible equals the fraction of knowledge already encoded.  Single constant
`I_MAX ≈ 0.79` (bf16 entropy floor), empirically measurable.

**zstd creep characterization** (May 2026):  On a fixed batch, zstd ratios creep at
0.001-0.009%/step (deeper layers faster).  Total creep over 300 steps: 0.18% (layer 0)
to 1.37% (layer 11).  zstd noise floor is effectively zero (deterministic for same bytes).
Signal is clean and persistent — no EMA or state tracking needed because the delta
itself IS the history.

## Training Harness

```bash
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --no-velvet --label my_run
```

| Flag | Default | Description |
|------|:-------:|-------------|
| `--task` | `sst2` | GLUE task |
| `--max-steps` | `2000` | Training steps |
| `--eval-interval` | `500` | Steps between evals |
| `--eval-steps` | `20` | Eval batches per run |
| `--velvet` / `--no-velvet` | on | VelvetController per-layer LR |
| `--attenuation-skip` / `--no-attenuation-skip` | on | Convergence gate |
| `--attenuation-skip-threshold` | `0.9` | Attenuation threshold for gate |
| `--batch-size` | `16` | Per-GPU batch size |
| `--lr` | `2e-5` | Learning rate |
| `--label` | `""` | Prefix for output directory |
| `--output-dir` | `runs` | Base output directory |
| `--seed` | `42` | Random seed |

### Output Structure

```
runs/my_run_2026-05-14_124350_704b6e3/
  metrics.jsonl         # per-step: loss, step_ms, gate_skipped, salience, vram
  ratio_log.jsonl       # per-step per-block: ratio, attenuation, delta_l2
  config.json           # full TrainerConfig snapshot
  summary.json          # final stats: peak_vram_mb, final_eval_metric, gate_skip_rate
  checkpoints/
    step_2000/
      0.meta, 0.zstd, 0.mask, 0.base_W   # per-layer delta state
      trainer_state.pt                    # optimizer + Velvet
```

## Components

| Module | Purpose |
|--------|---------|
| `zpackr_layer.py` | ZPackRLinear — frozen base + zstd delta, forward matmul with attenuation |
| `config.py` | ZPackRConfig dataclass |
| `layer_patcher.py` | `compress_model()` — replaces nn.Linear with ZPackRLinear |
| `prompt_gate.py` | `should_skip_backward()` — convergence-driven backward skip |
| `checkpoint.py` | Model-level save/load with zstd-compressed deltas |

## Tunables

| Constant | Default | Meaning |
|----------|:-------:|---------|
| `I_MAX` | `0.79` | bf16 entropy floor (1/1.27), the only constant in the AIT formula |
| `ATTENUATION_SKIP_THRESHOLD` | `0.9` | Gate fires when all blocks ≥ this attenuation |

## License

MIT
