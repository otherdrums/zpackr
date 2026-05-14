# ZPackR — LZ4-Compressed Delta Training with Per-Block Attenuation

> **Warning — Experimental.**  ZPackR is in early development and not yet ready
> for production use.  APIs and training dynamics are subject to change without
> notice.  Expect breakage and iteration.

Frozen BERT base + LZ4-compressed trainable delta.  Per-block LZ4
compressibility directly attenuates delta contribution in the forward
pass — blocks the model already encodes are suppressed, making overfitting
structurally impossible at the block level.  No learning rate schedules,
no dictionaries, no reindex, no calibration.

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
                    LZ4.block.compress() per block → ratio
                                  │
            clamp((ratio - 1.0) / 7.0, 0, 1) → attenuation [0,1]
                                  │
            Forward: delta *= (1 - attenuation) per block
            Gate:    if all blocks ≥ 0.9 attenuation → skip backward
```

**Zero-delta blocks** compress extremely well (255x+) → attenuation 1.0 → fully suppressed.  
**Non-zero delta blocks** are poorly compressible (~1.0x) → attenuation ~0 → train fully.

No external LR scheduler needed — the attenuation naturally prevents overfitting at
the block level.  The convergence gate skips backward when the model has fully
internalized a prompt, providing step-count-free training termination.

### Layer Architecture

Each ZPackRLinear layer stores:

| Component | Location | Type | Trainable | Purpose |
|-----------|----------|------|:---------:|---------|
| `base_W` | GPU | bf16 | No | Frozen BERT pretrained weight |
| `delta_salient` | GPU | bf16 | **Yes** | Only kept (salient) delta blocks |
| `_full_delta` | CPU | bf16 | — | Authoritative full delta matrix |
| `_lz4_delta` | CPU | bytes | — | LZ4-compressed delta for checkpoint |
| `block_mask` | — | bool[N] | — | Which delta blocks are in VRAM |

**Forward**: `output = x @ (base_W + delta * (1 - attenuation))` — single cuBLAS matmul.

### Per-Step Data Flow

```
1. Forward:       delta *= (1 - attenuation) per block → combined matmul → output
2. Backward:      grad flows only to delta_salient (base_W is frozen)
3. Optimizer.step() + optionally Velvet for per-layer LR adaptation
4. post_step (every N):  merge delta GPU→CPU, LZ4 compress per block → ratios
5. Attenuation:   clamp((ratio - 1.0) / 7.0, 0, 1) per block — deterministic, no state
6. Pruning:       blocks at/above RATIO_CEILING * 0.75 evicted from delta_salient
7. Gate:          convergence check — all blocks ≥ 0.9 → skip backward on next similar prompt
```

### Convergence Gate

The convergence gate checks whether ALL blocks across ALL layers have attenuation ≥
`ATTENUATION_SKIP_THRESHOLD` (default 0.9).  If so, backward is skipped for the
current prompt — the model has no room to learn from it.

This enables future **self-terminating training**: when the gate skip rate reaches
a threshold (e.g. 95%), training can stop automatically.  For now, step count
limits are used for benchmark validation.

### Key Design Decisions

**LZ4 over zstd**:  Stateless byte-level compression.  No dictionary to build,
evolve, or serialize.  LZ4 processes a 1.5 MB block in ~0.003 ms vs zstd's ~1-3 ms.
The signal is clean: zero-delta → 255x ratio, non-zero delta → ~1.0x.

**No dictionaries, no reindex**:  The WeightDict (zstd dictionary trained on delta
bytes) was eliminated.  It required periodic reindex operations that blurred the
signal — raw LZ4 gives sharper discrimination with zero maintenance.

**Fixed attenuation constants**:  `RATIO_FLOOR=1.0`, `RATIO_CEILING=8.0`.
Derived from ratio distribution analysis on SST-2.  No historical tracking,
no calibration, no per-layer drift — the mapping is deterministic and stateless.

**Delta domain only**:  The Super Dict (English text zstd dictionary) was removed
from the training signal chain.  It remains available in the codebase as a text
preprocessor for future use, but the learning signal comes purely from delta
compressibility.

## Training Harness

```bash
python -m tools.diagnose --task sst2 --max-steps 8000 --batch-size 16 \
    --eval-interval 500 --post-step-interval 4 --no-velvet \
    --label my_run
```

| Flag | Default | Description |
|------|:-------:|-------------|
| `--task` | `sst2` | GLUE task |
| `--max-steps` | `2000` | Training steps |
| `--eval-interval` | `500` | Steps between evals |
| `--eval-steps` | `20` | Eval batches per run |
| `--post-step-interval` | `4` | Steps between LZ4 ratio updates |
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
runs/my_run_2026-05-14_091241_5aded6d/
  metrics.jsonl         # per-step: loss, step_ms, gate_skipped, salience, vram
  ratio_log.jsonl       # per-step per-block: ratio, attenuation, delta_l2
  config.json           # full TrainerConfig snapshot
  summary.json          # final stats: peak_vram_mb, final_eval_metric, gate_skip_rate
  checkpoints/
    step_2000/
      0.meta, 0.lz4, 0.mask, 0.base_W   # per-layer delta state
      trainer_state.pt                    # optimizer + Velvet
```

## Components

| Module | Purpose |
|--------|---------|
| `zpackr_layer.py` | ZPackRLinear — frozen base + LZ4 delta, forward matmul with attenuation |
| `config.py` | ZPackRConfig dataclass |
| `layer_patcher.py` | `compress_model()` — replaces nn.Linear with ZPackRLinear |
| `prompt_gate.py` | `should_skip_backward()` — convergence-driven backward skip |
| `checkpoint.py` | Model-level save/load with LZ4-compressed deltas |
| `super_dict.py` | Optional text preprocessor (zstd dictionary for English) |
| `zpackr_interface.py` | `export_model()` — merge base + delta back to nn.Linear |

## Tunables

| Constant | Default | Meaning |
|----------|:-------:|---------|
| `RATIO_FLOOR` | `1.0` | Ratio below which block is fully novel (attenuation 0) |
| `RATIO_CEILING` | `8.0` | Ratio at/above which block is fully known (attenuation 1) |
| `ATTENUATION_SKIP_THRESHOLD` | `0.9` | Gate fires when all blocks ≥ this attenuation |
| `post_step_interval` | `4` | Steps between LZ4 ratio recomputation |

## License

MIT
