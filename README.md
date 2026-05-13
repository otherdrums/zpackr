# ZPackR — Zstd-Native Compressed Neural Network Training

> **Warning — Experimental.**  ZPackR is in early development and not yet ready
> for production use.  APIs and training dynamics are subject to change without
> notice.  Expect breakage and iteration.

Dual-dictionary architecture for adaptive weight compression.
Frozen BERT base + zstd-compressed trainable delta.  The WeightDict
learns to compress weight byte patterns — blocks it recognizes are
automatically attenuated and pruned from VRAM.

Early testing hit 94.1% on bert-base-uncased SST-2 in 8000 steps, matching
or exceeding full fine-tune.

Depends on [PackR](https://github.com/otherdrums/packr) (MIT) for
FusedQuantizedAdam and the kernel loader.

```bash
pip install zpackr
```
```

```python
from transformers import AutoModelForSequenceClassification
from zpackr import compress_model, ZPackRConfig
from packr.optim import FusedQuantizedAdam

config = ZPackRConfig(layer_scope="ffn")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = compress_model(model, config)
# model.super_zstd   ← frozen Super Dict (text codec)
# model.weight_dict  ← adaptive WeightDict (delta pattern compression)

model.cuda()
optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5)

for batch in loader:
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
python tools/train_harness.py --mode zpackr --task sst2 \
    --max-steps 2000 --eval-interval 500 --reindex-interval 1000
```

## Architecture

```
                         Super Dict                        Weight Dict
                         ──────────                        ───────────
Domain:              Text (English + GLUE)             bf16 delta byte patterns
Trained from:        Word list + GLUE corpus           zstd.train_dictionary()
Mutability:          FROZEN forever                    EVOLVES during training
Role:                Compresses prompts → gate         Compresses delta blocks
                     Decompresses zstd → text          → salience + VRAM mgmt
Signal:              ratio = uncomp/comp               ratio = uncomp/comp
                     "Is this text novel?"             "Did the delta change?"
```

### Layer Architecture

Each ZPackRLinear layer stores:

| Component | Location | Type | Trainable | Purpose |
|-----------|----------|------|:---------:|---------|
| `base_W` | GPU | fp16 | No | Frozen BERT pretrained weight |
| `delta_salient` | GPU | bf16 | **Yes** | Only kept (salient) delta blocks |
| `_full_delta` | CPU | bf16 | — | Authoritative full delta matrix |
| `zstd_delta` | CPU | bytes | — | Compressed delta for checkpoint |
| `block_mask` | — | bool[N] | — | Which delta blocks are in VRAM |

**Forward**: `output = x @ base_W + block_accumulate(x, delta_salient)`

The base matmul uses fp16 cuBLAS (fast path). The delta uses block-accumulate
over only the kept blocks, never materializing a full `[in, out]` delta matrix.

VRAM drops as delta blocks get pruned — when the WeightDict recognizes a block's
byte pattern (ratio ≥ threshold), its rows are evicted from `delta_salient`.

### Per-Step Data Flow

```
1. Pre-forward:   Super Dict ratio(prompt) ≥ threshold? → gate decision
2. Forward:       x @ base_W + block_accumulate(x, delta_salient) → output
3. Backward:      grad flows only to delta_salient (base_W is frozen)
4. post_step:     Merge delta GPU→CPU, compress per-block vs WeightDict
5. Salience:      ratio < threshold → keep block; ratio ≥ threshold → prune
6. Resize:        delta_salient rebuilt from _full_delta[kept_blocks]
7. Reindex:       zstd.train_dictionary() rescans delta bytes for patterns
```

### Zstd-Native Prompting

Prompts can be stored, transmitted, and fed to the model as zstd-compressed bytes:

```python
from packr import prompt_zstd, export_model

# Compress a prompt
compressed, ratio = model.super_zstd.prompt_roundtrip("Is this review positive?")

# Inference directly from zstd
output, ratio = prompt_zstd(model, compressed)

# Continuous learning: novel prompts trigger training
output, ratio, trained = prompt_zstd_with_learning(model, compressed, threshold=2.0)

# Export merged weights as standard HuggingFace model
export_model(model, output_path="./exported_model")
```

## Key Design Decisions

**Frozen base + zstd delta (not full-weight compression):**
Storing the BERT pretrained weight as a frozen fp16 parameter plus a bf16
trainable delta is more like a full-rank LoRA adapter. The base matmul
uses optimized cuBLAS, the delta uses block-accumulate over only salient blocks.
Export merges back to standard `nn.Linear` for inference.

**Accumulating WeightDict (not sealed/frozen):**
The WeightDict caches BERT base weight chunks at setup and adds delta
chunks at each periodic reindex. This means:
- Base patterns form the signal floor (zero-delta = 22,469:1 ratio)
- Delta patterns accumulate on top (the dict evolves with training)
- Signal gap stays at 17,000x+ — crystal clear separation between
  unchanged blocks (matching BERT) and changed blocks (diverging)

**WeightDict via zstd.train_dictionary() (not manual 16-byte windows):**
Manual sliding-window pattern extraction found zero repeats in bf16 data.
zstd's production-quality suffix-array trainer finds optimal compression
patterns from chunked weight bytes automatically.

**Auto-calibrating per-layer threshold (not fixed 2.0):**
Bf16 weight bytes compress at ~1.3:1, not text-like 40:1. The roadmap's
2.0 threshold is unachievable for floating-point data. Per-layer calibration
sets threshold at 1% of max observed ratio, cleanly separating zero blocks
(ratio 22,469) from non-zero blocks (ratio ~1.3).

**Sampled empty-dict guard:**
No calibration until the WeightDict has entries and ratios show variation
across blocks. Prevents catastrophic pruning on uniform zero-delta at init.

**zstd level 1 (not 3):**
3-5x faster compression with negligible ratio difference for bf16 data.
The ratio signal depends on compressibility, not bytes saved.

**nvcomp GPU compression not used:**
nvcomp's GPU zstd is 2.4x slower than CPU zstd L1 for bf16 weight blocks
(417ms vs 173ms for 180 blocks). GPU kernel launch overhead dominates at
1.5MB block sizes. Only beneficial for 100MB+ datasets.

## Training Harness

```bash
python tools/train_harness.py --mode zpackr --task sst2 \
    --max-steps 2000 \
    --eval-interval 500 \
    --eval-steps 20 \
    --post-step-interval 4 \
    --reindex-interval 1000 \
    --batch-size 16 \
    --lr 2e-5 \
    --label zpackr \
    --output-dir runs/comparison
```

| Flag | Default | Description |
|------|:-------:|-------------|
| `--mode` | `zpackr` | `packr` or `zpackr` |
| `--task` | `sst2` | GLUE task |
| `--max-steps` | `2000` | Training steps |
| `--eval-interval` | `500` | Steps between evals |
| `--eval-steps` | `20` | Eval batches per run |
| `--post-step-interval` | `4` | Steps between salience updates |
| `--reindex-interval` | `1000` | Steps between WeightDict reindex |
| `--warmup-steps` | `0` | Velvet warmup (0 = auto) |
| `--velvet` / `--no-velvet` | on | VelvetController |
| `--gate` / `--no-gate` | off | Super Dict training gate |
| `--gate-threshold` | `2.0` | Super Dict ratio threshold |
| `--batch-size` | `16` | Per-GPU batch size |
| `--lr` | `2e-5` | Learning rate |
| `--label` | `""` | Prefix for output directory |
| `--output-dir` | `runs` | Base output directory |
| `--seed` | `42` | Random seed |

### Output Structure

```
runs/comparison/zpackr_2026-05-11_225956_f74b7ae/
  metrics.jsonl         # per-step: loss, salience, vram_peak_mb, salient_vram_kb,
                        #   salient_vram_fraction, weight_dict_entries, velvet_multipliers
  config.json           # full TrainerConfig snapshot
  summary.json          # final stats: peak_vram_mb, final_eval_metric, gate_skip_rate
  checkpoints/
    step_2000/
      0.meta, 0.zstd, 0.mask, 0.wd.*, 0.base_W   # per-layer delta state
      trainer_state.pt                             # optimizer + Velvet
```

### Comparing Runs

```bash
python tools/report.py runs/comparison/ --compare
python tools/ablate.py --task sst2 --max-steps 500 \
    --param mode:packr,zpackr \
    --param post_step_interval:1,4,16
```

## Components

| Module | Purpose |
|--------|---------|
| `super_dict.py` | Frozen text codec — compress/decompress prompts, zstd-native interface |
| `prompt_gate.py` | Binary train/skip gate from Super Dict ratio |
| `zstd_dict.py` | WeightDict — adaptive zstd dictionary via `train_dictionary()` |
| `salience.py` | Block-level compression ratio → bool mask |
| `zpackr_layer.py` | ZPackRLinear — frozen base + zstd delta, block-accumulate forward |
| `zpackr_interface.py` | `prompt_zstd()`, `prompt_zstd_with_learning()`, `export_model()` |
| `checkpoint.py` | Model-level save/load for reversible training eras |

## Tunables

ZPackR uses **self-calibrating per-layer thresholds** by default —
no manual threshold tuning needed.  Each layer auto-calibrates on the
first post_step after reindex.

| Parameter | Default | Effect |
|-----------|:-------:|--------|
| `zstd_salience_threshold` | **auto** | Manual override for salience threshold |
| `zstd_max_entries` | `16384` | Max patterns in WeightDict |
| `post_step_interval` | `4` | Steps between delta salience updates |
| `reindex_interval` | `1000` | Steps between WeightDict reindex |
| `gate_threshold` | `2.0` | Super Dict ratio for train/skip on text |

### How Auto-Calibration Works

1. **First post_step after reindex**: scans all blocks, records the max compression
   ratio.  Sets threshold at 1% of that max.  Does NOT prune on this pass.
2. **Subsequent post_steps**: compares each block's ratio against the calibrated
   threshold.  Blocks whose ratio dropped below the threshold (became less
   compressible = novel) are **kept**.  Blocks at or above the threshold (still
   compressible = learned/unchanged) are **pruned**.
3. **Per-layer tracking**: `salience_thresholds` in `metrics.jsonl` shows each
   layer's current threshold.  Resets on each reindex.

This handles both starting conditions:
- **Zero delta** (fresh training): thresholds are high (~200), cleanly separating
  zero blocks (ratio 20000+) from trained blocks (ratio ~1.3)
- **Non-zero delta** (resume training): thresholds start low (~0.01), keeping
  everything until patterns emerge

## Deferred (v2.1+)

- Mini-dict MoE ensemble (WeightDict splits into task specialists)
- Weight Dict-driven expert routing for multi-task continual learning
- Continuous learning from inference (partial implementation in `prompt_zstd_with_learning`)
