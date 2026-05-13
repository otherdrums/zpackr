# ZPackR Implementation Guide — v2.0

Target repo: `github.com/otherdrums/packr`
Branch: `master` (or new `feature/zpackr`)
Package: `packr` (pip-installable via `pyproject.toml`)
Branch: `feature/zpackr` (development)

Two modes:
- `mode="packr"` (default, unchanged) — fixed 256-entry LUT, 3 bytes/weight
- `mode="zpackr"` (v2.0) — dual-dict architecture: static Super Dict (text codec) +
  adaptive Weight Dict (VRAM salience + checkpoint magic)


## Implementation Notes (deviations from original spec)

These notes document design decisions made during implementation that diverged
from the original roadmap, with rationale.

### Frozen Base + Zstd Delta (vs full-weight compression)

**Originally specified**: ZPackRLinear stores the full weight matrix as
zstd-compressed bytes, with salient blocks decompressed into VRAM.

**Implemented**: Frozen BERT pretrained `base_W` (fp16, not trainable) +
trainable `delta_salient` (bf16, zstd-compressible). Forward is
`x @ base_W + block_accumulate(x, delta)`.

**Why**: This is more like a full-rank LoRA adapter. The base weights never
change — only the delta gets trained and compressed. This means:
- Base matmul uses optimized fp16 cuBLAS (fast)
- Delta starts as zeros (no patterns yet → all blocks novel → all kept)
- WeightDict learns patterns from delta bytes as training produces non-zero deltas
- VRAM savings from delta pruning, not base pruning
- Export merges back to standard nn.Linear for inference deployment

GPU utilization improved from 55% to 94-100% with this architecture because
the base matmul is cuBLAS-optimized and the delta fast-path avoids Python loops.

### WeightDict: zstd.train_dictionary() vs Manual 16-Byte Window

**Originally specified**: Slide a 16-byte window with stride=8 over weight
bytes, count frequencies of exact 16-byte sequences, add patterns appearing
in ≥10% of blocks.

**Implemented**: Use `zstd.train_dictionary()` to find optimal compression
patterns from chunked weight bytes. No manual window scanning or LRU eviction.

**Why**:
- Random/near-random bf16 weight bytes have ZERO repeated 16-byte sequences
  (verified: 1250 unique patterns in 10K windows, no repeats)
- The original spec required 59K exact 16-byte matches per block (impossible)
- zstd's trainer uses production-quality pattern extraction (suffix arrays,
  optimal dictionary construction)
- Dictionary training on 4.7MB of chunked data produces 7K+ entries in <200ms
- Chunking into 8KB samples required for zstd's multi-sample training API

### Salience Threshold: 1.4 vs 2.0

**Originally specified**: `ratio >= 2.0 → prune, ratio < 2.0 → keep`

**Implemented**: Default threshold 1.4

**Why**: The 2.0 threshold was calibrated for text compression (English text
achieves 40:1 ratios with a trained dictionary). Bf16 weight bytes achieve
only ~1.3:1 compression due to floating-point entropy. At initialization,
all delta blocks have ratio ~1.3. Threshold 1.4 keeps everything at init
and prunes blocks whose compressibility improves during training.

### Zstd Level: 1 vs 3

**Originally specified**: Not specified (would have defaulted to zstd default)

**Implemented**: zstd compression level 1

**Why**: Level 1 is 3-5x faster than level 3 for large blocks (1.5MB each)
with negligible ratio difference for the salience signal. The ratio tells us
compressibility, not bytes saved — level 1 preserves the signal at higher speed.

### post_step: Delta-Only Compression

**Originally specified**: Compress the full weight matrix after each step.

**Implemented**: Only compress delta blocks (`_full_delta`). Base weights
are frozen and never compressed. Also made `zstd_delta` lazy — only recomputed
at checkpoint save time, not every post_step.

**Why**: Base weights never change, so re-compressing them is wasted work.
Delta bytes are smaller (start as zeros, grow slowly), making compression
faster and the salience signal more sensitive to actual training changes.

### Reindex Seeds the WeightDict from Initial Weights

**Originally specified**: WeightDict starts empty, evolved at explicit reindex
call-sites.

**Implemented**: At `from_linear()`, immediately run `reindex()` with
`min_frequency=0.001` and `min_count=3` on the initial delta (all zeros).
This captures zero-byte patterns and provides a baseline dictionary.

### Zstd-Native Prompting

**Added beyond spec**: `prompt_zstd()`, `prompt_zstd_with_learning()`,
`export_model()` in `zpackr_interface.py`. Prompts can be stored as Super
Dict-compressed zstd bytes, decompressed at the BERT input boundary.
Novel prompts (low Super Dict ratio) can trigger inline training.

### VRAM Tracking

**Added beyond spec**: Per-step `vram_peak_mb` (highest VRAM spike in each
step) and run-level `peak_vram_mb` in summary. Per-layer `salient_vram_kb`
and `salient_vram_fraction` to track delta VRAM savings directly.

### Auto-Calibrating Per-Layer Salience Threshold

**Originally specified**: Fixed global threshold `zstd_salience_threshold: 2.0`
for all layers.  `ratio >= threshold → prune, ratio < threshold → keep`.

**Implemented**: Zero-config per-layer auto-calibration.  On the first
post_step after each reindex, each layer independently calibrates its own
threshold at 1% of the maximum observed compression ratio.  This calibration
pass does NOT prune — it only sets the baseline.

On subsequent post_steps: blocks whose ratio stayed at/above the calibration
baseline are "unchanged" (still compressible) and get pruned.  Blocks whose
ratio dropped below the baseline are "changed" (less compressible = novel)
and stay in VRAM.

This handles both starting conditions cleanly:
- **Zero delta** (fresh training): calibration max is ~20000+ (all-zeros
  compress extremely well).  Threshold = 200+.  Zero blocks stay above →
  pruned.  Trained blocks drop to ~1.3 → kept.
- **Non-zero delta** (resume training): calibration max is ~1.3.  Threshold
  = 0.01.  Everything stays kept until compression patterns emerge.
- **Per-layer thresholds** tracked in `salience_thresholds` in metrics.jsonl.
  Reset on each reindex (recalibrates against updated dictionary).

### Forward Speed Optimizations

**base_W stored as bf16**: Originally fp16 for cuBLAS.  Changed to bf16 to
eliminate per-forward `.to()` dtype conversion.  Both x and base_W are
now bf16 → one less allocation per forward.

**Pre-cast x to bf16 once**: Previously `x.to(torch.bfloat16)` was called
in both the full-salience and partial-salience paths.  Now pre-cast once
at the top of forward.

**Cached salient_count**: Replaced `block_mask.sum().item()` (GPU sync per
layer) with a Python int updated in post_step.  Eliminates 24 GPU syncs
per step in metrics collection.

**Removed no-op `.to(torch.bfloat16)` on delta slices**: Delta was already
bf16 — the extra `.to()` call was pure overhead.

**WeightDict reindex uses zstd level 1**: Reindex was accidentally creating
a level-3 compressor, overriding the level-1 default.  Fixed to level 1
(3-5x faster, negligible ratio difference).

**Per-block `.tobytes()`** in post_step**: Instead of copying the full
4.7MB delta to a Python bytes object and slicing, now slices the numpy
view and calls `.tobytes()` per block.  Lower memory pressure, smaller
GC objects.

### PackR Backward Hot Path Optimizations (May 2026)

**Decode kernel used in backward**: Previously `lut[W_p.long()].to(torch.bfloat16)`
in the backward pass materialized the full weight matrix via PyTorch indexing,
creating an int64 intermediate (8× the uint8 source).  Now uses the same
`decode_packed(W_p, lut)` kernel as forward — Triton cubin / JIT / CUDA JIT /
PyTorch fallback pipeline — producing fp16 output cast to bf16.  Eliminates
the 8× int64 intermediate entirely.  For BERT's 768×3072 FFN layer this
saves ~19 MB of transient VRAM per layer per backward pass.

**Fused LUT gradient kernel**: `kernel.py` now includes a Triton JIT kernel
that fuses `torch.bincount` and `scatter_add_` into a single GPU pass.
Reads `W_p` directly as uint8 (no `.long()` expansion), does one memory
scan with two atomic adds per element (count + gradient sum per bucket).
Eliminates the 8× int64 intermediate from `W_p.flatten().long()` and halves
GPU memory bandwidth for the LUT gradient computation.  Falls back to the
pure-PyTorch bincount+scatter_add path when Triton is unavailable.

### ZPackR Forward Path Optimizations (May 2026)

**Batched block matmul**: The slow path (partial salience) previously used a
Python for-loop of per-block matmuls with `.contiguous()` copies inside.
Now uses `torch.bmm` — all kept blocks stacked into `[num_kept, M, block_size]`
× `[num_kept, block_size, N]` → one batched matmul → sum-reduce.  Replaces
N serial small matmul kernel launches with one batched launch.  Zeros-padded
for the partial last block.  The accumulator is lazily allocated per batch
size and reused.

**Helper method `_block_accumulate()`**: Extracted from `forward()` for clarity.
Manages the pre-allocated accumulator, zeroing or re-allocating as needed.

**Cached kept-block indices**: `block_mask.nonzero()` was called every forward
pass even though the mask only changes during `post_step`.  Now stored as
`_kept_indices` (updated in `_rebuild_delta_salient` when the mask changes).
Also used internally by `_sync_full_delta`.

**Guarded `.to(device)` on delta**: `self.delta_salient.to(device=dev)` was
called unconditionally every forward, emitting a no-op kernel launch even
when already resident.  Now wrapped in `if delta.device != dev`.

**Activation memory freed after bf16 cast**: When x is fp32, the original
fp32 tensor stayed alive alongside `x_bf16`, doubling activation memory.
Now `del x` after the cast frees the fp32 copy.

### Offload Path Optimizations (May 2026)

**Async W_p prefetch**: `prefetch_wp()` now uses the offload CUDA stream
(`self._stream`) for pool-reuse copies instead of the default stream.
The copies no longer block the default stream where the forward matmul
runs.  `ensure_wp()` synchronizes the offload stream before binding
a prefetched buffer.  Cold-path (`.to(device)`) remains on the default
stream since it's only called once at startup.

**Eliminated wasted `.clone()` in evict_wp**: `wp.data = self._cpu_buffers[name].clone()`
created a non-pinned CPU copy on every eviction, only to be overwritten
on the next `ensure_wp`.  Now uses the pinned canonical buffer directly
as the parameter's placeholder data.

**Import note**: `register_wp()` also eliminated its `.clone()` call.  The
parameter's `.data` now directly references the pinned canonical buffer
until the first `ensure_wp` binds a GPU tensor.

### Velvet Batch GPU Sync (May 2026)

**VelvetController now syncs once per step, not once per parameter**:
Previously `_dequantize_v_mean().item()` forced a GPU→CPU synchronization
for every parameter (~200 for BERT-base).  Now:
1. Pass 1: collect all v_mean GPU scalars via `_dequantize_v_mean_tensor()`
   (returns 0-d GPU tensor, no `.item()`)
2. Single `torch.cuda.synchronize()` after all means are collected
3. Pass 2: iterate and call `.item()` on each (now local, no sync needed)
Reduces ~200 GPU syncs/steps to 1.

### Accumulating WeightDict (Base + Delta Reindex)

**Originally specified**: WeightDict evolves via reindex() at pass boundaries,
replacing the entire dictionary each time.

**Implemented**: WeightDict caches BERT base weight chunks at setup via
`set_base_samples()` and adds delta chunks from current training state at
each periodic reindex. The base patterns form a permanent signal floor.

**Why**: A sealed/frozen dict produced correct but stale signals — it couldn't
adapt as the model learned. But a dict rebuilt from scratch on each reindex
lost the BERT baseline and degraded to ~1.3:1 ratios. The accumulating approach
keeps the 17,000x+ signal gap (zero-delta vs non-zero-delta) while allowing
the dict to evolve with training. This is critical for continuous learning:
the dict remembers both the BERT foundation AND the model's training history.

### Gate Enabled by Default

**Originally specified**: `gate_enabled: False` — Super Dict gate is opt-in.

**Implemented**: `gate_enabled: True` by default in both TrainerConfig and CLI.

**Why**: The gate creates selective training — familiar English prompts skip
the backward pass. This produces differential updates: some blocks stay at
zero-delta while others accumulate non-zero deltas from novel prompts. The
salience signal can then detect which blocks changed (ratio drops from 22,469
to ~1.3) vs stayed the same. Without the gate, all blocks train equally and
all become non-zero — no pruning signal is possible.

### nvcomp GPU Compression: Evaluated and Rejected

**Evaluated**: NVIDIA nvcomp v5.2.0 GPU compression library (Zstd, LZ4,
GDeflate, Cascaded). Installed via `pip install nvidia-nvcomp-cu12`.

**Result**: nvcomp GPU Zstd is 2.4x SLOWER than CPU zstd L1 for bf16 weight
blocks (417ms vs 173ms for 180 blocks, ~108MB total). GPU kernel launch
overhead dominates at 1.5MB block sizes. The library also doesn't support
custom dictionaries (Zstd dict training), which is essential for the
WeightDict signal.

**Decision**: Not integrated. CPU zstd L1 is the right tool for this workload.
nvcomp would only be beneficial for 100MB+ monolithic datasets where GPU
parallelism amortizes the launch overhead.

### Reindex Resets Auto-Calibration Threshold

**Added**: Each reindex resets `_salience_threshold = None` on all
ZPackRLinear layers. The next post_step recalibrates against the updated
(accumulated) WeightDict. This keeps thresholds aligned with the evolving
dictionary rather than stale from a previous calibration.

### Per-Step Reindex: Replaced with Periodic (Every ~1000 Steps)

**Originally specified**: `reindex()` called at user-defined boundaries,
potentially soft-gated by `zstd_dict_evolution: bool = True`.

**Implemented**: Periodic reindex at `reindex_interval=1000` steps. Each
call adds current delta bytes to the WeightDict (accumulating with cached
base samples). The auto-calibration threshold is also reset at each reindex.

### Performance Benchmarks

All runs on BERT-base, SST-2, batch_size=16, sm_75 (GTX 1650).  May 2026.

| Method | ms/step | VRAM peak | Accuracy | Notes |
|--------|--------:|----------:|---------:|-------|
| Standard BERT | 838ms | 1073MB | — | baseline |
| LoRA (r=8) | 640ms | 438MB | — | 0.3M params |
| PackR (v1) | **802ms** | 1142MB | 89.7% | matches standard BERT |
| PackR (v1, offload) | 815ms | 1142MB | 85.9% | +1.6% overhead |
| ZPackR (gate on) | 2060ms | 1217MB | 89.4% | 46% gate rate |
| ZPackR (novelty, pre-fuse) | 1757ms | 1176MB | 86.6% | dual-matmul forward |
| ZPackR (fused forward) | **1264ms** | 1176MB | — | single cuBLAS matmul |
| ZPackR (target) | ~900ms | 1176MB | — | +background thread |

**PackR v1 — essentially at GPU saturation**: 802ms (vs 838ms standard).  The
decode kernel in forward, fused LUT gradient in backward, and batched block
matmul in ZPackR eliminate the overhead that was present in early builds (1216ms).
The offload path is within 2% of no-offload.  Asynchrony and clone elimination
make the CPU→GPU streaming essentially free on this hardware.

**ZPackR v2 — 2.1× standard, down from 2.9×**: Post-step compression amortized to
~48ms/step at interval=4 with incremental skip (unchanged blocks reuse cached
ratios).  Forward attenuation and shrink_known_delta add negligible cost
(~1ms each).  GPU→CPU delta staging enables future overlap with forward pass.

**Optimizations applied** (May 2026):

| Optimization | Impact | Where |
|-------------|--------|-------|
| Decode kernel in backward | Eliminates int64 intermediate (~19MB/layer) | `autograd.py` |
| Fused LUT gradient kernel | Single-pass bincount+scatter_add via Triton | `kernel.py` |
| Batched block matmul | 1 `torch.bmm` replaces N serial small matmuls | `zpackr_layer.py` |
| Cached kept indices | Avoids `block_mask.nonzero()` every forward | `zpackr_layer.py` |
| Get-block-ratios fast path | Skips GPU→CPU sync when post_step data cached | `zpackr_layer.py` |
| Gate compression LRU cache | Avoids re-compressing repeated prompts | `train_harness.py` |
| JSONL buffered flush | Flushes every 10 steps instead of every step | `train_harness.py` |
| Incremental compression skip | Reuses cached ratios for unchanged blocks | `zpackr_layer.py` |
| Async delta staging | D2D snapshot enables future overlap with compute | `zpackr_layer.py` |
| Fused forward matmul | Single cuBLAS matmul replaces base+delta dual launch | `zpackr_layer.py` |
| Bulk zstd compression | multi_compress_to_buffer (C-level, GIL-released) | `zstd_dict.py` |
| Sub-sampled ratios | First 256KB per block (ratio is homogenous) | `zpackr_layer.py` |
| Delta variance gating | Skip compression for unchanged blocks (15% threshold) | `zpackr_layer.py` |
| Velvet batched sync | 1 GPU sync instead of ~200 per-param `.item()` | `velvet.py` |
| Async W_p prefetch | Offload copies on dedicated CUDA stream | `offload.py` |
| Clone elimination in evict | Avoids wasted CPU allocation per eviction | `offload.py` |

**Remaining headroom**: ZPackR's `post_step` compression (zstd of 180 blocks across
24 layers, ~192ms every 4 steps) can be moved to a background thread, overlapping
with the next forward pass on GPU.  This would bring ZPackR step time toward
~1700ms.  The GPU→CPU delta copy can be made fully async on a dedicated CUDA
stream (currently synchronous in `_sync_full_delta`), shaving another ~24ms per
post_step.  Combined, these bring ZPackR toward 1.9× standard BERT.


## The Headline

> "Two zstd dictionaries on two domains.  The static Super Dict compresses language
> into stable byte representations.  The evolving Weight Dict compresses the model's
> own weight deltas to tell you what it actually knows.  Together they make a system
> where compressibility IS knowledge — and checkpointing a different Weight Dict
> temporarily rewinds the model to a different training era."

### What PackR + ZPackR + Velvet Eliminate

| Tunable | Eliminated By | Mechanism |
|----------|:---:|-----------|
| LR schedule (warmup, decay, cosine) | Velvet | Continuous velocity-to-LR translation from optimizer `exp_avg_sq` |
| How many epochs to train | Super Dict gate | `should_train(prompt) → False` when compressibility says "already known" |
| Knowing when to stop | Super Dict + Velvet | Low ratio = keep training; Velvet drops LR as gradients flatten |
| Which layers to tune | Velvet per-group granularity | Saturated groups get `min_multiplier` (0.175×); hungry groups get `max_multiplier` (1.0×) |


## Architecture — Dual Dict, Dual Signal

ZPackR uses two zstd dictionaries on two different domains.  The signal from
each independently confirms or refutes "does the model know this already?"

```
                         Super Dict                        Weight Dict
                         ──────────                        ───────────
Domain:              Text (English + GLUE)             bf16 weight bytes (model's own patterns)
Built from:          Collegiate dictionary,             The model's _full_weight byte patterns
                     thesaurus, GLUE corpus             extracted during reindex()
Mutability:          FROZEN forever                     EVOLVES during training
                     (model learns IN this encoding)    (grows, evicts via reindex)
Role:                Compresses prompts/training input  Compresses W_f delta after each step
                     into stable byte representation    → salience gate (keep in VRAM?)
Lives at:            model.super_zstd                   model.weight_dict
Signal:              ratio_super = uncomp/comp          ratio_weight = uncomp/comp
                     "Is this text novel?"              "Is this weight shift novel?"
Enables:             Stable input encoding              VRAM management + checkpoint save/load
                                                        + reversible training eras + MoE routing
```

### The Dual Signal — Why Two Dicts Are Required

A single dict on either domain produces false negatives:

```
Text: "Brian Thomas Mulkern. My name is Brian. Brian with a B."
       →  Super Dict ratio: HIGH (common English, proper noun patterns)
       →  "This looks like familiar language"
       →  Single-dict conclusion: SKIP (wrong!)

BUT the model has never seen this person's name:
       →  Forward pass → loss → backward
       →  W_f shifts in an unfamiliar byte pattern
       →  Weight Dict compresses the delta → ratio: LOW
       →  "This specific information is NOT in the weights"
       →  Dual-dict conclusion: TRAIN (correct!)
```

The Super Dict can only check surface/text statistics.  The Weight Dict checks
the model's actual weight evolution — the truth of whether information was
already internalized.

**Decision matrix**:

| Super Dict ratio | Weight Dict ratio | Meaning | Action |
|:---:|:---:|---------|--------|
| HIGH | HIGH | Familiar text, familiar weight shift | KNOWN → skip/light training |
| HIGH | LOW  | Familiar text, novel weight shift | NEW CONTENT → full training |
| LOW  | HIGH | Unfamiliar text, familiar weight shift | (rare, possible domain drift) |
| LOW  | LOW  | Unfamiliar text, novel weight shift | NEW DOMAIN → full training |

The Brian example is row 2.  The dual dict correctly catches it.

### VRAM Invariant

ZPackR's forward allocates `output [M, out]` + `x [M, in]` + `salient_view [kept * 256, out]`.
Never allocates a full `[in, out]` matrix.  At BERT-base FFN scale (3072×768),
this eliminates ~9MB of spike per intermediate layer and ~9MB per output layer.

### Per-Step Data Flow

```
0a. Pre-forward:  Super Dict ratio(prompt) ≥ 2.0? → familiar text?
                    No → train.  Yes → flag for post-backward check.
0b. Forward:      block_accumulate(x, salient_view) → output (no VRAM spike)
1. Backward:      grad(W_f) on salient blocks only (pruned blocks get zero gradient)
2. Merge:         scatter-updated blocks → _full_weight on CPU
3. Compress:      incremental — only re-compress blocks in salient_view (bytes changed)
4. Weight check:  Weight Dict ratio per updated block
                    ratio ≥ 2.0 → weight shift was familiar → confirmed learned
                    ratio <  2.0 → weight shift was novel → must remain salient
5. Salience:      new block_mask from Weight Dict ratios
                    (combine with cached ratios for pruned blocks)
6. Regrow:        tiny Gaussian noise on pruned blocks (keeps authoritative dense)
7. Extract:       salient_view = _full_weight[active_blocks] → GPU
8. Velvet:        vel.step() — modulates LR for blocks that trained
```


## Components to Build

### 1. Super Dict — Frozen Text Codec (`super_dict.zdict`)

A zstd dictionary built offline from:
- Collegiate English dictionary word list
- English thesaurus entries
- Spell-check dictionary
- Full GLUE training corpus (SST-2, MNLI, QNLI, QQP, RTE, MRPC, CoLA, STS-B)

One-time build.  Ships as `packr/super_dict.zdict` (~100-200 KB).  Loaded at
`compress_model()` time.  Never modified during training.

```python
# packr/super_dict.py
def load_super_dict(path: str = None) -> ZstdCompressionDict:
    """Load the frozen Super Dict.  Returns zstd compression dict object."""
    ...

# Called during compress_model():
model.super_zstd = load_super_dict("packr/super_dict.zdict")
```

The Super Dict provides one method: `compress(text_bytes: bytes) -> ratio: float`.
That's it.  No mutation, no evolution, no state.

**Build script** (one-time, not in the package):
```bash
# tools/build_super_dict.py
python tools/build_super_dict.py --output packr/super_dict.zdict
```

### 2. `packr/prompt_gate.py` — Super Dict Binary Training Gate

A minimal module that converts the Super Dict ratio into a boolean train/skip signal.
This is the natural companion to Velvet's continuous LR modulation — together they
replace not just LR schedule tuning but also "how many epochs / when to stop" tuning.

```python
def should_train(
    prompt_bytes: bytes,
    super_zstd,
    threshold: float = 2.0,
) -> bool:
    """Compress prompt against frozen Super Dict.

    ratio >= threshold  →  already known (compressible) → SKIP (return False)
    ratio <  threshold  →  novel information             → TRAIN (return True)
    """
    ratio = super_zstd.ratio(prompt_bytes)
    return ratio < threshold
```

**~20 lines of code.**  Velvet reads this to zero out the multiplier for known
prompts.  Any training loop can check `should_train(prompt_bytes, model.super_zstd)`
before calling `loss.backward()`.  The Super Dict ratio IS the gate; Velvet IS the knob.

### 3. `packr/zstd_dict.py` — WeightDict

Adaptive dictionary built from weight byte patterns.  Manages salience state,
enables checkpoint save/load, and evolves via reindex().

**Ratio convention**: `ratio = len(uncompressed_bytes) / len(compressed_bytes)`.
≥1.0 always.  2.0 = 2:1 compression.  5.0 = very redundant.

**Key constraints**:
- Lossless roundtrip for bf16 weight bytes
- Max 16384 entries.  LRU eviction when full.
- Evolves at explicit `reindex()` call-sites only — never mid-step.
- Entries are 16-byte sequences (8 bf16 values).
- Pattern extraction: slide 16-byte window, stride=8.  Count frequencies.
  Discard patterns appearing in <10% of blocks.  Add top-N as new entries.

**API**:
```python
class WeightDict:
    def __init__(self, max_entries: int = 16384):
        ...

    def compress(self, weight_bytes: bytes) -> bytes: ...
    def decompress(self, zstd_bytes: bytes, shape: tuple) -> torch.Tensor: ...
    def ratio(self, weight_bytes: bytes) -> float:
        """uncompressed / compressed.  ≥1.0.  ≥2.0 = learned, <2.0 = novel."""

    def add_pattern(self, pattern: bytes): ...
    def evict_lru(self): ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str) -> "WeightDict": ...
```

### 4. `packr/salience.py` — Block Salience

```python
def compute_salience(
    weight: torch.Tensor,        # [in_features, out_features]
    weight_dict: WeightDict,
    block_size: int = 256,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Return bool[num_blocks].  True = keep in VRAM (salient).

    ratio = uncompressed / compressed.
    ratio >= threshold → learned → prune.
    ratio <  threshold → novel  → keep.
    """
    ...
```

Block size 256 matches `FusedQuantizedAdam.block_size` — aligned indexing.

### 5. `packr/zpackr_layer.py` — ZPackRLinear

Replaces `nn.Linear`.  Same interface as `PackRLinear`, different internals.

**Fields**:
```
CPU/pinned (authoritative):
    zstd_weights:     bytes               lossless compressed full matrix
    _full_weight:     torch.Tensor        cached decompressed [in, out] bf16

GPU/VRAM (dynamic):
    salient_view:     torch.Tensor        [num_kept * 256, out] bf16
    block_mask:       torch.Tensor        bool[num_blocks]
    bias:             torch.Tensor        [out] bf16 (optional)
    _acc:             torch.Tensor        [M, out] pre-allocated accumulator (reused)
    _super_cached:    float               cached Super Dict ratio for this step's prompt
```

**Forward — zero-spike block accumulation**:
```python
def forward(self, x, prompt_bytes: bytes = None):
    """Block-accumulate matmul.  Never allocates a full [in, out] matrix.

    For each salient block [start:end]:
        partial = x[:, start:end] @ salient_view[block_idx]
        output += partial

    VRAM: output [M, out] + x [M, in] + salient_view [kept * 256, out].
    """
    M = x.shape[0]
    if self._acc is None or self._acc.shape[0] != M:
        self._acc = torch.zeros(M, self.out_features,
                                dtype=torch.bfloat16, device=x.device)
    self._acc.zero_()

    kept = self.block_mask.nonzero(as_tuple=True)[0]
    for view_idx, block_idx in enumerate(kept):
        start = int(block_idx) * self.block_size
        end = min(start + self.block_size, self.in_features)
        x_blk = x[:, start:end].contiguous()
        w_blk = self.salient_view[view_idx * self.block_size:
                                  (view_idx + 1) * self.block_size]
        self._acc += x_blk @ w_blk.to(torch.bfloat16)

    out = self._acc
    if self.bias is not None:
        out = out + self.bias
    return out
```

**post_step — incremental**:
```python
def post_step(self):
    """Called after optimizer.step():

    1. Gather updated salient blocks GPU→CPU, scatter into _full_weight.
    2. Compress only newly-updated blocks against WeightDict → ratio.
       Cached ratios for unchanged blocks are reused.
    3. Regrow pruned blocks: Gaussian noise (config.zstd_regrow_noise × block_std).
    4. New block_mask: blocks with ratio >= threshold → prune; else → keep.
    5. Re-extract salient_view → GPU.
    """
    ...

def reindex(self):
    """Called at user-defined boundaries (e.g. after each pass).

    1. Extract frequent 16-byte patterns from _full_weight.
    2. Add patterns crossing min_frequency threshold as new dict entries.
    3. Evict LRU if dict is full.
    """
    ...

def save_checkpoint(self, path: str):
    """Save zstd_weights + WeightDict state + block_mask to disk."""

@classmethod
def load_checkpoint(cls, path: str, weight_dict: WeightDict) -> "ZPackRLinear":
    """Restore from checkpoint.  Recomputes salience from restored dict + weights."""
```

### 6. `packr/checkpoint.py` — Model-Level Checkpoint API

```python
def save_zpackr_checkpoint(model, path: str):
    """Save zstd_weights + WeightDict + block_mask for every ZPackRLinear layer."""

def load_zpackr_checkpoint(model, path: str):
    """Restore model to a previous training era.

    Decompresses zstd_weights, restores WeightDict state, recomputes salience.
    Blocks that were "learned" at the checkpoint's era become novel again under
    the restored dict — the model temporarily "forgets" everything after that point.

    Velvet state is re-initialized for blocks whose salience changed.
    """
```

### 7. `packr/config.py` — PackRConfig additions

```python
@dataclass
class PackRConfig:
    mode: str = "packr"            # "packr" or "zpackr"

    # --- existing PackR fields (unchanged) ---
    scheme: SchemeType = "phr"
    learnable_lut: bool = True
    layer_scope: str = "ffn"
    gradient_checkpointing: bool = True
    offload: bool = False
    block_size: int = 256

    # --- ZPackR fields (only used when mode="zpackr") ---
    zstd_super_dict_path: str = "packr/super_dict.zdict"  # REQUIRED frozen text codec
    zstd_max_entries: int = 16384
    zstd_salience_threshold: float = 2.0
    zstd_dict_evolution: bool = True
    zstd_regrow_noise: float = 1e-4
    zstd_pattern_bytes: int = 16
    zstd_min_frequency: float = 0.10
    zstd_post_step_interval: int = 1
```

### 8. `packr/layer_patcher.py` — Mode dispatch

```python
def compress_model(model, config=None):
    if config is None:
        config = PackRConfig()

    if config.mode == "packr":
        # === EXISTING CODE PATH — DO NOT MODIFY ===
        ...

    elif config.mode == "zpackr":
        from .super_dict import load_super_dict
        from .zstd_dict import WeightDict
        from .zpackr_layer import ZPackRLinear

        # Load frozen Super Dict (text codec)
        model.super_zstd = load_super_dict(config.zstd_super_dict_path)

        # Create adaptive Weight Dict (weight codec + VRAM manager)
        weight_dict = WeightDict(max_entries=config.zstd_max_entries)
        model.weight_dict = weight_dict

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not _matches_scope(name, config.layer_scope):
                continue

            zpackr = ZPackRLinear.from_linear(
                module, weight_dict, block_size=config.block_size
            )
            _replace_module(model, name, zpackr)

        if config.gradient_checkpointing:
            _enable_gradient_checkpointing(model)

    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    return model
```

### 8. `packr/__init__.py` — Exports

```python
from .zstd_dict import WeightDict
from .zpackr_layer import ZPackRLinear
from .checkpoint import save_zpackr_checkpoint, load_zpackr_checkpoint
```


## Implementation Order

| Step | File | What | Tests |
|:----:|------|------|-------|
| 1 | `tools/build_super_dict.py` | One-time: build `super_dict.zdict` from English + GLUE corpus | Produces valid zstd dict; compresses English text at >1.5 ratio |
| 2 | `packr/super_dict.py` | `load_super_dict()` — load frozen dict from disk | Returns valid zstd compression dict; same object every load |
| 3 | `packr/prompt_gate.py` | `should_train()` — Super Dict ratio → binary train/skip signal | Known prompts skipped, novel prompts flagged for training |
| 4 | `packr/zstd_dict.py` | WeightDict — compress/decompress/ratio roundtrip + save/load + reindex | Byte-identical bf16 roundtrip; ratio >2 for repeated patterns, ~1 for random; save/load preserves all entries |
| 5 | `packr/salience.py` | Block-wise ratio → bool mask, threshold=2.0 | Known → pruned; random → kept; mask shape correct |
| 6 | `packr/zpackr_layer.py` | ZPackRLinear — init, forward (block-accumulate), post_step (incremental + Weight Dict ratio check), reindex, save/load_checkpoint | Forward == nn.Linear (full salience); zero VRAM spike; incremental post_step only touches changed blocks; checkpoint roundtrip restores identical output + mask |
| 7 | `packr/checkpoint.py` | save/load_zpackr_checkpoint (model-level) | full roundtrip: forward output + block_mask identical; amnesia test: train, checkpoint, train more, load checkpoint → matches earlier state exactly |
| 8 | `packr/config.py` | Add ZPackR fields to PackRConfig | mode="packr" defaults unchanged |
| 9 | `packr/layer_patcher.py` | Mode dispatch, load Super Dict + create WeightDict | mode="packr" bit-identical to current |
| 10 | End-to-end | Build zpackr model, full step + post_step cycle, checkpoint save/load/revert, dual-signal verification | Forward output matches nn.Linear; salience evolves with training; checkpoint reverts work; high Super Dict ratio + low Weight Dict ratio = correctly triggers training |


## Integration Constraints

1. **`mode="packr"` is sacred** — zero new imports, zero new allocations, zero perf change.
2. **New dependencies** — `zstandard` (PyPI).  Lazy import, only when `mode="zpackr"`.
   ```toml
   [project.optional-dependencies]
   zpackr = ["zstandard>=0.22"]
   ```
3. **Block size 256** — aligns with `FusedQuantizedAdam.block_size`.
4. **No full-weight GPU allocation ever** — forward uses block-accumulate loop with
   pre-allocated accumulator.  VRAM ceiling: `output + x + salient_view`.
5. **Salience + post_step are CPU-bound and parallel** — per-block compression is
   independent.  Incremental: only re-compresses blocks that trained (salient_view).
6. **`reindex()` is explicit** — user calls at pass boundaries.  Scans _full_weight
   bytes, adds frequent patterns.  Milliseconds on CPU for BERT-scale layers.
7. **Optimizer state on blocks moving pruned → salient** — re-initialized (zeros).
   Block was regrown with noise — stale momentum would be incorrect.
8. **Super Dict is shipped as a file, not built at install time** — the build script
   is a one-time offline step.  The `.zdict` file is committed to the repo.


## Test Plan

### tools/test_build_super_dict.py
- `test_built_dict_compresses_english` — text from GLUE trainset compresses at ratio ≥1.5
- `test_built_dict_is_reproducible` — same inputs → same .zdict bytes

### test_zstd_dict.py
- `test_compress_decompress_bit_identical` — bf16 tensor → bytes → compress → decompress → identical
- `test_ratio_repeated_vs_random` — repeated patterns ratio >2, random ~1
- `test_lru_eviction` — fill dict, add new → oldest gone
- `test_save_load_preserves_all_entries` — disk roundtrip
- `test_frequency_filter` — 2 of 100 blocks → skipped; 15 of 100 → added
- `test_reindex_adds_new_patterns` — train model, reindex → new entries appear

### test_salience.py
- `test_random_weights_all_salient` — all blocks kept (ratio ~1)
- `test_repeating_pattern_mostly_pruned` — most blocks pruned (ratio >>2)
- `test_mask_cardinality` — mask length = ceil(in_features / 256)

### test_zpackr_layer.py
- `test_forward_matches_nn_linear_full_salience` — identical weights → identical output
- `test_forward_no_full_matrix_allocation` — no [in, out] tensor allocated
- `test_post_step_incremental_only_touches_changed_blocks`
- `test_post_step_weight_dict_ratio_update` — after step on novel data, Weight Dict ratio < threshold on updated blocks
- `test_full_post_step_cycle` — init → train → post_step → weights updated, zstd_weights updated, salience refreshed
- `test_checkpoint_roundtrip` — save → load → identical output + identical block_mask
- `test_checkpoint_amnesia` — train, save, train more, load → matches original state exactly

### test_dual_signal.py
- `test_super_dict_high_but_weight_dict_low` — familiar text (Super Dict ratio ≥2) that produces unfamiliar weight deltas (Weight Dict ratio <2) → posts step correctly marks blocks as novel

### test_mode_packr_unchanged.py
- `test_packr_mode_identical_weights`
- `test_packr_mode_no_zstd_import`
- `test_packr_mode_forward_bit_identical`

### test_checkpoint.py
- `test_model_checkpoint_roundtrip` — save/load model, verify all layers match
- `test_model_checkpoint_amnesia` — full training → checkpoint → more training → revert → matches


## Quick Start

```bash
git clone https://github.com/otherdrums/packr.git
cd packr
pip install -e .
pip install "packr[zpackr]"
```

```python
from transformers import AutoModelForSequenceClassification
from packr import compress_model, PackRConfig, FusedQuantizedAdam, VelvetController
from packr.checkpoint import save_zpackr_checkpoint, load_zpackr_checkpoint

# PackR mode (existing, unchanged)
config = PackRConfig(mode="packr")
model = compress_model(model, config)

# ZPackR mode
config = PackRConfig(mode="zpackr")
model = compress_model(model, config)
# model.super_zstd   ← frozen Super Dict (text codec)
# model.weight_dict  ← adaptive Weight Dict (VRAM manager + checkpoint state)

optimizer = FusedQuantizedAdam(model.parameters(), lr=2e-5)
vel = VelvetController(optimizer)

for epoch in range(3):
    for batch in loader:
        # Super Dict compresses the prompt → ratio_prompt
        # (StreamCC/AGPL handles the prompt-level train/skip decision)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        vel.step()
        optimizer.zero_grad()

        # Weight Dict post_step: compress weight deltas, update salience
        for layer in model.modules():
            if hasattr(layer, 'post_step'):
                layer.post_step()

    # Reindex at epoch boundary — Weight Dict evolves
    for layer in model.modules():
        if hasattr(layer, 'reindex'):
            layer.reindex()

    # Checkpoint captures Weight Dict state + zstd_weights
    save_zpackr_checkpoint(model, f"zpackr_epoch_{epoch}.pt")

# Revert to epoch 0 — weights, salience, Weight Dict all rewound
load_zpackr_checkpoint(model, "zpackr_epoch_0.pt")
```


## Deliverables

1. `tools/build_super_dict.py` — one-time offline Super Dict builder
2. `packr/super_dict.zdict` — frozen text codec, committed to repo
3. `packr/super_dict.py` — `load_super_dict()` loader
4. `packr/prompt_gate.py` — `should_train()` — Super Dict ratio → binary train/skip gate
5. `packr/zstd_dict.py` — `WeightDict` (adaptive, save/load, reindex)
6. `packr/salience.py` — `compute_salience()`
7. `packr/zpackr_layer.py` — `ZPackRLinear` (zero-spike forward, incremental post_step with Weight Dict ratio check, reindex, save/load_checkpoint)
8. `packr/checkpoint.py` — model-level save/load helpers
9. `packr/config.py` — ZPackR fields added (existing unchanged)
10. `packr/layer_patcher.py` — mode dispatch, Super Dict + Weight Dict wiring
11. `packr/__init__.py` — export additions
12. `tests/test_build_super_dict.py`
13. `tests/test_prompt_gate.py`
14. `tests/test_zstd_dict.py`
15. `tests/test_salience.py`
16. `tests/test_zpackr_layer.py`
17. `tests/test_dual_signal.py`
18. `tests/test_checkpoint.py`
19. `tests/test_mode_packr_unchanged.py`
20. `tests/test_vram_regression.py` — VRAM budget asserts for packr mode

### May 2026 Performance Optimizations Delivered

| File | Change |
|------|--------|
| `packr/kernel.py` | `decode_packed()` public API; `fused_lut_gradient()` Triton kernel + PyTorch fallback |
| `packr/autograd.py` | Backward uses `decode_packed()` instead of `lut[W_p.long()]`; uses `fused_lut_gradient()` instead of two-pass bincount+scatter_add |
| `packr/zpackr_layer.py` | Batched block matmul (`torch.bmm`); cached `_kept_indices`; guarded `.to(device)`; `del x` after bf16 cast; `_block_accumulate()` helper |
| `packr/offload.py` | Async W_p prefetch on dedicated stream; eliminated `.clone()` in `evict_wp` and `register_wp` |
| `packr/velvet.py` | Batched GPU sync: one `torch.cuda.synchronize()` per step instead of per-param `.item()` |
| `tests/test_vram_regression.py` | 4 VRAM budget tests for packr mode (forward, full-step, no-int64-temp, offload) |


## Deferred to v2.1+

| Feature | Rationale |
|---------|-----------|
| Mini-dict MoE ensemble (Weight Dict splits into task specialists) | Builds on v2.0 checkpoint + evolving dict foundation; needs real multi-task training data to calibrate routing stability + expert collapse prevention |
| Ratio-as-primary-gate + Velvet-as-LR-modulator within weight blocks | Needs real training runs to calibrate threshold vs Velvet interaction; v2.0 uses Velvet velocity + small ratio bias to augment LR modulation |
| Weight Dict-driven expert routing for multi-task continual learning | Natural extension of checkpoint magic — switch Weight Dict = switch expert era; needs multi-task GLUE benchmark data |

### Per-Block Novelty System (May 2026)

**Theory**: The WeightDict compression ratio per block is a pre-backward signal of the
block's semantic novelty.  A high ratio (compressible delta = block matches known
patterns) means the block has learned this content — training on this block produces
redundancy, not learning.  A low ratio (novel delta) means the delta patterns are
unfamiliar to the WeightDict — genuine learning is happening.

The system has two mechanisms operating in tandem:

1. **Forward attenuation**: each block's delta contribution to the output is scaled
   by its novelty score.  A known block (novelty ≈ 0) contributes almost nothing to
   the forward pass — its information is "already in the base model."  The chain
   rule propagates attenuated gradients back, so known blocks see near-zero gradient
   regardless of the optimizer's LR.

2. **Post-optimizer decay**: after `optimizer.step()` + `zero_grad()`, known blocks
   actively shrink toward zero delta.  The decay rate is driven by `(1 - novelty)`,
   so fully known blocks decay at `_gap_decay_rate` per step (default 5%), while
   fully novel blocks experience no decay.

Together: known blocks contribute less to the model output AND actively forget.
Novel blocks train normally with whatever LR the optimizer uses.  There is no
"learning rate" for known blocks — they simply regress toward the base model.

This replaces Velvet's reactive EMA-of-gradient-velocity with a pre-backward,
domain-separated, per-block signal.  The WeightDict (compression context) separates
signal from noise; the Super Dict (text context) cleans the input.  The novelty
score is continuous [0,1], not binary — "kinda known" content gets kinda
attenuated and kinda decayed.

**Implementation** (`zpackr_layer.py`):

| Component | Purpose |
|-----------|---------|
| `_block_gaps` | Per-block weight_ratio list, computed in `post_step()` from compressed delta bytes |
| `_compute_novelty()` | Maps gaps → novelty [0,1] using historical range normalization |
| `_gap_hist_max/min` | Tracks historical ratio extremes for stable normalization (prevents oscillation when gaps are uniform) |
| `shrink_known_delta()` | Decays known blocks toward zero: `delta *= 1 - (1-novelty) * decay_rate` |
| `_block_accumulate()` | Scales each block's delta contribution by novelty in the forward matmul |
| `_gap_enabled` | bool toggle per layer (default True) |
| `_gap_decay_rate` | Max decay per step for fully known blocks (default 0.05 = 5%) |

**Forward attenuation formula** (in `_block_accumulate`):
```python
result = torch.bmm(x_stacked, delta_stacked)         # [K, M, N]
result = result * novelty_tensor.view(K, 1, 1)       # scale per block
```

**Novelty formula** (`_compute_novelty`):
```python
self._gap_hist_max = max(self._gap_hist_max, max(gaps))
self._gap_hist_min = min(self._gap_hist_min, min(gaps))
span = max(self._gap_hist_max - self._gap_hist_min, 0.1)
novelty = clamp((gap_hist_max - gap) / span, 0.0, 1.0)
```
Historical range normalization: at the start of training, gaps are high (~6 for
zero delta), so novelty ≈ 0 for all blocks (everything looks "known" — correct,
since nothing has been learned yet).  As training produces non-zero deltas, gaps
drop to ~1.3, the historical range expands, and novelty rises toward 1.0.  When
all blocks are in the same state (uniform training, no gate differentiation),
novelty is uniform (~0.08 at step 200) — the system correctly identifies that
there's no bimodal separation and applies conservative, uniform mild decay.

**Decay formula** (`shrink_known_delta`):
```python
decay = (1.0 - novelty) * _gap_decay_rate
delta_salient[block_rows] *= (1.0 - decay)
```
At step 200 with uniform training: novelty ≈ 0.09, decay ≈ 0.91 × 0.05 = 0.0455/step.
Each block shrinks ~4.5% per step, balanced by optimizer gradient from forward pass
(which is also novelty-attenuated).  The net effect is that fully known blocks
relax toward zero, while novel blocks' behavior is dominated by the optimizer.

**Wired into**: `train_harness.py` and `diagnose.py` — `shrink_known_delta()`
called after `optimizer.step()` + `zero_grad()`.  Forward attenuation runs
inside `_block_accumulate()` automatically on every forward pass.  Falls back
to no-op when gaps not yet computed (first few steps before first `post_step`).

**Diagnostic tools**:

| Tool | Purpose |
|------|---------|
| `tools/diagnose.py` | Logs per-block `novelty` scores in `ratio_log.jsonl` alongside `ratio`, `gap`, and `delta_l2` |
| `tools/calibrate.py --sweep-decay` | Sweeps `decay_rate` values against recorded gap data, measures known-vs-novel separation |
| `tools/calibrate.py` | Standard threshold confusion-matrix report for salience pruning |

**Current findings** (BERT-base on SST-2, 200 steps, no gate):

| Metric | Value |
|--------|-------|
| Accuracy | 87.7% (novelty on) vs 89.4% (gate-on baseline) |
| ms/step | 2459ms (novelty on) vs 2060ms (gate on) |
| VRAM peak | 1176 MB |
| Novelty at step 200 | ~0.09 uniform across all blocks |
| Delta L2 at step 200 | 0.062 (slowly decaying from 0.065 at step 100) |

The novelty is uniform (~0.09) because without gate-driven differential training,
all blocks undergo the same trajectory.  The system correctly applies conservative
uniform mild decay rather than catastrophically pruning active blocks.  The decay
is visible — delta_l2 drops from 0.065 (step 100) to 0.062 (step 200) — but the
optimizer gradient from the forward pass counterbalances it, maintaining training
progress.

Next steps for the system:
- Run with gate enabled at threshold=1.0 to create bimodal gaps, where zero-delta
  blocks coexist with trained blocks → novelty scores diversify
- Calibrate `decay_rate` against bimodal data to find the value that eliminates
  known-block contributions without touching novel blocks
- Test on multi-task or longer training to observe natural gap divergence
- Wire gap-derived multiplier as a soft replacement for learning rate entirely
  (remove `optimizer.param_groups[].lr` variable)

**File changes**:
| File | Change |
|------|--------|
| `packr/zpackr_layer.py` | `_block_gaps`, `_compute_novelty()`, `_gap_hist_max/min`, `shrink_known_delta()`, `_gap_enabled`, `_gap_decay_rate`, forward attenuation in `_block_accumulate()`, reset on reindex |
| `tools/train_harness.py` | `shrink_known_delta()` call after `optimizer.step()` + `zero_grad()` |
| `tools/diagnose.py` | `shrink_known_delta()` wiring, per-block `novelty` field in ratio log |
| `tools/calibrate.py` | `--sweep-decay` flag, `sweep_decay_rates()`, `print_decay_report()` |

