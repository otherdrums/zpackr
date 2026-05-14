# ZPackR Progress Notes — May 2026

## Architecture Evolution

| Version | Compressor | Signal | State | Status |
|---------|-----------|--------|-------|--------|
| v1 (initial) | zstd + WeightDict | Dictionary-learned delta patterns | Reindex every 1000 steps, auto-calibrating thresholds, novelty tracking, shrinkage | **Superseded** |
| v2 (LZ4, May 13) | LZ4 per-block | Stateless byte compression | Fixed attenuation constants, no dictionary | **Superseded** — LZ4 gives ratios < 1.0 (data inflation) |
| v3 (zstd, May 14) | zstd per-block | Deterministic: `clamp((ratio-1.0)/7.0, 0, 1)` | No dictionary, no reindex, no calibration, no state | **Current** |

## What We Learned

### 1. Super Dict gate is useless for training signal
- zstd ratio on text depends on word length/spelling, not model knowledge
- Gate creates differential training (zero-delta blocks) — useful for bimodality
- Replaced with convergence gate: skip when all blocks fully attenuated

### 2. LZ4 can't do the job
- Ratios < 1.0 on non-zero bf16 (inflation)
- Zero creep, zero noise — binary signal only (zero vs non-zero)
- No continuous "how known is this block" signal

### 3. zstd gives clean, creeping signal
- Non-zero delta ratio: ~1.27 floor
- Zero delta ratio: 13,000+
- Creep rate: 0.001-0.009%/step (deeper layers faster)
- Noise floor: effectively zero (deterministic for same bytes)
- Total creep over 300 steps on fixed batch: 0.18% (L0) to 1.37% (L11)
- Signal characterization run: `tools/signal_char.py`

### 4. The delta IS the history
- No need for prev_ratio tracking, EMA, or creep computation
- The delta's current zstd compressibility already encodes everything
- Formula: `attenuation = clamp((ratio - 1.0) / 7.0, 0, 1)` — pure function of current ratio

## Test Data (SST-2, no gate, no velvet)

| Run | Steps | Peak Acc | Notes |
|-----|-------|---------|-------|
| `no_gate_calibration` | 2000 | 92.5% | Phase A data capture, zstd+WeightDict, post_step_interval=1 |
| `epoch2_full` (user) | 8000 | 94.06% | Original v1 run, gate on |
| `zstd_attenuation_sst2` | 8000 | pending | v3 current, running |

## KNOWN BUGS / ISSUES

- Convergence gate never fires in practice because ALL blocks train uniformly (no gate to create zero-delta blocks).  Gate is functional but won't activate until we add differential training.
- `post_step` with zstd is slower than LZ4 (~1-3ms/block vs ~0.003ms).  Acceptable with post_step_interval=4.

## NOT TO RETRY

- LZ4 for the ratio signal (ratios < 1.0, useless)
- WeightDict + reindex (complexity without signal improvement)
- Calibration multiplier / auto-thresholds (dead code, never wired)
- shrink_known_delta / novelty tracking (post-hoc, fights forward attenuation)
- Relative creep tracking (EMA, prev_ratio) — the delta IS the history
- Super Dict as gate (word-length dependent, not model-knowledge dependent)
