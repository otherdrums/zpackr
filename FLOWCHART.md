# ZPackR — Block-Level System Flowchart (v4 LSH)

## 1. System Overview

```
                      ┌─────────────────────────────────────────┐
                      │              TRAINING LOOP               │
                      │                                         │
 Input Text ──────────▶  HuggingFace BERT forward               │
 (tokenized)           │      │                                  │
                       │      ▼                                  │
                       │  ZPackRLinear.forward()                 │
                       │  x @ (base_W + delta*(1-attenuation))   │
                       │      │                                  │
                       │      ▼                                  │
                       │  loss.backward()                        │
                       │      │                                  │
                       │      ▼                                  │
                       │  optimizer.step()  ← FusedQuantizedAdam │
                       │      │                                  │
                       │      ├─ compute_hash_gpu()  every step  │
                       │      │    LSH hash → sliding window     │
                       │      │    multi-scale cos_sim → attn    │
                       │      │                                  │
                       │      ├─ convergence gate (threshold 0.99)│
                       │      │    all blocks ≥ 0.99? → skip     │
                       │      │                                  │
                       │      └─ optimizer.zero_grad()           │
                       └─────────────────────────────────────────┘
                                      │
                                      ▼
                               Predictions
```

## 2. Per-Step Detail

```
STEP N (every step):
═══════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────┐
  │ 1. FORWARD                                                   │
  │                                                              │
  │   x = batch.to(device)          # [M, in_features]           │
  │                                                              │
  │   For each ZPackRLinear layer:                               │
  │     if all blocks salient:                                   │
  │       nv = tensor(attenuation_factors).repeat_interleave(256)│
  │       W = base_W + delta_salient * (1.0 - nv)                │
  │       out = x @ W                                            │
  │     elif partial salience:                                   │
  │       W = base_W.clone()                                     │
  │       scatter attenuated delta into W via index_add_         │
  │       out = x @ W                                            │
  │                                                              │
  │   loss = outputs.loss / grad_accum_steps                     │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 2. CONVERGENCE GATE                                          │
  │                                                              │
  │   if all(layer._attenuation_factors >= 0.99                  │
  │           for layer in zpl_layers):                          │
  │       gate_skipped = True                                    │
  │       → skip backward, record step                           │
  │   else:                                                      │
  │       gate_skipped = False                                   │
  │                                                              │
  │   Hash is computed EVERY step (even on gate-skipped)         │
  └──────────────────────────────────────────────────────────────┘
                              │ (if not skipped)
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 3. BACKWARD                                                  │
  │                                                              │
  │   loss.backward()                                            │
  │   → gradients flow through attenuated delta                  │
  │   → fully-attenuated blocks get ~0 gradient                  │
  │   → base_W gets 0 gradient (frozen)                          │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 4. OPTIMIZER + HASH                                          │
  │                                                              │
  │   optimizer.step()                                           │
  │   → FusedQuantizedAdam updates delta_salient on GPU          │
  │   → optimizer.zero_grad()                                    │
  │                                                              │
  │   compute_hash_gpu() — every step, ~1ms total on GPU:        │
  │     padded = F.pad(delta_salient, ...)                       │
  │     blocks = padded.reshape(n_blocks, block_elements)        │
  │     proj = blocks.float() @ projections_gpu.t()              │
  │     hash = (proj > 0).to(torch.uint8)                        │
  │     push hash → sliding window                               │
  │     attenuation = compute_attenuation(hash)                   │
  │     → attenuation = mean_sim * (1 - flatness)                │
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. Signal Flow — LSH Hash Chain (every step, GPU)

```
DELTA SALIENT on GPU [in_features × out_features] bf16
│
│  F.pad + reshape → [num_blocks, block_size × out_features]
│
├── Block 0: [block_size × out_features] bf16 → flat
├── Block 1: [block_size × out_features] bf16 → flat
├── ...
└── Block N-1: [block_size × out_features] bf16 → flat

                    │
                    ▼

  blocks_flat.float() @ projections_gpu.t()   # GPU matmul
  → [num_blocks, K] float32
  → sign → [num_blocks, K] uint8 hash

  projections_gpu: two shared matrices (120MB total):
    - intermediate: [64, 256×3072] bf16  (96MB)
    - output:       [64, 256×768]  bf16  (24MB)

                    │
                    ▼

  For offsets o in (1, 5, 10, 25, 50):
    if o ≤ len(window):
      stored = window[-o]              # hash from o steps ago
      matching = (hash == stored).mean(dim=1)
      cos_sim = 2 × matching - 1

  mean_sim = mean(cos_sim across offsets)
  variance = var(cos_sim across offsets)
  flatness = sqrt(variance)
  attenuation = mean_sim × (1 - flatness)

                    │
                    ▼

  FORWARD:  combined delta contribution = delta[i] × (1 - attenuation[i])
            → applied per block in the cuBLAS matmul

  GATE:     if all(attenuation[i] >= 0.99 for all i in all layers)
            → should_skip_backward() = True
```

---

## 4. Block State Machine

A single block (256 × out_features) goes through these states:

```
                   ┌────────────────┐
                   │    FRESH       │ delta = 0
                   │    BLOCK       │ LSH hash = f(zeros)
                   │                │ attenuation = 0.0 (no window)
                   │   on GPU ✓     │
                   └───────┬────────┘
                           │ optimizer.step()
                           │ delta becomes non-zero
                           ▼
                   ┌────────────────┐
                   │   LEARNING     │ delta ≠ 0, direction changes
                   │    BLOCK       │ LSH changes each step
                   │                │ mean_sim < 0.9
                   │   on GPU ✓     │ flatness > 0.05
                   └───────┬────────┘
                           │ delta direction stabilizes
                           ▼
                   ┌────────────────┐
                   │  CONVERGING    │ delta ≠ 0, direction stable
                   │    BLOCK       │ LSH changes slowly
                   │                │ mean_sim 0.9-0.99
                   │   on GPU ✓     │ flatness 0.01-0.05
                   └───────┬────────┘
                           │ delta very stable
                           ▼
                   ┌────────────────┐
                   │  CONVERGED     │ delta stable
                   │    BLOCK       │ LSH unchanged across window
                   │                │ mean_sim ≈ 1.0
                   │   on GPU ✓     │ flatness ≈ 0.0
                   └───────┬────────┘
                           │
                           │ new prompt changes delta
                           ▼
                   ┌────────────────┐
                   │   LEARNING     │ (cycle repeats)
                   │    BLOCK       │
                   └────────────────┘
```

---

## 5. Determinism Chain

```
Every component is a pure function:

seed (42) → torch.Generator → random projections [K, block_elements]
                                              │
                                              ▼
delta_salient (GPU) → F.pad → reshape → blocks_flat
                                              │
                                              ▼
            blocks_flat @ projections.T → sign → hash (uint8)
                                              │
                                              ▼
    hash vs window[-1], window[-5], ..., window[-50]
    → cos_sim per offset → mean_sim, flatness → attenuation
                                              │
                                              ▼
                Forward: delta × (1 - attenuation)
```

---

## 6. Convergence Gate Decision

```
Every training step, before backward:

  For each ZPackRLinear layer (0..23):
      factors = layer._attenuation_factors
      if factors is None:
          return False  ← no factors yet → keep training

      if any(f < 0.99 for f in factors):
          return False  ← at least one block still novel

  return True  ← ALL blocks across ALL layers fully attenuated

                   │
                   ├── True  → skip backward (gate_skipped = True)
                   └── False → loss.backward() + optimizer.step()

  Hash is still computed on gate-skipped steps (window evolves).
  If a block dips below 0.99, gate opens on next step.
```

---

## 7. Checkpoint — Save/Restore

```
SAVE:
  for each ZPackRLinear layer:
      delta_cpu = delta_salient.cpu()             # GPU→CPU copy
      zstd.compress(delta_cpu bytes) → .zstd file
      torch.save(base_W) → .base_W file
      torch.save(block_mask) → .mask file
      torch.save(metadata) → .meta file

RESTORE:
  zstd.decompress(.zstd file) → full_delta bf16
  load base_W, block_mask
  rebuild delta_salient from kept blocks
  initialize fresh DeltaSignatureDB (empty window)
```
