# ZPackR — System Flowchart (v8, BERT-base SST-2)

## 1. Training Loop Overview

```
                        ┌──────────────────────────────────────────┐
                        │             TRAINING LOOP                 │
                        │  batch=16, seq=128, bf16 model           │
 Input text ───────────▶│  HuggingFace BERT forward                 │
 (tokenized)            │     │                                     │
                        │     ▼                                     │
                        │  ZPackRLinear.forward()                   │
                        │  x @ (base_W + delta × (1 - atten_byte))  │
                        │  (per-row attenuation, dtype-agnostic)     │
                        │     │                                     │
                        │     ▼                                     │
                        │  loss.backward()                           │
                        │     │                                     │
                        │     ▼                                     │
                        │  convergence gate (threshold = 1.0)        │
                        │  checks _atten_byte.min() / 255            │
                        │     │                                     │
                        │     ├── True (all rows byte=255) → skip   │
                        │     └── False → optimizer.step()          │
                        │                  │                        │
                        │                  ▼                         │
                        │  CUDA8BitAdam | FusedQuantizedAdam        │
                        │  8-bit int8 m/v, per-block scales         │
                        │     │                                     │
                        │     ▼                                     │
                        │  compute_hash_gpu() — every step or       │
                        │  every N steps (hash_interval config)     │
                        │     │                                     │
                        │     ├── fused Triton hash kernel          │
                        │     │    1D grid (in_features,), K=16     │
                        │     │    sign(delta × projection)         │
                        │     │                                     │
                        │     ├── compute_attenuation               │
                        │     │    CPU pinned → GPU batch transfer  │
                        │     │    continuous byte comparison        │
                        │     │    offsets (1,3,10,30,100,300,1000) │
                        │     │    attn = (mean_sim × (1-flat))²    │
                        │     │                                     │
                        │     └── push() → CPU pinned ring buffer   │
                        │         4200 entries, async copy          │
                        │                                          │
                        └──────────────────────────────────────────┘
                                       │
                                       ▼
                                Predictions
```

## 2. Per-Step Detail (BERT-base, ~110M FFN params, 24 ZPackRLinear)

```
STEP (hash-interval=8, non-hash step shown):
═══════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────┐
  │ FORWARD (≈290ms GPU)                                         │
  │                                                              │
  │  x shape:          [16, 128, 768] bf16                       │
  │  For each layer (×24):                                       │
  │    nv = _atten_byte.float() / 255 → bf16 [in_f, 1]          │
  │    W = base_W + delta_salient × (1 - nv)  ← dtype-agnostic  │
  │    out = x @ W → then .to(orig_dtype)                        │
  │  → base_W frozen (bf16), delta_salient trainable (bf16)     │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ CONVERGENCE GATE (≈0ms, CUDA sync = forward cost)            │
  │                                                              │
  │  should_skip_backward():                                     │
  │    for each ZPackRLinear (24):                               │
  │      if _atten_byte.float().min().item() / 255 < 1.0:       │
  │        return False   ← any row not fully converged          │
  │    return True        ← ALL rows byte=255                    │
  │                                                              │
  │  Gate is currently 0% — no row has hit byte=255 yet          │
  └──────────────────────────────────────────────────────────────┘
                              │ (if not skipped)
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ BACKWARD (≈250ms GPU)                                        │
  │                                                              │
  │  loss.backward()                                             │
  │  → gradient flows through delta × (1 - nv)                  │
  │  → base_W gets 0 gradient (frozen)                           │
  │  → delta_salient gets full gradient                          │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ OPTIMIZER (≈129ms, includes zero_grad)                       │
  │                                                              │
  │  CUDA8BitAdam: 76 per-param kernel launches                  │
  │    dtype-agnostic (bf16/fp32 via register shift)             │
  │    warp-level __shfl_xor_sync reductions                     │
  │    int8 m/v states, per-block float32 scales                  │
  │  OR FusedQuantizedAdam (Triton 8-bit)                        │
  │  OR standard AdamW (fp32 states)                             │
  │                                                              │
  │  Velvet (optional): read exp_avg_sq, adjust LR               │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ HASH (≈619ms raw, ≈87ms net — overlaps with next step)       │
  │  (runs every hash_interval steps, not every step)             │
  │                                                              │
  │  1. fused Triton kernel (1D grid):                           │
  │       grid (in_features,)  — one block per row               │
  │       each block: load delta[row] ONCE                       │
  │       compute K=16 dot products, store K hash bits            │
  │       16× less delta memory traffic than 2D grid             │
  │                                                              │
  │  2. compute_attenuation:                                     │
  │       batch-transfer 7 past hashes from CPU pinned → GPU     │
  │       continuous byte: 1 - |hash - stored| / 255              │
  │       matching = byte_sim.mean(dim=1) → cos_sim = 2m - 1     │
  │       mean_sim, variance across offsets                       │
  │       atten = clamp((mean_sim × (1 - sqrt(var)))², 0, 1)     │
  │       store as uint8 via in-place copy_()                    │
  │                                                              │
  │  3. push: async non_blocking GPU→CPU pinned copy              │
  │                                                              │
  │  Skip logic: when hash_counter < hash_interval, return        │
  └──────────────────────────────────────────────────────────────┘
```

## 3. LSH Hash Chain (fused Triton kernel)

```
DELTA SALIENT [in_features × out_features] bf16
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ FUSED TRITON KERNEL: _lsh_hash_fused_kernel                 │
│                                                             │
│ Grid: (in_features,) — 1D, one block per row               │
│ Each block:                                                 │
│   acc = tl.zeros([K], dtype=tl.float32)                     │
│   for each chunk (BLOCK_OUT=256):                           │
│     load delta[row, chunk] — ONCE                           │
│     for k in range(K):                                      │
│       load proj[k, chunk]                                   │
│       acc[k] += tl.sum(delta × proj)                        │
│                                                             │
│   tl.store(hash_ptr + row*K + tl.arange(0,K),              │
│            (acc > 0).to(tl.uint8))                          │
│                                                             │
│ PACK: [in_f, K] uint8 → [in_f, K//8] uint8                │
│       K=16 → 2 bytes/row                                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ WINDOW: pinned CPU ring buffer                              │
│  size: 4200 × total_rows × (K/8) bytes = ~0MB GPU, 369MB CPU│
│  push: async non_blocking copy (GPU→CPU pinned)             │
│  compute_attenuation: batch-transfer 7 offsets CPU→GPU      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ MULTI-SCALE COMPARISON                                     │
│  offsets: (1, 3, 10, 30, 100, 300, 1000)                   │
│                                                             │
│  current = current_hashes.unsqueeze(0)  [1, rows, 2]        │
│  stored = past_hashes                [n_off, rows, 2]        │
│  diff = (current - stored).abs()                             │
│  byte_sim = 1 - diff / 255                                   │
│  matching = byte_sim.mean(dim=2)   [n_off, rows]             │
│  cos_sim = 2 × matching - 1        [n_off, rows]             │
│                                                             │
│  mean_sim = cos_sim.mean(dim=0)    [rows]                    │
│  variance = cos_sim.var(dim=0)     [rows]                    │
│  flatness = sqrt(clamp(var, 0))    [rows]                    │
│  attenuation = clamp((mean_sim × (1 - flatness))², 0, 1)    │
│  → _atten_byte = (atten × 255).to(uint8)                    │
└─────────────────────────────────────────────────────────────┘
```

## 4. Row State Machine

Each row (of delta_salient) goes through these states:

```
                ┌─────────────────────┐
                │      NOVEL          │ delta = 0
                │      ROW            │ LSH hash = f(zeros)
                │                     │ attenuation = 0 (no window)
                │   GPU delta ✓       │
                └─────────┬───────────┘
                          │ optimizer.step() → delta ≠ 0
                          ▼
                ┌─────────────────────┐
                │     LEARNING        │ delta changes direction
                │      ROW            │ LSH changes each step
                │                     │ mean_sim < 0.9
                │   GPU delta ✓       │ flatness > 0.05
                └─────────┬───────────┘
                          │ delta direction stabilizes
                          ▼
                ┌─────────────────────┐
                │    CONVERGING       │ delta stable
                │      ROW            │ LSH changes slowly
                │                     │ mean_sim 0.9-0.99
                │   GPU delta ✓       │ flatness 0.01-0.05
                └─────────┬───────────┘
                          │ delta very stable → atten → 1.0
                          ▼
                ┌─────────────────────┐
                │    CONVERGED        │ delta stable
                │      ROW            │ LSH unchanged
                │                     │ _atten_byte = 255
                │   GPU delta ✓       │ flatness ≈ 0
                └─────────┬───────────┘
                          │ new gradient changes delta
                          ▼
                ┌─────────────────────┐
                │     LEARNING        │ (cycle repeats)
                │      ROW            │
                └─────────────────────┘
```

## 5. VRAM Breakdown (BERT-base, batch=16, seq=128, bf16, CUDA optimizer)

```
┌──────────────────────────────────────────────┬───────────┐
│ Component                                    │ Size      │
├──────────────────────────────────────────────┼───────────┤
│ Model weights (bf16)                         │  220 MB   │
│   base_W (FFN, frozen, bf16)                 │  113 MB   │
│   delta_salient (FFN, trainable, bf16)       │  113 MB   │
│   attention / embeddings / pooler (bf16)     │  108 MB   │
├──────────────────────────────────────────────┼───────────┤
│ Optimizer states (int8 m/v + fp32 scales)    │  223 MB   │
│ Gradients (bf16, freed after step)           │  220 MB   │
│ Pinned CPU window                            │    0 MB   │
├──────────────────────────────────────────────┼───────────┤
│ torch.cuda.memory_allocated (steady)         │  565 MB   │
│ torch.cuda.max_memory_allocated (peak)        │  861 MB   │
│ nvidia-smi (includes caching allocator)      │ ~1.5 GB   │
└──────────────────────────────────────────────┴───────────┘
```

## 6. Determinism Chain

```
Every component is a pure function of current state:

seed (42) → torch.Generator → random projections [K, out_features]
    → normalized per row → cached GPU tensor (class-level)

delta_salient (GPU bf16) → fused Triton kernel → hash bits [in_f, K]
    → pack to [in_f, K//8] uint8

CPU pinned window: 4200 × hashes

multi-scale comparison → mean_sim × flatness → attenuation²
    → uint8 via (attn × 255).to(torch.uint8)
    → in-place copy_() into register_buffer

Gate: _atten_byte.float().min().item() / 255 ≥ 1.0

Checkpoint: zstd lossless roundtrip of delta bytes
```

## 7. Checkpoint — Save/Restore

```
SAVE:
  for each ZPackRLinear layer:
    delta_cpu = delta_salient.cpu()
    zstd.compress(delta_cpu.view(uint8).numpy().tobytes())
    → .zstd file
    torch.save(base_W.data) → .base_W file
    torch.save(metadata) → .meta file

RESTORE:
  zstd.decompress(.zstd file)
  → torch.frombuffer(...).view(bfloat16).view(in_f, out_f)
  → delta_salient = nn.Parameter(restored)
  load base_W from .base_W file
  create fresh DeltaSignatureDB (empty window)
```

## 8. Optimizer Dispatch (dtype-agnostic)

```
CUDA8BitAdam.step():
  for each param p:
    if p.dtype == torch.bfloat16:
      is_bf16 = 1  (kernel shifts uint16 left by 16)
    else:
      is_bf16 = 0  (kernel loads float directly)

    CUDA kernel: bf16_to_float via __uint_as_float(p_u << 16)
                 compute AdamW with int8 dequant/requant
                 float_to_bf16 via __float_as_uint(pn) >> 16

  Prebuilt: packr._adam_8bit_cuda (from setup.py CUDAExtension)
  Fallback: torch.utils.cpp_extension.load_inline (JIT nvcc)
```
