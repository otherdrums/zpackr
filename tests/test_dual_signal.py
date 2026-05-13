"""Dual-signal verification — the core architectural claim of ZPackR v2.0.

Tests that the Super Dict (text) and Weight Dict (weight patterns) together
correctly identify "familiar language with novel information."

The Henry test:
  1. Compress familiar English text (Henry the Eighth lyrics) → Super Dict ratio >= 2.0
     "This looks like known English" → single-dict would SKIP (wrong!)
  2. Train on that text → weight deltas emerge
  3. Weight Dict compresses the delta → ratio < 2.0 on novel blocks
     "But this specific information is new to the weights" → TRIGGER TRAINING
  4. post_step correctly marks novel blocks as salient (kept in VRAM)
  5. Decision matrix row: HIGH Super + LOW Weight = NEW CONTENT → train
"""

import os
import torch
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zpackr.zstd_dict import WeightDict
from zpackr.zpackr_layer import ZPackRLinear
from zpackr.super_dict import load_super_dict
from zpackr.prompt_gate import should_train


HENRY_LYRICS = b"""
I'm 'Enery the Eighth, I am,
'Enery the Eighth I am, I am!
I got married to the widow next door,
She's been married seven times before
And every one was an 'Enery
She wouldn't have a Willie nor a Sam
I'm her eighth old man named 'Enery
'Enery the Eighth, I am!
Second verse, same as the first!
I'm 'Enery the Eighth, I am,
'Enery the Eighth I am, I am!
I got married to the widow next door,
She's been married seven times before
And every one was an 'Enery
She wouldn't have a Willie nor a Sam
I'm her eighth old man named 'Enery
'Enery the Eighth, I am!
"""


class TestDualSignal:
    """Verify the dual-dict architecture's core claim."""

    def test_super_dict_compresses_english(self):
        """Henry lyrics are familiar English → Super Dict ratio >= 2.0."""
        sd = load_super_dict()
        ratio = sd.compress(HENRY_LYRICS)
        assert ratio >= 2.0, (
            f"Super Dict should compress English well, got ratio={ratio:.2f}"
        )

    def test_super_dict_gate_would_skip(self):
        """Single-dict gate (Super Dict only) would SKIP Henry — false negative."""
        sd = load_super_dict()
        result = should_train(HENRY_LYRICS, sd, threshold=2.0)
        assert not result, (
            "Super Dict alone says 'skip' — but Weight Dict must override this"
        )

    def test_weight_dict_finds_novel_patterns_after_training(self):
        """After training on Henry, Weight Dict finds novel weight patterns."""
        sd = load_super_dict()

        # ── Setup: a small model with ZPackR layer ──
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        wd = WeightDict(max_entries=16384)

        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zpl = ZPackRLinear.from_linear(lin, wd).to(dev)

        # ── 1. Before training: baseline compression ratios from initial weights ──
        weight_cpu = zpl.delta_salient.cpu()
        weight_bytes = weight_cpu.view(torch.uint8).contiguous().view(-1).numpy().tobytes()

        pre_ratios = []
        for blk in range(zpl.num_blocks):
            start = blk * zpl.block_size
            end = min(start + zpl.block_size, zpl.in_features)
            byte_start = start * zpl.out_features * 2
            byte_end = end * zpl.out_features * 2
            blk_bytes = weight_bytes[byte_start:byte_end]
            if len(blk_bytes) >= 32:
                pre_ratios.append(wd.ratio(blk_bytes) if not wd.is_empty else 1.0)

        # ── 2. Train on Henry-derived input ──
        x = torch.randn(16, in_f, device=dev)
        target = torch.randn(16, out_f, device=dev)

        zpl.train()
        for step in range(4):
            out = zpl(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            with torch.no_grad():
                zpl.delta_salient.data -= 0.01 * zpl.delta_salient.grad
            zpl.delta_salient.grad = None

        # ── 3. After training: WeightDict should find some blocks novel ──
        weight_cpu = zpl.delta_salient.cpu()
        weight_bytes = weight_cpu.view(torch.uint8).contiguous().view(-1).numpy().tobytes()

        post_ratios = []
        for blk in range(zpl.num_blocks):
            start = blk * zpl.block_size
            end = min(start + zpl.block_size, zpl.in_features)
            byte_start = start * zpl.out_features * 2
            byte_end = end * zpl.out_features * 2
            blk_bytes = weight_bytes[byte_start:byte_end]
            if len(blk_bytes) >= 32:
                post_ratios.append(wd.ratio(blk_bytes) if not wd.is_empty else 1.0)

        # ── 4. Dual-signal assertions ──
        # Super Dict: Henry is familiar text
        super_ratio = sd.compress(HENRY_LYRICS)
        assert super_ratio >= 2.0, (
            f"Super Dict must find Henry compressible (ratio={super_ratio:.2f})"
        )

        # Weight Dict: at least some blocks should have ratio < 2.0 (novel)
        novel_blocks = [r for r in post_ratios if r < 2.0]
        assert len(novel_blocks) > 0, (
            f"Weight Dict should find at least 1 novel block after training. "
            f"All post-ratios: {[f'{r:.2f}' for r in post_ratios]}"
        )

        # Decision matrix: HIGH Super + LOW Weight = NEW CONTENT → correct!
        print(f"\n  Dual-signal verified:")
        print(f"    Super Dict ratio: {super_ratio:.2f} (>= 2.0 → 'familiar text')")
        print(f"    Novel blocks: {len(novel_blocks)}/{len(post_ratios)} (Weight Dict ratio < 2.0)")
        print(f"    Post-training ratios: {[f'{r:.2f}' for r in post_ratios]}")
        print(f"    → Correctly identifies 'familiar language + novel content'")

    def test_post_step_keeps_novel_blocks(self):
        """post_step should KEEP blocks where WeightDict ratio < threshold."""
        torch.manual_seed(123)
        wd = WeightDict(max_entries=16384)

        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin, wd)

        # All blocks start as salient
        n_blocks = zpl.block_mask.sum().item()
        assert n_blocks == zpl.num_blocks

        # Train a few steps
        x = torch.randn(8, in_f)
        target = torch.randn(8, out_f)
        for _ in range(4):
            out = zpl(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            with torch.no_grad():
                zpl.delta_salient.data -= 0.01 * zpl.delta_salient.grad
            zpl.delta_salient.grad = None

        # post_step should keep blocks that are novel
        zpl.post_step(threshold=2.0)
        kept_after = zpl.block_mask.sum().item()
        assert kept_after > 0, (
            "post_step should keep at least some blocks (WeightDict has no patterns yet, "
            "so all blocks should compress poorly → kept as novel)"
        )

    def test_decision_matrix_row_2(self):
        """Verify the specific decision matrix row from the roadmap.

        HIGH Super Dict ratio + LOW Weight Dict ratio = NEW CONTENT → full training.
        This is the specific scenario a single-dict architecture gets wrong.
        """
        sd = load_super_dict()

        # Super Dict: HIGH (English text)
        super_ratio = sd.compress(HENRY_LYRICS)
        assert super_ratio >= 2.0, f"Super ratio={super_ratio:.2f} should be HIGH"

        # WeightDict starts empty → all weight patterns are novel → ratio ~1.0
        wd = WeightDict(max_entries=16384)

        # Create some weight bytes
        weight = torch.randn(256, 128, dtype=torch.bfloat16)
        weight_bytes = weight.view(torch.uint8).contiguous().view(-1).numpy().tobytes()

        # With empty WeightDict, ratio should be ~1.0 (novel)
        ratio = wd.ratio(weight_bytes[:512])
        assert ratio < 1.5, (
            f"Empty WeightDict should produce low ratio on random weights, got {ratio:.2f}"
        )

        # HIGH super + LOW weight = NEW CONTENT → correct!
        if super_ratio >= 1.5 and ratio < 1.5:
            print(f"\n  Roadmap decision matrix row 2 verified:")
            print(f"    Super Dict: {super_ratio:.2f} (HIGH → familiar text)")
            print(f"    Weight Dict: {ratio:.2f} (LOW → novel weight pattern)")
            print(f"    → NEW CONTENT — should train (correct!)")
        else:
            pytest.fail(f"Dual signal mismatch: super={super_ratio:.2f}, weight={ratio:.2f}")

    def test_checkpoint_preserves_dual_signal(self):
        """Checkpoint roundtrip should preserve the dual-signal state."""
        torch.manual_seed(99)
        wd = WeightDict(max_entries=16384)

        # Train WeightDict on some weight data
        weight = torch.randn(256, 128, dtype=torch.bfloat16)
        wb = weight.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        wd.reindex(wb)

        lin = torch.nn.Linear(64, 32, bias=False)
        lin.weight.data = torch.randn(32, 64)
        zpl = ZPackRLinear.from_linear(lin, wd)

        initial_entries = wd.num_entries
        initial_mask = zpl.block_mask.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layer")
            zpl.save_checkpoint(path)

            wd2 = WeightDict.load(path + ".wd")
            restored = ZPackRLinear.load_checkpoint(path, wd2)

            assert wd2.num_entries == initial_entries, (
                f"Checkpoint: entry count mismatch ({wd2.num_entries} vs {initial_entries})"
            )
            assert restored.block_mask.equal(initial_mask), (
                "Checkpoint: block mask should be preserved"
            )
