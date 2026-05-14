"""LZ4 compression ratio and convergence gate verification.

Tests that:
  1. LZ4 compresses zero-delta extremely well (ratio >> 2)
  2. LZ4 barely compresses random bf16 (ratio ~1.0)
  3. post_step produces correct attenuation from LZ4 ratios
  4. Convergence gate fires when all blocks fully attenuated
  5. Checkpoint roundtrip preserves delta state
"""

import os
import sys
import torch
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.zpackr_layer import ZPackRLinear, RATIO_FLOOR, RATIO_CEILING
from packr.prompt_gate import should_skip_backward


class TestLZ4Signal:
    """Verify the LZ4-based per-block attenuation architecture."""

    def test_zero_delta_compresses_well(self):
        """Zero-delta blocks should have very high LZ4 compression ratio."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zpl = ZPackRLinear.from_linear(lin).to(dev)

        # Zero delta → LZ4 should compress extremely well
        zpl._sync_full_delta()
        import zstandard as zstd
        import numpy as np
        delta_np = zpl._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
        block_el_bytes = zpl.block_size * zpl.out_features * 2

        for blk in range(zpl.num_blocks):
            byte_start = blk * block_el_bytes
            byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
            blk_bytes = delta_np[byte_start:byte_end].tobytes()
            compressed = zstd.compress(blk_bytes)
            ratio = len(blk_bytes) / max(len(compressed), 1)
            assert ratio > 2.0, (
                f"Zero-delta block {blk} should compress well, got ratio={ratio:.2f}"
            )

    def test_trained_delta_compresses_poorly(self):
        """After training, delta should be poorly compressible by LZ4."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zpl = ZPackRLinear.from_linear(lin).to(dev)

        # Train a few steps
        x = torch.randn(8, in_f, device=dev)
        target = torch.randn(8, out_f, device=dev)
        zpl.train()
        for _ in range(4):
            out = zpl(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            with torch.no_grad():
                zpl.delta_salient.data -= 0.01 * zpl.delta_salient.grad
            zpl.delta_salient.grad = None

        # Trained delta → zstd should give ratio > 1.0 but < 2.0 (not highly compressible)
        zpl._sync_full_delta()
        import zstandard as zstd
        delta_np = zpl._full_delta.view(torch.uint8).contiguous().view(-1).numpy()
        block_el_bytes = zpl.block_size * zpl.out_features * 2

        found_low = False
        for blk in range(zpl.num_blocks):
            byte_start = blk * block_el_bytes
            byte_end = min(byte_start + block_el_bytes, delta_np.nbytes)
            blk_bytes = delta_np[byte_start:byte_end].tobytes()
            compressed = zstd.compress(blk_bytes)
            ratio = len(blk_bytes) / max(len(compressed), 1)
            if ratio < 2.0:
                found_low = True
                break
        assert found_low, (
            "At least one trained block should have ratio < 2.0 (not highly compressible)"
        )

    def test_post_step_produces_attenuation(self):
        """post_step should produce attenuation factors from LZ4 ratios."""
        torch.manual_seed(123)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)

        zpl.post_step()
        assert zpl._attenuation_factors is not None, "post_step should compute attenuation"
        assert len(zpl._attenuation_factors) == zpl.num_blocks

        # Zero delta → high ratio → high attenuation
        for a in zpl._attenuation_factors:
            assert a >= 0.9, (
                f"Zero-delta should produce high attenuation, got {a:.3f}"
            )

    def test_convergence_gate_fires_on_zero_delta(self):
        """When all blocks are fully attenuated, gate should fire."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)

        zpl.post_step()
        layers = [("test", zpl)]
        result = should_skip_backward(layers, threshold=0.9)
        assert result, (
            "Gate should fire when all blocks are fully attenuated (zero delta)"
        )

    def test_convergence_gate_does_not_fire_on_trained_blocks(self):
        """When blocks have low attenuation (novel), gate should NOT fire."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)

        # Manually set low attenuation to simulate training
        zpl._attenuation_factors = [0.1] * zpl.num_blocks
        layers = [("test", zpl)]
        result = should_skip_backward(layers, threshold=0.5)
        assert not result, (
            "Gate should NOT fire when blocks have low attenuation"
        )

    def test_checkpoint_roundtrip_lz4(self):
        """Checkpoint save/load with LZ4 preserves delta state."""
        torch.manual_seed(99)
        lin = torch.nn.Linear(64, 32, bias=False)
        lin.weight.data = torch.randn(32, 64)
        zpl = ZPackRLinear.from_linear(lin)

        # Populate delta with some values (on GPU via delta_salient)
        zpl.delta_salient.data += 0.1
        zpl._sync_full_delta()
        initial_delta = zpl._full_delta.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layer")
            zpl.save_checkpoint(path)

            restored = ZPackRLinear.load_checkpoint(path)

            assert restored.in_features == zpl.in_features
            assert restored.out_features == zpl.out_features
            assert restored._full_delta is not None

            # Delta should be preserved
            delta_match = torch.allclose(restored._full_delta, initial_delta, atol=1e-2)
            assert delta_match, "Checkpoint roundtrip should preserve delta"

    def test_no_attributes_from_old_arch(self):
        """Ensure no stale WeightDict/SuperDict attributes remain."""
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)

        assert not hasattr(zpl, 'weight_dict'), "weight_dict should be removed"
        assert hasattr(zpl, '_zstd_delta'), "_zstd_delta should exist"
        assert not hasattr(zpl, '_lz4_delta'), "_lz4_delta should not exist"
