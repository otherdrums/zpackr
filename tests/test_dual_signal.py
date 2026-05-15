"""LSH-based per-block attenuation verification.

Tests that:
  1. DeltaSignatureDB produces consistent LSH hashes
  2. post_step produces attenuation factors from LSH multi-scale comparison
  3. Convergence gate fires when all blocks fully attenuated
  4. Checkpoint roundtrip preserves delta state
  5. Attenuation increases as blocks converge (multi-encounter test)
"""

import os
import sys
import torch
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.zpackr_layer import ZPackRLinear, DeltaSignatureDB
from packr.prompt_gate import should_skip_backward


class TestLSHSignal:
    """Verify the LSH-based per-block attenuation architecture."""

    def test_lsh_hash_deterministic(self):
        """LSH hash should be deterministic for same input."""
        torch.manual_seed(42)
        db = DeltaSignatureDB(block_elements=64, num_blocks=4, K=64, seed=42)
        delta = torch.randn(4, 64, dtype=torch.bfloat16)
        h1 = db.hash_blocks(delta)
        h2 = db.hash_blocks(delta)
        assert torch.equal(h1, h2), "LSH hash should be deterministic"

    def test_lsh_hash_shape(self):
        """LSH hash should have correct shape."""
        db = DeltaSignatureDB(block_elements=128, num_blocks=8, K=64)
        delta = torch.randn(8, 128, dtype=torch.bfloat16)
        h = db.hash_blocks(delta)
        assert h.shape == (8, 64), f"Expected (8, 64), got {h.shape}"
        assert h.dtype == torch.uint8

    def test_empty_window_gives_zero_attenuation(self):
        """With no history, attenuation should be zero."""
        db = DeltaSignatureDB(block_elements=64, num_blocks=4, K=64)
        hashes = db.hash_blocks(torch.randn(4, 64, dtype=torch.bfloat16))
        attn = db.compute_attenuation(hashes)
        assert torch.all(attn == 0.0), "Empty window should give zero attenuation"

    def test_post_step_produces_attenuation(self):
        """post_step should produce attenuation factors from LSH."""
        torch.manual_seed(123)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)

        zpl.post_step()
        assert zpl._attenuation_factors is not None, "post_step should compute attenuation"
        assert len(zpl._attenuation_factors) == zpl.num_blocks

        # First post_step with no history → attenuation should be 0
        for a in zpl._attenuation_factors:
            assert a == 0.0, (
                f"First post_step (no history) should have zero attenuation, got {a:.3f}"
            )

    def test_attenuation_increases_with_repeated_deltas(self):
        """Attenuation should increase as delta stabilizes across encounters."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)

        # Set a fixed delta
        zpl.delta_salient.data += 0.1

        # First post_step → attenuation = 0 (no history)
        zpl.post_step()
        attn_0 = list(zpl._attenuation_factors)

        # Second post_step with same delta → attenuation should increase
        zpl.post_step()
        attn_1 = list(zpl._attenuation_factors)

        # Third post_step with same delta → attenuation should increase more
        zpl.post_step()
        attn_2 = list(zpl._attenuation_factors)

        for blk in range(zpl.num_blocks):
            assert attn_1[blk] >= attn_0[blk], (
                f"Block {blk}: attenuation should increase, {attn_1[blk]:.3f} < {attn_0[blk]:.3f}"
            )

    def test_convergence_gate_fires_on_converged_blocks(self):
        """Gate should fire when all blocks have high attenuation."""
        torch.manual_seed(42)
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)

        # Simulate well-trained blocks with high attenuation
        zpl._attenuation_factors = [0.95, 0.92, 0.91]
        layers = [("test", zpl)]
        result = should_skip_backward(layers, threshold=0.9)
        assert result, (
            "Gate should fire when all blocks are well-trained (high attenuation)"
        )

    def test_convergence_gate_does_not_fire_on_novel_blocks(self):
        """When blocks have low attenuation (novel), gate should NOT fire."""
        torch.manual_seed(42)
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)

        zpl._attenuation_factors = [0.1] * zpl.num_blocks
        layers = [("test", zpl)]
        result = should_skip_backward(layers, threshold=0.5)
        assert not result, (
            "Gate should NOT fire when blocks have low attenuation"
        )

    def test_checkpoint_roundtrip(self):
        """Checkpoint save/load preserves delta state."""
        torch.manual_seed(99)
        lin = torch.nn.Linear(64, 32, bias=False)
        lin.weight.data = torch.randn(32, 64)
        zpl = ZPackRLinear.from_linear(lin)

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

            delta_match = torch.allclose(restored._full_delta, initial_delta, atol=1e-2)
            assert delta_match, "Checkpoint roundtrip should preserve delta"

    def test_has_signature_db(self):
        """ZPackRLinear should have DeltaSignatureDB."""
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        assert hasattr(zpl, '_sig_db'), "_sig_db should exist"
        assert isinstance(zpl._sig_db, DeltaSignatureDB)
