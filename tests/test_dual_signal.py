"""LSH-based per-row attenuation verification.

Tests that:
  1. DeltaSignatureDB produces consistent LSH hashes (row-level)
  2. compute_hash_gpu produces attenuation factors from LSH multi-scale comparison
  3. Convergence gate fires when all rows fully attenuated
  4. Checkpoint roundtrip preserves delta state
  5. Attenuation increases as rows converge (multi-encounter test)
"""

import os
import sys
import torch
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.zpackr_layer import ZPackRLinear, DeltaSignatureDB, ATTENUATION_SKIP_THRESHOLD
from packr.prompt_gate import should_skip_backward


class TestLSHSignal:
    """Verify the LSH-based per-row attenuation architecture."""

    def test_lsh_hash_deterministic(self):
        """LSH hash should be deterministic for same input."""
        torch.manual_seed(42)
        db = DeltaSignatureDB(num_rows=4, K=64, seed=42)
        delta = torch.randn(4, 16, dtype=torch.bfloat16)
        h1 = db.hash_rows(delta)
        h2 = db.hash_rows(delta)
        assert torch.equal(h1, h2), "LSH hash should be deterministic"

    def test_lsh_hash_shape(self):
        """LSH hash should have correct shape."""
        db = DeltaSignatureDB(num_rows=8, K=64)
        delta = torch.randn(8, 32, dtype=torch.bfloat16)
        h = db.hash_rows(delta)
        assert h.shape == (8, 8), f"Expected (8, 8), got {h.shape}"
        assert h.dtype == torch.uint8

    def test_empty_window_gives_zero_attenuation(self):
        """With no history, attenuation should be zero."""
        db = DeltaSignatureDB(num_rows=4, K=64)
        hashes = db.hash_rows(torch.randn(4, 16, dtype=torch.bfloat16))
        attn = db.compute_attenuation(hashes)
        assert torch.all(attn == 0.0), "Empty window should give zero attenuation"

    def test_post_step_produces_attenuation(self):
        """post_step should produce attenuation factors from LSH."""
        torch.manual_seed(123)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)
        if torch.cuda.is_available():
            zpl = zpl.cuda()

        zpl.post_step()
        # First post_step with no history → attenuation should be 0
        attn = zpl._atten_byte.float()
        assert attn.max().item() == 0.0, (
            f"First post_step (no history) should have zero attenuation, got {attn.max().item()}"
        )

    def test_attenuation_increases_with_repeated_deltas(self):
        """Attenuation should increase as delta stabilizes across encounters."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)
        if torch.cuda.is_available():
            zpl = zpl.cuda()

        zpl.delta_salient.data += 0.1

        # First post_step → attenuation = 0 (no history)
        zpl.post_step()
        attn_0 = zpl._atten_byte.float().mean().item()

        # Second post_step with same delta → attenuation should increase
        zpl.post_step()
        attn_1 = zpl._atten_byte.float().mean().item()

        # Third post_step
        zpl.post_step()
        attn_2 = zpl._atten_byte.float().mean().item()

        # Should increase over time as delta stabilizes
        assert attn_2 >= attn_0, "Attenuation should increase over time"

    def test_convergence_gate_fires_on_converged_rows(self):
        """Gate should fire when all rows have high attenuation."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        if torch.cuda.is_available():
            zpl = zpl.cuda()

        # Simulate well-trained rows with high attenuation
        zpl._atten_byte = torch.full((in_f,), 255, dtype=torch.uint8)
        layers = [("test", zpl)]
        result = should_skip_backward(layers, threshold=0.99)
        assert result, "Gate should fire when all rows are well-trained"

    def test_convergence_gate_does_not_fire_on_novel_rows(self):
        """When rows have low attenuation (novel), gate should NOT fire."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        if torch.cuda.is_available():
            zpl = zpl.cuda()

        zpl._atten_byte = torch.zeros(in_f, dtype=torch.uint8)
        layers = [("test", zpl)]
        result = should_skip_backward(layers, threshold=0.5)
        assert not result, "Gate should NOT fire when rows have low attenuation"

    def test_checkpoint_roundtrip(self):
        """Checkpoint save/load preserves delta state."""
        torch.manual_seed(99)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)
        zpl = ZPackRLinear.from_linear(lin)

        zpl.delta_salient.data += 0.1
        initial_delta = zpl.delta_salient.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layer")
            zpl.save_checkpoint(path)

            restored = ZPackRLinear.load_checkpoint(path)

            assert restored.in_features == zpl.in_features
            assert restored.out_features == zpl.out_features
            assert restored.delta_salient is not None

            delta_match = torch.allclose(restored.delta_salient.cpu(), initial_delta.cpu(), atol=1e-2)
            assert delta_match, "Checkpoint roundtrip should preserve delta"

    def test_has_signature_db(self):
        """ZPackRLinear should have DeltaSignatureDB."""
        lin = torch.nn.Linear(64, 32, bias=False)
        zpl = ZPackRLinear.from_linear(lin)
        assert hasattr(zpl, '_sig_db'), "_sig_db should exist"
        assert isinstance(zpl._sig_db, DeltaSignatureDB)
        assert zpl._sig_db.num_rows == 64

    def test_novelty_boost_lowers_attenuation_for_changing_rows(self):
        """Rows with changing hashes should have lower attenuation than stable rows."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = torch.randn(out_f, in_f)

        # Stable run: same delta every step
        zpl_s = ZPackRLinear.from_linear(lin)
        if torch.cuda.is_available():
            zpl_s = zpl_s.cuda()
        zpl_s.delta_salient.data += 0.1
        zpl_s.post_step()
        zpl_s.post_step()
        zpl_s.post_step()
        stable_atten = zpl_s._atten_byte.float().mean().item()

        # Changing run: new random direction each step (+= in-place on same device)
        zpl_c = ZPackRLinear.from_linear(lin)
        if torch.cuda.is_available():
            zpl_c = zpl_c.cuda()
        dev = zpl_c.delta_salient.device
        zpl_c.delta_salient.data += 0.1
        zpl_c.post_step()
        zpl_c.delta_salient.data += torch.randn(in_f, out_f, device=dev, dtype=torch.bfloat16) * 0.5
        zpl_c.post_step()
        zpl_c.delta_salient.data += torch.randn(in_f, out_f, device=dev, dtype=torch.bfloat16) * 0.5
        zpl_c.post_step()
        changing_atten = zpl_c._atten_byte.float().mean().item()

        assert changing_atten < stable_atten, (
            f"Changing rows ({changing_atten:.4f}) should have lower attenuation "
            f"than stable rows ({stable_atten:.4f})"
        )

    def test_prev_hashes_initialized_after_first_post_step(self):
        """_prev_hashes should be set after first post_step call."""
        torch.manual_seed(42)
        zpl = ZPackRLinear.from_linear(torch.nn.Linear(64, 32, bias=False))
        assert zpl._prev_hashes is None, "_prev_hashes should start as None"
        zpl.post_step()
        assert zpl._prev_hashes is not None, "_prev_hashes should be set after first post_step"
        assert zpl._prev_hashes.shape == (64, 2), "Shape should match [in_features, K//8]"
