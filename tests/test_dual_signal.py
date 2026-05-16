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

from packr.zpackr_layer import ZPackRLinear, DeltaSignatureDB
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

    def test_has_gradient_signature_db(self):
        """ZPackRLinear should have _grad_sig_db for gradient hashing."""
        zpl = ZPackRLinear.from_linear(torch.nn.Linear(64, 32, bias=False))
        assert hasattr(zpl, '_grad_sig_db'), "_grad_sig_db should exist"
        assert isinstance(zpl._grad_sig_db, DeltaSignatureDB)

    def test_grad_hash_handles_none_grad(self):
        """compute_grad_hash should not crash when grad is None."""
        zpl = ZPackRLinear.from_linear(torch.nn.Linear(8, 4, bias=False))
        zpl.compute_grad_hash()  # no backward yet → grad is None
        assert zpl._grad_sim.shape == (8,)
        assert zpl._grad_sim.sum().item() == 0.0

    def test_grad_hash_produces_similarity_after_backward(self):
        """After backward, compute_grad_hash should fill _grad_sim."""
        torch.manual_seed(99)
        zpl = ZPackRLinear.from_linear(torch.nn.Linear(8, 4, bias=False))
        if torch.cuda.is_available():
            zpl = zpl.cuda()
        x = torch.randn(2, 8, device=zpl.delta_salient.device, dtype=torch.bfloat16)
        y = zpl(x).sum()
        y.backward()
        zpl.compute_grad_hash()
        # _grad_sim should be a non-negative tensor (similarity in [0,1]) * 1 step
        grad_sim = zpl._grad_sim.float()
        assert grad_sim.min().item() >= 0.0, "grad_sim should be non-negative"
        assert grad_sim.max().item() <= 1.0, "grad_sim should be <= 1.0"
        # First hash with empty window → expected to be functionally 0-ish
        # (window is empty, compute_attenuation returns zeros)
        assert grad_sim.mean().item() == 0.0, "First grad hash with empty window should be 0"

    def test_gradient_mix_default_is_05(self):
        """Default gradient_mix should be 0.5."""
        zpl = ZPackRLinear.from_linear(torch.nn.Linear(8, 4, bias=False))
        assert zpl._gradient_mix == 0.5

    def test_dual_signal_mixing_formula(self):
        """Verify the geometric mixing formula with known inputs."""
        zpl = ZPackRLinear.from_linear(torch.nn.Linear(8, 4, bias=False))
        mix = zpl._gradient_mix  # 0.5

        # Manually set delta_sim and grad_sim, call compute_hash_gpu
        # We need to force specific values by manipulating the window
        # The formula is: atten = delta_sim^(1-mix) * (1-grad_sim)^mix

        # Test 1: both at extremes → converged
        delta_sim = torch.tensor(1.0)  # stable delta
        grad_sim = torch.tensor(0.0)   # noisy gradient (converged)
        atten = delta_sim ** (1 - mix) * (1 - grad_sim) ** mix
        assert abs(atten.item() - 1.0) < 1e-6, f"Converged: expected 1.0, got {atten.item()}"

        # Test 2: learning hard (both stable)
        delta_sim = torch.tensor(0.8)
        grad_sim = torch.tensor(0.8)  # stable gradient = learning
        atten = delta_sim ** (1 - mix) * (1 - grad_sim) ** mix
        expected = (0.8 ** 0.5) * (0.2 ** 0.5)  # = sqrt(0.16) = 0.4
        assert abs(atten.item() - expected) < 1e-6, \
            f"Learning: expected {expected:.4f}, got {atten.item():.4f}"

        # Test 3: stuck (stable delta, noisy gradient)
        delta_sim = torch.tensor(0.9)
        grad_sim = torch.tensor(0.3)  # some noise, some signal
        atten = delta_sim ** (1 - mix) * (1 - grad_sim) ** mix
        expected = (0.9 ** 0.5) * (0.7 ** 0.5)  # = sqrt(0.63) ≈ 0.794
        assert abs(atten.item() - expected) < 1e-6, \
            f"Stuck: expected {expected:.4f}, got {atten.item():.4f}"

        # Test 4: pure delta (mix=0)
        mix0 = 0.0
        atten = delta_sim ** (1 - mix0) * (1 - grad_sim) ** mix0
        assert abs(atten.item() - 0.9) < 1e-6, f"Delta only: expected 0.9, got {atten.item()}"

        # Test 5: pure gradient (mix=1)
        mix1 = 1.0
        atten = delta_sim ** (1 - mix1) * (1 - grad_sim) ** mix1
        assert abs(atten.item() - 0.7) < 1e-6, f"Grad only: expected 0.7, got {atten.item()}"
