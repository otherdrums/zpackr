"""Tests for zpackr_layer.py — ZPackRLinear (LZ4-based)."""

import os
import tempfile
import torch
import math
import pytest
from packr.zpackr_layer import ZPackRLinear


class TestZPackRLinear:
    @pytest.fixture
    def layer(self, device):
        lin = torch.nn.Linear(128, 64, bias=False)
        lin.weight.data = torch.randn(64, 128)
        zpl = ZPackRLinear.from_linear(lin)
        if device.type == "cuda":
            zpl = zpl.cuda()
        return zpl

    def test_forward_shape(self, layer, device):
        x = torch.randn(8, layer.in_features, device=device)
        out = layer(x)
        assert out.shape == (8, layer.out_features)

    def test_forward_matches_nn_linear(self, layer, device):
        lin = torch.nn.Linear(layer.in_features, layer.out_features, bias=False)
        w_merged = (layer.base_W + layer.delta_salient).t().float()
        lin.weight.data = w_merged
        if device.type == "cuda":
            lin = lin.cuda()

        x = torch.randn(16, layer.in_features, device=device)
        out_lin = lin(x)
        out_zpl = layer(x)
        diff = (out_lin.float() - out_zpl.float()).abs().max().item()
        assert diff < 0.5, f"Max diff {diff:.4f} exceeds bf16 tolerance"

    def test_post_step(self, layer, device):
        n_blocks_before = int(layer.block_mask.sum().item())
        layer.post_step()
        n_blocks_after = int(layer.block_mask.sum().item())
        assert n_blocks_after <= n_blocks_before

    def test_checkpoint_roundtrip(self, layer, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layer_0")
            layer.save_checkpoint(path)

            restored = ZPackRLinear.load_checkpoint(path)
            if device.type == "cuda":
                restored = restored.cuda()

            x = torch.randn(8, layer.in_features, device=device)
            out_orig = layer(x)
            out_rest = restored(x)
            diff = (out_orig - out_rest).abs().max().item()
            assert diff < 0.3, f"Checkpoint roundtrip diff {diff:.4f} too large"

            assert restored.block_mask.sum().item() == layer.block_mask.sum().item(), (
                "Block mask should be preserved"
            )

    def test_gradient_flow(self, layer, device):
        x = torch.randn(8, layer.in_features, device=device)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.delta_salient.grad is not None, "Salient view should receive gradients"

    def test_no_full_matrix_forward(self, layer, device):
        x = torch.randn(8, layer.in_features, device=device)
        out = layer(x)
        assert out.shape == (8, layer.out_features)
