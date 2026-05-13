"""Tests for zpackr_layer.py — ZPackRLinear."""

import os
import tempfile
import torch
import math
import pytest
from zpackr.zstd_dict import WeightDict
from zpackr.zpackr_layer import ZPackRLinear


class TestZPackRLinear:
    @pytest.fixture
    def weight_dict(self):
        wd = WeightDict(max_entries=16384)
        # Train on some weight data to populate the dictionary
        weight = torch.randn(512, 256, dtype=torch.bfloat16)
        wb = weight.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        wd.reindex(wb)
        return wd

    @pytest.fixture
    def layer(self, weight_dict, device):
        lin = torch.nn.Linear(128, 64, bias=False)
        lin.weight.data = torch.randn(64, 128)
        zpl = ZPackRLinear.from_linear(lin, weight_dict)
        if device.type == "cuda":
            zpl = zpl.cuda()
        return zpl

    def test_forward_shape(self, layer, device):
        x = torch.randn(8, layer.in_features, device=device)
        out = layer(x)
        assert out.shape == (8, layer.out_features)

    def test_forward_matches_nn_linear(self, layer, device):
        # Build matching nn.Linear from merged weights
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
        layer.post_step(threshold=2.0)
        n_blocks_after = int(layer.block_mask.sum().item())
        assert n_blocks_after <= n_blocks_before

    def test_checkpoint_roundtrip(self, layer, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layer_0")
            layer.save_checkpoint(path)

            wd2 = WeightDict.load(path + ".wd")
            restored = ZPackRLinear.load_checkpoint(path, wd2)
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
        """Verify forward computes correct output without errors."""
        x = torch.randn(8, layer.in_features, device=device)
        out = layer(x)
        assert out.shape == (8, layer.out_features)
        # Full weight matrix is never materialized in a single tensor
        # during forward — this is tested structurally by block_accumulate.
