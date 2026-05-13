"""Tests for checkpoint.py — model-level save/load."""

import os
import tempfile
import torch
import pytest
from packr import PackRConfig, compress_model
from zpackr.checkpoint import save_zpackr_checkpoint, load_zpackr_checkpoint


class TestCheckpoint:
    @pytest.fixture
    def zpackr_model(self, device):
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32, bias=False),
            torch.nn.Linear(32, 16, bias=False),
        )
        config = PackRConfig(mode="zpackr", layer_scope="all")
        model = compress_model(model, config)
        if device.type == "cuda":
            model = model.cuda()
        return model

    def test_model_checkpoint_roundtrip(self, zpackr_model, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            x = torch.randn(8, 64, device=device)
            out_before = zpackr_model(x)

            save_zpackr_checkpoint(zpackr_model, tmpdir)
            load_zpackr_checkpoint(zpackr_model, tmpdir)

            out_after = zpackr_model(x)
            diff = (out_before - out_after).abs().max().item()
            assert diff < 0.5, f"Checkpoint roundtrip diff {diff:.4f}"
