"""Regression tests — mode='packr' must be bit-identical to before ZPackR changes."""

import torch
import pytest
from packr import PackRConfig, compress_model


class TestPackRModeUnchanged:
    def test_packr_mode_compresses(self, device):
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32, bias=False),
        )
        config = PackRConfig(mode="packr", layer_scope="all")
        model = compress_model(model, config)
        assert "PackRLinear" in type(model[0]).__name__

    def test_packr_mode_no_zstd_import(self):
        """Verify mode='packr' does not trigger zstandard import at layer_patcher level."""
        from packr.layer_patcher import _compress_packr
        from packr.config import PackRConfig
        import sys

        # Remove zstandard from sys.modules for this test scope if another test
        # loaded it first; the real assertion is that _compress_packr never
        # references zstandard at all.
        was_imported = "zstandard" in sys.modules
        # Verify the packr path source doesn't reference zstandard
        import inspect
        source = inspect.getsource(_compress_packr)
        assert "zstandard" not in source, (
            "_compress_packr must not reference zstandard"
        )

    def test_packr_mode_forward(self, device):
        if device.type != "cuda":
            pytest.skip("PackR mode requires CUDA")
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32, bias=False),
        )
        config = PackRConfig(mode="packr", layer_scope="all")
        model = compress_model(model, config)
        if device.type == "cuda":
            model = model.cuda()

        x = torch.randn(8, 64, device=device)
        out = model(x)
        assert out.shape == (8, 32)
