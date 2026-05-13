"""Tests for zstd_dict.py — WeightDict (adaptive zstd dictionary v2)."""

import os
import tempfile
import torch
import pytest
from zpackr.zstd_dict import WeightDict


class TestWeightDict:
    @pytest.fixture
    def wd_empty(self):
        return WeightDict(max_entries=16384)

    @pytest.fixture
    def wd_trained(self):
        wd = WeightDict(max_entries=16384)
        weight = torch.randn(256, 128, dtype=torch.bfloat16)
        wb = weight.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        wd.reindex(wb)
        return wd

    def test_compress_decompress_roundtrip(self, wd_empty):
        data = os.urandom(1024)
        compressed = wd_empty.compress(data)
        decompressed = wd_empty.decompress(compressed)
        assert data == decompressed, "Roundtrip should be lossless"

    def test_compress_decompress_with_trained_dict(self, wd_trained):
        data = os.urandom(1024)
        compressed = wd_trained.compress(data)
        decompressed = wd_trained.decompress(compressed)
        assert data == decompressed, "Roundtrip with trained dict should be lossless"

    def test_ratio_random(self, wd_empty):
        random_bytes = os.urandom(4096)
        ratio = wd_empty.ratio(random_bytes)
        assert ratio < 1.5, f"Random bytes ratio should be near 1.0, got {ratio:.2f}"

    def test_reindex_produces_entries(self):
        wd = WeightDict(max_entries=16384)
        weight = torch.randn(512, 256, dtype=torch.bfloat16)
        wb = weight.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        entries = wd.reindex(wb)
        assert entries > 0, f"Reindex should produce entries, got {entries}"

    def test_reindex_on_bert_scale_data(self):
        wd = WeightDict(max_entries=16384)
        weight = torch.randn(768, 3072, dtype=torch.bfloat16)
        wb = weight.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
        entries = wd.reindex(wb)
        assert entries > 0, f"BERT-scale reindex should produce entries, got {entries}"
        assert entries <= 16384, "Should not exceed max entries"

    def test_save_load_preserves_compression(self, wd_trained):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wd")
            wd_trained.save(path)
            restored = WeightDict.load(path)

            assert restored.num_entries == wd_trained.num_entries

            test_data = os.urandom(1024)
            c_orig = wd_trained.compress(test_data)
            c_rest = restored.compress(test_data)
            assert restored.decompress(c_rest) == test_data

    def test_empty_dict_works(self, wd_empty):
        assert wd_empty.is_empty
        assert wd_empty.num_entries == 0
        data = os.urandom(1024)
        c = wd_empty.compress(data)
        assert wd_empty.decompress(c) == data
