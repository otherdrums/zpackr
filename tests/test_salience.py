"""Tests for salience.py — block salience computation."""

import torch
import math
from zpackr.salience import compute_salience, block_mask_to_indices, num_salient_blocks


class TestComputeSalience:
    def test_random_weights_all_salient(self, weight_dict):
        weight = torch.randn(1024, 128, dtype=torch.bfloat16)
        mask = compute_salience(weight, weight_dict, block_size=256, threshold=2.0)
        num_blocks = math.ceil(1024 * 128 / 256)
        assert len(mask) == num_blocks, f"Mask length should be {num_blocks}, got {len(mask)}"
        assert mask.sum() > 0, "At least some blocks should be salient"

    def test_mask_cardinality(self, weight_dict):
        in_f, out_f = 512, 256
        weight = torch.randn(in_f, out_f, dtype=torch.bfloat16)
        mask = compute_salience(weight, weight_dict, block_size=256, threshold=2.0)
        expected = math.ceil(in_f * out_f / 256)
        assert len(mask) == expected

    def test_block_mask_to_indices(self, weight_dict):
        weight = torch.randn(768, 256, dtype=torch.bfloat16)
        mask = compute_salience(weight, weight_dict, block_size=256, threshold=2.0)
        indices = block_mask_to_indices(mask, block_size=256)
        assert indices.shape[1] == 2
        assert indices.shape[0] == num_salient_blocks(mask)
